import pandas as pd
import torch
import os
from torch import nn
from sklearn.model_selection import KFold
import datetime
from typing import Callable
from .utils import Accumulator, Animator, backup_script
from .data import make_DataLoader
from ._constants import SUBMISSION_DIR


class RegressionAutoManager:
    """
    适用于2D表格数据的普通回归。
    """

    def __init__(
        self,
        *,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        net_cls: nn.Module,
        net_params: dict,
        criterion_cls: nn.Module,
        criterion_params: dict,
        optimizer_cls: torch.optim.Optimizer,
        optimizer_params: dict,
        device="cuda",
    ):
        """
        建议原来的 Tensor 都在 CPU 上，在需要的时候 Manager 会自动转到 device 上。以减少 VRAM 占用。\n
        Args:
            net_cls: 模型本身的架构，用于作为类在内部函数初始化实例。
        """
        if criterion_params.get("reduction", "mean") != "mean":
            raise ValueError("目前criterion_params中的reduction参数只支持'mean'")
        self.device = device
        self.X_train = X_train  # 之后对每个小批量使用.to(device)，而不是一开始就把所有数据放到device上，减少VRAM占用
        self.y_train = y_train
        self.X_test = X_test
        self.net_cls = net_cls
        self.net_params = net_params
        self.optimizer_cls = optimizer_cls
        self.optimizer_params = optimizer_params
        self.criterion = criterion_cls(**criterion_params)  # model, criterion, optimizer中criterion自始至终不需要改变，所以只需要初始化一次
        self.metric_map = {  # 用于把其他方法参数里面的字符串映射为函数（函数格式：(y_hat, y) -> float，返回某小批量的某指标的总和）
            "loss": lambda y_hat, y: (self.criterion(y_hat, y) * y_hat.shape[0]).item(),
        }

    def exploratory_train(
        self,
        *,
        subset_size: int,
        num_epochs: int,
        metric_names: list[str] = ["loss"],
        ylim: tuple[float, float] = (0, 1),
    ):
        """
        探索性训练，先看看模型能不能在小部分训练集上过拟合
        metric_names: 用于评估模型的表现。如果有自定义的metric，请传入 (y_hat, y) -> float 的函数的名字字符串（"loss", "acc"是内置的特殊名字）
        """
        # 初始化 metric_funcs
        metric_funcs = self._init_metric_funcs(metric_names)
        # 初始化动画
        animator = Animator(
            num_axes=3,
            num_epochs=num_epochs,
            ylim=ylim,
            legend=["train-" + metric_name for metric_name in metric_names],  # lengend 的顺序：train、metric_names
        )  # exploratory里面只有训练集，没有验证集

        # Prepare potentially modified params
        optimizer_params_effective = self.optimizer_params
        if "weight_decay" in self.optimizer_params:
            print("检测到weight_decay，将自动临时视作0以屏蔽该参数")
            optimizer_params_copy = self.optimizer_params.copy()
            optimizer_params_copy["weight_decay"] = 0
            optimizer_params_effective = optimizer_params_copy  # Use the copy
        net_params_effective = self.net_params
        if "dropout_ps" in self.net_params:
            print("检测到dropout_ps，将自动临时视作0以屏蔽该参数")
            net_params_copy = self.net_params.copy()
            net_params_copy["dropout_ps"] = [0 for _ in range(len(net_params_copy["dropout_ps"]))]
            net_params_effective = net_params_copy  # Use the copy

        X_len = len(self.X_train)

        # 进行3次，每次选取一小部分作为训练集。在整个训练集上训练，并验证模型是否能记住所有数据并过拟合整个训练集
        for i in range(3):
            # 初始化 - Use effective params
            model = self.net_cls(**net_params_effective).to(self.device)
            optimizer = self.optimizer_cls(model.parameters(), **optimizer_params_effective)
            # 选取一小部分作为训练集
            randint = torch.randint(0, X_len, size=(subset_size,))
            X_train_subset, y_train_subset = self.X_train[randint].to(self.device), self.y_train[randint].to(self.device)

            for epoch in range(1, num_epochs + 1):
                # 开始本周期的训练（这里的训练不是小批量，而是整个训练集）
                model.train()
                optimizer.zero_grad()
                y_hat = model(X_train_subset)
                l = self.criterion(y_hat, y_train_subset)
                l.backward()
                optimizer.step()

                # 评估本周期的训练成果（以过拟合为佳）
                with torch.no_grad():
                    model.eval()
                    # 在训练集上评估（直接把整个训练集作为一个批量）
                    animator.add(i, epoch, [metric_func(y_hat.detach(), y_train_subset) / subset_size for metric_func in metric_funcs])

    def train_and_eval(
        self,
        *,
        k_folds: int,
        batch_size: int,
        num_epochs: int,
        metric_names: list[str] = ["loss"],
        ylim: tuple[float, float] = (0, 1),
    ):
        """
        训练和评估模型 (K折交叉验证)\n
        metric_names: 用于评估模型的表现。如果有自定义的metric，请传入 (y_hat, y) -> float 的函数的名字字符串（"loss", "acc"是内置的特殊名字）
        """
        # 初始化 metric_funcs
        metric_funcs = self._init_metric_funcs(metric_names)

        # 初始化动画
        animator = Animator(
            num_axes=k_folds,
            num_epochs=num_epochs,
            ylim=ylim,
            legend=["train-" + metric_name for metric_name in metric_names] + ["val-" + metric_name for metric_name in metric_names],  # lengend 的顺序：train上的metric_names val上的metric_names
        )

        # 进行K折交叉验证
        for i, (train_idx, val_idx) in enumerate(KFold(k_folds, shuffle=True, random_state=42).split(self.X_train)):
            # 对于每折初始化模型
            model = self.net_cls(**self.net_params).to(self.device)
            optimizer = self.optimizer_cls(model.parameters(), **self.optimizer_params)
            # 本折的训练集，验证集
            X_train_fold, y_train_fold = self.X_train[train_idx], self.y_train[train_idx]
            X_val_fold, y_val_fold = self.X_train[val_idx], self.y_train[val_idx]
            # 用于for的数据加载器
            train_loader = make_DataLoader(batch_size, False, X_train_fold, y_train_fold)
            val_loader = make_DataLoader(batch_size, False, X_val_fold, y_val_fold)
            # 开始训练
            for epoch in range(1, num_epochs + 1):
                train_accumulator = Accumulator(1 + len(metric_funcs))  # 先后顺序：样本数目，训练集上各个metric的值
                val_accumulator = Accumulator(1 + len(metric_funcs))  # 先后顺序：样本数目，验证集上各个metric的值
                # 本周期的训练
                model.train()
                for X, y in train_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    y_hat = model(X)
                    l = self.criterion(y_hat, y)
                    l.backward()
                    optimizer.step()
                    train_accumulator.add(X.shape[0].item(), *[metric_func(y_hat.detach(), y) for metric_func in metric_funcs])

                # 本周期训练结束，开始在验证集评估
                with torch.no_grad():
                    model.eval()
                    for X, y in val_loader:
                        X, y = X.to(self.device), y.to(self.device)
                        y_hat = model(X)
                        val_accumulator.add(X.shape[0].item(), *[metric_func(y_hat.detach(), y) for metric_func in metric_funcs])

                    # 计算本周期训练集和验证集上的各个平均metric作参考
                    train_metrics = [metric_value / train_accumulator.data[0] for metric_value in train_accumulator.data[1:]]
                    val_metrics = [metric_value / val_accumulator.data[0] for metric_value in val_accumulator.data[1:]]
                    # 过程可视化
                    animator.add(i, epoch, [*train_metrics, *val_metrics])

    def final_train(
        self,
        *,
        batch_size: int,
        num_epochs: int,
        metric_names: list[str] = ["loss"],
        ylim: tuple[float, float] = (0, 1),
    ) -> nn.Module:
        """
        最终训练，返回训练好的模型model
        """
        # 初始化 metric_funcs
        metric_funcs = self._init_metric_funcs(metric_names)

        # 初始化动画
        animator = Animator(
            num_axes=1,
            num_epoch=num_epochs,
            ylim=ylim,
            legend=["train-" + metric_name for metric_name in metric_names],
        )

        # 初始化模型
        model = self.net_cls(**self.net_params).to(self.device)
        optimizer = self.optimizer_cls(model.parameters(), **self.optimizer_params)
        # 用于for的数据加载器
        train_loader = make_DataLoader(batch_size, False, self.X_train, self.y_train)
        # 开始训练
        for epoch in range(1, num_epochs + 1):
            train_accumulator = Accumulator(1 + len(metric_funcs))  # 先后顺序：样本数目，训练集上各个metric的值
            # 本周期的训练
            model.train()
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_hat = model(X)
                l = self.criterion(y_hat, y)
                l.backward()
                optimizer.step()
                train_accumulator.add(X.shape[0].item(), *[metric_func(y_hat.detach(), y) for metric_func in metric_funcs])

            # 本周期的训练结束，开始计算训练集上的各个平均metric作参考
            with torch.no_grad():
                model.eval()
                train_metrics = [metric_value / train_accumulator.data[0] for metric_value in train_accumulator.data[1:]]
                # 过程可视化
                animator.add(0, epoch, train_metrics)

        return model

    def predict(
        self,
        *,
        test_df: pd.DataFrame,
        model: nn.Module,
        pred_col_name: str,
        batch_size: int = 512,
        model_path: str,
        submission_dir: str = SUBMISSION_DIR,
        device="cuda",
    ):
        """
        预测，返回预测结果
        model_path: 传入模型源文件的路径（可以是文件或目录），进行备份。
        """
        os.makedirs(submission_dir, exist_ok=True)
        model = model.to(device)
        model.eval()
        with torch.inference_mode():
            # 分批预测，避免显存不足的尴尬情况
            y_pred = torch.zeros(len(self.X_test), dtype=torch.float32)
            for i in range(0, len(self.X_test), batch_size):
                X_test_batch = self.X_test[i : i + batch_size].to(device)
                y_pred[i : i + batch_size] = model(X_test_batch).squeeze()

        # 转换成DataFrame并保存为csv
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        pred_series = pd.Series(y_pred.detach().cpu().numpy(), name=pred_col_name)
        df = pd.concat([test_df.iloc[:, 0], pred_series], axis=1)
        output_path = os.path.join(submission_dir, f"{model.__class__.__name__}_{timestamp}_pred.csv")
        df.to_csv(output_path, index=False)
        # 收尾
        print(f"模型：\n{model=}\n预测的结果已保存到 {output_path}")
        backup_script(model_path)  # 最后再进行备份。

    def _init_metric_funcs(self, metric_names: list[str] = None) -> list[Callable[[torch.Tensor, torch.Tensor], float]]:
        """初始化 metric_funcs"""
        if not metric_names:
            raise ValueError("metric_names不能为空")
        else:
            metric_funcs = []
            for metric_name in metric_names:
                if metric_name in self.metric_map:
                    metric_funcs.append(self.metric_map[metric_name])
                else:
                    try:
                        metric_funcs.append(globals()[metric_name])
                    except Exception as e:
                        raise ValueError(f"metric_name: {metric_name} 不存在")
        return metric_funcs


class ClassificationAutoManager(RegressionAutoManager):
    """
    适用于2D表格数据的分类预测。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_map["acc"] = lambda y_hat, y: (y_hat.argmax(dim=1) == y).sum().item()

    def exploratory_train(
        self,
        *,
        subset_size: int,
        num_epochs: int,
        metric_names: list[str] = ["loss", "acc"],
        ylim: tuple[float, float] = (0, 1),
    ):
        super().exploratory_train(subset_size=subset_size, num_epochs=num_epochs, metric_names=metric_names, ylim=ylim)

    def train_and_eval(
        self,
        *,
        k_folds: int,
        batch_size: int,
        num_epochs: int,
        metric_names: list[str] = ["loss", "acc"],
        ylim: tuple[float, float] = (0, 1),
    ):
        super().train_and_eval(k_folds=k_folds, batch_size=batch_size, num_epochs=num_epochs, metric_names=metric_names, ylim=ylim)

    def final_train(
        self,
        *,
        batch_size: int,
        num_epochs: int,
        metric_names: list[str] = ["loss", "acc"],
        ylim: tuple[float, float] = (0, 1),
    ) -> nn.Module:
        return super().final_train(batch_size=batch_size, num_epochs=num_epochs, metric_names=metric_names, ylim=ylim)

    def predict(
        self,
        *,
        test_df: pd.DataFrame,
        model: nn.Module,
        pred_col_name: str,
        batch_size: int = 512,
        model_path: str,
        submission_dir: str = SUBMISSION_DIR,
        device="cuda",
    ):
        super().predict(test_df=test_df, model=model, pred_col_name=pred_col_name, batch_size=batch_size, model_path=model_path, submission_dir=submission_dir, device=device)
