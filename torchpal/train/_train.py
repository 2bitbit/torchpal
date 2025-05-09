import pandas as pd
import torch
import os
from torch import nn
from sklearn.model_selection import KFold
import datetime
from typing import Callable
from ..utils import Accumulator, Animator, backup_script
from ..data import make_DataLoader
from .._config._constants import SUBMISSION_DIR, BACKUPS_DIR
import time
import abc


class _BaseManager(abc.ABC):
    """
    所有 Manager 的基类。
    所有主要张量都以 2D-Tensor 为操作准则。
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
        device: str = "cuda",
    ):
        """
        建议原来的 Tensor 都在 CPU 上，在需要的时候 Manager 会自动转到 device 上。以减少 VRAM 占用。

        Args:
            X_train: 应为 2D-Tensor，形状为(样本数目, 特征数目)
            y_train: 应为 2D-Tensor，形状为(样本数目, 1)
            net_cls: 模型本身的架构类。要求：输出结果为2D-Tensor。（例如：分类模型输出logits，回归模型输出 2D-Tensor-1-col 预测结果）
        """
        if X_train.ndim != 2:
            raise ValueError("X_train应为2D张量，形状为(样本数目, 特征数目)")
        if y_train.ndim != 2:
            raise ValueError("y_train应为2D张量，形状为(样本数目,1)")
        if X_test.ndim != 2:
            raise ValueError("X_test应为2D张量，形状为(样本数目, 特征数目)")
        if criterion_params.get("reduction", "mean") != "mean":
            raise ValueError("目前criterion_params中的reduction参数只支持'mean'")
        # 检查模型输出维度是否为2D
        with torch.inference_mode():
            net = net_cls(**net_params).to(device)
            sample_output = net(X_train[:5].to(device))
            if sample_output.ndim != 2:
                raise ValueError(f"模型输出应为2D张量，但得到了{sample_output.ndim}D张量")

        self.device = device
        self.X_train = X_train  # 之后对每个小批量使用.to(device)，而不是一开始就把所有数据放到device上，减少VRAM占用
        self.y_train = y_train
        self.X_test = X_test
        self.net_cls = net_cls
        self.net_params = net_params
        self.optimizer_cls = optimizer_cls
        self.optimizer_params = optimizer_params
        self.criterion = criterion_cls(**criterion_params)  # model, criterion, optimizer中criterion自始至终不需要改变，所以只需要初始化一次
        self._default_metric_names = self._get_default_metric_names()
        self._metric_map = self._get_metric_map()

    def exploratory_train(
        self,
        *,
        subset_size: int,
        num_epochs: int,
        metric_names: list[str | Callable[[torch.Tensor, torch.Tensor], float]] | None = None,
        ylim: tuple[float, float] = (0, 1),
        update_interval: int = 3,
    ):
        """先看看模型能不能在小部分训练集上过拟合，对模型架构和超参数进行初步验证。

        Args:
            metric_names:用于评估模型表现的指标。
            如果有自定义的metric，请传入 (model_output: 2D-Tensor, label: 2D-Tensor-1-col) -> 对该小批量预测成果的最终评价(float类型)。（例如：分类模型的output应该输出 logits）
            "loss"，"acc"是内置的特殊名字。
            默认是None，使用默认的metric_names=["loss"]（回归）或["loss", "acc"]（分类）

            update_interval: 只在第一个周期结束后， num_epochs % update_interval = 0 ，最后一个周期结束后。这三个时机绘制一次。
        """
        # prepare effective params
        metric_names_effective = metric_names or self._default_metric_names
        metric_funcs = self._get_metric_funcs_by_names(metric_names_effective)

        optimizer_params_effective = self.optimizer_params
        if "weight_decay" in self.optimizer_params:
            optimizer_params_copy = self.optimizer_params.copy()  # 不可直接修改self.optimizer_params，因为optimizer_params_effective会引用它
            optimizer_params_copy["weight_decay"] = 0
            optimizer_params_effective = optimizer_params_copy  # Use the copy
            print("检测到weight_decay，已自动临时视作0以屏蔽该参数")

        net_params_effective = self.net_params
        if "dropout_ps" in self.net_params:
            net_params_copy = self.net_params.copy()  # 不可直接修改self.net_params，因为net_params_effective会引用它
            net_params_copy["dropout_ps"] = [0 for _ in range(len(net_params_copy["dropout_ps"]))]
            net_params_effective = net_params_copy  # Use the copy
            print("检测到dropout_ps，已自动临时视作0以屏蔽该参数")

        # 初始化动画
        legend = self._init_legend(metric_names_effective, need_val=False)
        animator = Animator(
            num_axes=3,
            num_epochs=num_epochs,
            ylim=ylim,
            legend=legend,  # lengend 的顺序：train-metric_names
        )

        # 开始训练
        # exploratory里面只有训练集，没有验证集
        # 进行3次，每次选取一小部分作为训练集。在整个训练集上训练，并验证模型是否能记住所有数据并过拟合整个训练集
        for i in range(3):
            # 初始化 - Use effective params
            model = self.net_cls(**net_params_effective).to(self.device)
            optimizer = self.optimizer_cls(model.parameters(), **optimizer_params_effective)
            # 选取一小部分作为训练集
            indices = torch.randint(0, len(self.X_train), size=(subset_size,))
            X_train_subset, y_train_subset = self.X_train[indices].to(self.device), self.y_train[indices].to(self.device)

            for epoch in range(1, num_epochs + 1):
                # 开始本周期的训练（这里的训练不是小批量，而是整个训练集）
                model.train()
                optimizer.zero_grad()
                model_output = model(X_train_subset)
                l = self.criterion(model_output, self._process_y_for_criterion(y_train_subset))
                l.backward()
                optimizer.step()

                # 评估本周期的训练成果（以过拟合为佳）
                with torch.no_grad():
                    model.eval()
                    # 在训练集上评估（直接把整个训练集作为一个批量）
                    if (epoch % update_interval == 0) or (epoch in (1, num_epochs)):
                        animator.add(i, epoch, [metric_func(model_output.detach(), y_train_subset) for metric_func in metric_funcs])

    def train_and_eval(
        self,
        *,
        k_folds: int,
        batch_size: int,
        num_epochs: int,
        metric_names: list[str | Callable[[torch.Tensor, torch.Tensor], float]] | None = None,
        ylim: tuple[float, float] = (0, 1),
        update_interval: int = 3,
    ) -> None:
        """训练和评估模型 (K折交叉验证)

        Args:
            metric_names:用于评估模型表现的指标。
            如果有自定义的metric，请传入 (model_output: 2D-Tensor, label: 2D-Tensor-1-col) -> 对该小批量预测成果的最终评价(float类型)。（例如：分类模型的output应该输出 logits）
            "loss"，"acc"是内置的特殊名字。
            默认是None，使用默认的metric_names=["loss"]（回归）或["loss", "acc"]（分类）

            update_interval: 只在第一个周期结束后， num_epochs % update_interval = 0 ，最后一个周期结束后。这三个时机绘制一次。
        """
        # prepare effective params
        metric_names_effective = metric_names or self._default_metric_names
        metric_funcs = self._get_metric_funcs_by_names(metric_names_effective)

        # 初始化动画
        legend = self._init_legend(metric_names_effective, need_val=True)
        animator = Animator(
            num_axes=k_folds,
            num_epochs=num_epochs,
            ylim=ylim,
            legend=legend,  # lengend 的顺序：train-metric_names，val-metric_names
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
                    model_output = model(X)
                    l = self.criterion(model_output, self._process_y_for_criterion(y))
                    l.backward()
                    optimizer.step()
                    with torch.no_grad():
                        train_accumulator.add(X.shape[0], *[metric_func(model_output.detach(), y) * X.shape[0] for metric_func in metric_funcs])

                # 本周期训练结束，开始在验证集评估
                with torch.no_grad():
                    model.eval()
                    for X, y in val_loader:
                        X, y = X.to(self.device), y.to(self.device)
                        model_output = model(X)
                        val_accumulator.add(X.shape[0], *[metric_func(model_output.detach(), y) * X.shape[0] for metric_func in metric_funcs])

                    # 计算本周期训练集和验证集上的各个平均metric作参考
                    train_metrics = [metric_value / train_accumulator.data[0] for metric_value in train_accumulator.data[1:]]
                    val_metrics = [metric_value / val_accumulator.data[0] for metric_value in val_accumulator.data[1:]]
                    # 过程可视化
                    if (epoch % update_interval == 0) or (epoch in (1, num_epochs)):
                        animator.add(i, epoch, [*train_metrics, *val_metrics])

    def final_train(
        self,
        *,
        batch_size: int,
        num_epochs: int,
        metric_names: list[str | Callable[[torch.Tensor, torch.Tensor], float]] | None = None,
        ylim: tuple[float, float] = (0, 1),
        update_interval: int = 3,
    ) -> nn.Module:
        """最终训练，返回训练好的模型model

        Args:
            metric_names:用于评估模型表现的指标。
            如果有自定义的metric，请传入 (model_output: 2D-Tensor, label: 2D-Tensor-1-col) -> 对该小批量预测成果的最终评价(float类型)。（例如：分类模型的output应该输出 logits）
            "loss"，"acc"是内置的特殊名字。
            默认是None，使用默认的metric_names=["loss"]（回归）或["loss", "acc"]（分类）

            update_interval: 只在第一个周期结束后， num_epochs % update_interval = 0 ，最后一个周期结束后。这三个时机绘制一次。
        """
        # prepare effective params
        metric_names_effective = metric_names or self._default_metric_names
        metric_funcs = self._get_metric_funcs_by_names(metric_names_effective)

        # 初始化模型
        model = self.net_cls(**self.net_params).to(self.device)
        optimizer = self.optimizer_cls(model.parameters(), **self.optimizer_params)
        # 用于for的数据加载器
        train_loader = make_DataLoader(batch_size, False, self.X_train, self.y_train)

        # 初始化动画
        legend = self._init_legend(metric_names_effective, need_val=False)
        animator = Animator(
            num_axes=1,
            num_epochs=num_epochs,
            ylim=ylim,
            legend=legend,
        )

        # 开始训练
        for epoch in range(1, num_epochs + 1):
            train_accumulator = Accumulator(1 + len(metric_funcs))  # 先后顺序：样本数目，训练集上各个metric的值
            # 本周期的训练
            model.train()
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                model_output = model(X)
                l = self.criterion(model_output, self._process_y_for_criterion(y))
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    train_accumulator.add(X.shape[0], *[metric_func(model_output.detach(), y) * X.shape[0] for metric_func in metric_funcs])

            # 本周期的训练结束，开始计算训练集上的各个平均metric作参考
            with torch.no_grad():
                model.eval()
                train_metrics = [metric_value / train_accumulator.data[0] for metric_value in train_accumulator.data[1:]]
                # 过程可视化
                if (epoch % update_interval == 0) or (epoch in (1, num_epochs)):
                    animator.add(0, epoch, train_metrics)

        print("训练完成，已返回模型")
        return model

    def predict(
        self,
        *,
        model: nn.Module,
        test_df: pd.DataFrame,
        pred_col_name: str,
        backup_src_path: str,
        backup_dst_path: str = BACKUPS_DIR,
        submission_path: str = SUBMISSION_DIR,
        batch_size: int = 512,
        device: str = "cuda",
    ):
        """预测，返回预测结果

        Args:
            backup_src_path: 传入模型源文件的路径（可以是文件或目录），进行备份。
        """
        if self.X_test.shape[0] != test_df.shape[0]:
            raise ValueError("X_test 和 test_df 的样本数目不匹配")
        os.makedirs(submission_path, exist_ok=True)
        model = model.to(device)
        model.eval()
        with torch.inference_mode():
            # 分批预测，避免显存不足的尴尬情况
            y_hat = torch.zeros(len(self.X_test), dtype=torch.float32)
            for i in range(0, len(self.X_test), batch_size):
                X_test_batch = self.X_test[i : i + batch_size].to(device)
                y_hat[i : i + batch_size] = self._get_y_hat_batch(model, X_test_batch)

        # 转换成DataFrame并保存为csv
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        pred_series = pd.Series(y_hat.detach().cpu().numpy(), name=pred_col_name)
        df = pd.concat([test_df.iloc[:, 0], pred_series], axis=1)
        output_path = os.path.join(submission_path, f"{model.__class__.__name__}_{timestamp}_pred.csv")
        df.to_csv(output_path, index=False)
        # 收尾
        print(f"模型：\n{model=}\n预测的结果已保存到 {output_path}")
        time.sleep(3)  # 留点时间，将上面所有内容输出完再备份。
        backup_script(timestamp, backup_src_path, backup_dst_path)  # 最后再进行备份。

    def _init_legend(self, metric_names: list[str | Callable[[torch.Tensor, torch.Tensor], float]], need_val: bool) -> list[str]:
        if need_val:
            return ["train-" + metric_name if isinstance(metric_name, str) else "train-" + metric_name.__name__ for metric_name in metric_names] + [
                "val-" + metric_name if isinstance(metric_name, str) else "val-" + metric_name.__name__ for metric_name in metric_names
            ]
        else:
            return ["train-" + metric_name if isinstance(metric_name, str) else "train-" + metric_name.__name__ for metric_name in metric_names]

    def _get_metric_funcs_by_names(self, metric_names: list[str | Callable[[torch.Tensor, torch.Tensor], float]] | None) -> list[Callable[[torch.Tensor, torch.Tensor], float]]:
        """初始化 metric_funcs"""
        if not metric_names:
            raise ValueError("metric_names不能为空")
        
        metric_funcs = []
        for metric_name_or_func in metric_names:
            if isinstance(metric_name_or_func, str):
                # 如果是字符串，则在 _metric_map 中查找
                if metric_name_or_func in self._metric_map:
                    metric_funcs.append(self._metric_map[metric_name_or_func])
                else:
                    # 如果字符串不在 _metric_map 中，则明确指出错误
                    raise ValueError(f"预定义的 metric_name: '{metric_name_or_func}' 不存在于 _metric_map 中。")
            elif callable(metric_name_or_func):
                # 如果是可调用对象（函数），则直接使用
                metric_funcs.append(metric_name_or_func)
            else:
                # 如果既不是字符串也不是可调用对象，则类型不匹配
                raise TypeError(f"metric_names中的元素必须是字符串或可调用对象，但得到了 {type(metric_name_or_func)} 类型的 '{metric_name_or_func}'")
        return metric_funcs

    @abc.abstractmethod
    def _get_default_metric_names(self) -> list[str]: ...

    @abc.abstractmethod
    def _get_metric_map(self) -> dict[str, Callable[[torch.Tensor, torch.Tensor], float]]:
        """映射到metric的函数的格式：(model_output: 2D-Tensor, label: 2D-Tensor-1-col) -> 对该小批量预测成果的最终评价(float类型)"""
        ...

    @abc.abstractmethod
    def _process_y_for_criterion(self, y: torch.Tensor) -> torch.Tensor:
        """仅仅用于适配criterion的输入要求。结果不用于其他用途。（如metric计算）"""
        ...

    @abc.abstractmethod
    def _get_y_hat_batch(self, model: nn.Module, X_test_batch: torch.Tensor) -> torch.Tensor:
        """返回结果应为1D-Tensor，形状为(样本数目,)"""
        ...


class RegressionManager(_BaseManager):
    """适用于 2D表格数据->单结果的回归（结果为1-col，标注了该样本的回归结果）。"""

    def _get_default_metric_names(self) -> list[str]:
        return ["loss"]

    def _get_metric_map(self) -> dict[str, Callable[[torch.Tensor, torch.Tensor], float]]:
        return {
            "loss": lambda model_output, y: (self.criterion(model_output, self._process_y_for_criterion(y))).item(),
        }

    def _process_y_for_criterion(self, y: torch.Tensor) -> torch.Tensor:
        """回归模型不需要处理y"""
        return y

    def _get_y_hat_batch(self, model: nn.Module, X_test_batch: torch.Tensor) -> torch.Tensor:
        return model(X_test_batch).squeeze()


class ClassificationManager(_BaseManager):
    """适用于 2D表格数据->多分类（结果为1-col，标注了类别对应的索引）"""

    def _get_default_metric_names(self) -> list[str]:
        return ["loss", "acc"]

    def _get_metric_map(self) -> dict[str, Callable[[torch.Tensor, torch.Tensor], float]]:
        return {
            "loss": lambda model_output, y: (self.criterion(model_output, self._process_y_for_criterion(y))).item(),
            "acc": lambda model_output, y: (model_output.argmax(dim=1) == y.squeeze()).sum().item() / model_output.shape[0],
        }

    def _process_y_for_criterion(self, y: torch.Tensor) -> torch.Tensor:
        """分类模型需要视损失函数的情况，对y进行降维，来满足损失函数的传参要求"""
        if self.criterion.__class__.__name__ in ("CrossEntropyLoss", "NLLLoss"):
            # CrossEntropyLoss 和 NLLLoss 要求第二个参数是1D-Tensor，形状为(样本数目,)
            return y.squeeze()
        else:
            return y

    def _get_y_hat_batch(self, model: nn.Module, X_test_batch: torch.Tensor) -> torch.Tensor:
        return model(X_test_batch).argmax(dim=1)
