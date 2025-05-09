import pandas as pd
import torch
from torch import nn
from typing import Callable

class RegressionManager("_train._BaseManager"):
    device: str
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    net_cls: nn.Module
    net_params: dict
    optimizer_cls: torch.optim.Optimizer
    optimizer_params: dict
    criterion: nn.Module
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
    ) -> None: ...
    def exploratory_train(
        self, *, subset_size: int, num_epochs: int, metric_names: list[str | Callable[[torch.Tensor, torch.Tensor], float]] | None = None, ylim: tuple[float, float] = (0, 1), update_interval: int = 3
    ) -> None: ...
    def train_and_eval(
        self, *, k_folds: int, batch_size: int, num_epochs: int, metric_names: list[str | Callable[[torch.Tensor, torch.Tensor], float]] | None = None, ylim: tuple[float, float] = (0, 1), update_interval: int = 3
    ) -> None: ...
    def final_train(
        self, *, batch_size: int, num_epochs: int, metric_names: list[str | Callable[[torch.Tensor, torch.Tensor], float]] | None = None, ylim: tuple[float, float] = (0, 1), update_interval: int = 3
    ) -> nn.Module: ...
    def predict(
        self,
        *,
        model: nn.Module,
        test_df: pd.DataFrame,
        pred_col_name: str,
        backup_src_path: str,
        backup_dst_path: str = ...,
        submission_path: str = ...,
        batch_size: int = 512,
        device: str = "cuda",
    ) -> None: ...

class ClassificationManager("_train._BaseManager"):
    device: str
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    net_cls: nn.Module
    net_params: dict
    optimizer_cls: torch.optim.Optimizer
    optimizer_params: dict
    criterion: nn.Module
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
    ) -> None: ...
    def exploratory_train(
        self, *, subset_size: int, num_epochs: int, metric_names: list[str | Callable[[torch.Tensor, torch.Tensor], float]] | None = None, ylim: tuple[float, float] = (0, 1), update_interval: int = 3
    ) -> None: ...
    def train_and_eval(
        self, *, k_folds: int, batch_size: int, num_epochs: int, metric_names: list[str | Callable[[torch.Tensor, torch.Tensor], float]] | None = None, ylim: tuple[float, float] = (0, 1), update_interval: int = 3
    ) -> None: ...
    def final_train(
        self, *, batch_size: int, num_epochs: int, metric_names: list[str | Callable[[torch.Tensor, torch.Tensor], float]] | None = None, ylim: tuple[float, float] = (0, 1), update_interval: int = 3
    ) -> nn.Module: ...
    def predict(
        self,
        *,
        model: nn.Module,
        test_df: pd.DataFrame,
        pred_col_name: str,
        backup_src_path: str,
        backup_dst_path: str = ...,
        submission_path: str = ...,
        batch_size: int = 512,
        device: str = "cuda",
    ) -> None: ...
