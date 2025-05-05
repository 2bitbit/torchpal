import torch
import torch.nn as nn
import pandas as pd

class RegressionAutoManager:
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
    ): ...
    def exploratory_train(
        self,
        *,
        subset_size: int,
        num_epochs: int,
        metric_names: list[str] = ["loss"],
        ylim: tuple[float, float] = (0, 1),
    ): ...
    def train_and_eval(
        self,
        *,
        k_folds: int,
        batch_size: int,
        num_epochs: int,
        metric_names: list[str] = ["loss"],
        ylim: tuple[float, float] = (0, 1),
    ): ...
    def final_train(
        self,
        *,
        batch_size: int,
        num_epochs: int,
        metric_names: list[str] = ["loss"],
        ylim: tuple[float, float] = (0, 1),
    ) -> nn.Module: ...
    def predict(
        self,
        *,
        test_df: pd.DataFrame,
        model: nn.Module,
        pred_col_name: str,
        batch_size: int = 512,
        model_path: str,
        submission_dir: str = ...,
        device: str = "cuda",
    ): ...

class ClassificationAutoManager(RegressionAutoManager):
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
    ): ...
    def exploratory_train(
        self,
        *,
        subset_size: int,
        num_epochs: int,
        metric_names: list[str] = ["loss", "acc"],
        ylim: tuple[float, float] = (0, 1),
    ): ...
    def train_and_eval(
        self,
        *,
        k_folds: int,
        batch_size: int,
        num_epochs: int,
        metric_names: list[str] = ["loss", "acc"],
        ylim: tuple[float, float] = (0, 1),
    ): ...
    def final_train(
        self,
        *,
        batch_size: int,
        num_epochs: int,
        metric_names: list[str] = ["loss", "acc"],
        ylim: tuple[float, float] = (0, 1),
    ) -> nn.Module: ...
    def predict(
        self,
        *,
        test_df: pd.DataFrame,
        model: nn.Module,
        pred_col_name: str,
        batch_size: int = 512,
        model_path: str,
        submission_dir: str = ...,
        device: str = "cuda",
    ): ...
