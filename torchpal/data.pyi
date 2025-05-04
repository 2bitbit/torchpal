import torch
from ._constants import DATASETS_DIR as DATASETS_DIR
from torch.utils import data
from typing import Any

def load_dataset(name: str, batch_size: int, resize: Any = None) -> tuple[data.DataLoader, data.DataLoader]: ...
def make_DataLoader(batch_size: int, is_test: bool, X: torch.Tensor, y: torch.Tensor = None) -> data.DataLoader: ...
