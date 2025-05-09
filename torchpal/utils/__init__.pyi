import torch
from PIL.Image import Image
from typing import Iterable
import matplotlib.pyplot as plt

def backup_script(timestamp: str | None, src_file_or_dir_path: str, dst_dir: str = ...): ...

class Accumulator:
    data: list[float]
    def __init__(self, n: int) -> None: ...
    def add(self, *args: list[float]): ...
    def reset(self) -> None: ...

class Animator:
    axes: list[plt.Axes]
    legend: list[str]
    fmts: list[str]
    num_legends: int
    matrices: list[plt.Figure]
    def __init__(self, *, num_axes, num_epochs, ylim, legend, xlabel: str = 'epoch', ylabel: str = 'value', xscale: str = 'linear', yscale: str = 'linear', fmts=..., ax_size=..., line_width: int = 1) -> None: ...
    def add(self, ax_num: int, new_epoch: int, new_values: list[float]) -> None: ...

def show_images(imgs: Iterable[torch.Tensor | Image], num_rows: int, num_cols: int, titles: Iterable[str] = None, scale: float = 1.5): ...
def save_model_state(model: torch.nn.Module, dir: str = ...) -> str: ...
def load_model_state(model: torch.nn.Module, path: str): ...
