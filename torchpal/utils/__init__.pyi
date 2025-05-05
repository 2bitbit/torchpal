from PIL.Image import Image
import torch
from typing import Iterable

# region backup
def backup_script(src_file_or_dir_path: str, dst_dir: str = ...): ...

# endregion

# region accumulator
class Accumulator:
    def __init__(self, n: int): ...
    def add(self, *args: list[float]): ...
    def reset(self): ...

# endregion

# region visualization
class Animator:
    def __init__(self, *, num_axes, num_epochs, ylim, legend, xlabel="epoch", ylabel="value", xscale="linear", yscale="linear", fmts=("-", "--", "-.", ":"), ax_size=(3.5, 2.5), line_width=1): ...
    def add(self, ax_num: int, new_epoch: int, new_values: list[float]) -> None: ...

def show_images(imgs: Iterable[torch.Tensor | Image], num_rows: int, num_cols: int, titles: Iterable[str] = None, scale: float = 1.5): ...

# endregion

# region model save & load
def save_model_state(model: torch.nn.Module, dir: str = ...): ...
def load_model_state(model: torch.nn.Module, path: str): ...

# endregion
