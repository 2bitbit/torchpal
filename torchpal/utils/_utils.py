from PIL.Image import Image
import torch
import os
import shutil
import matplotlib.pyplot as plt
import datetime
from matplotlib_inline import backend_inline
from IPython.display import display, update_display
from typing import Iterable
from .._config._constants import BACKUPS_DIR, STATE_DICTS_DIR


# region settings
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文字体显示异常问题
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常问题
# endregion


# region backup
def backup_script(timestamp: str | None, src_file_or_dir_path: str, dst_dir: str = BACKUPS_DIR):
    """
    备份文件或文件夹的脚本。日期格式：月日_时分秒\n
    timestamp: 时间戳，如果为None，则使用当前时间。
    """
    src_file_or_dir_path = os.path.normpath(src_file_or_dir_path)
    os.makedirs(dst_dir, exist_ok=True)  # 生成一个保存备份的目录
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S") if timestamp is None else timestamp
    try:
        if os.path.isdir(src_file_or_dir_path):
            src_dir_path = src_file_or_dir_path
            src_dir_name = os.path.basename(src_dir_path)
            backup_dst_path = os.path.join(dst_dir, f"{src_dir_name}_{timestamp}")
            shutil.copytree(src_dir_path, backup_dst_path)  # copytree将src目录（不含）下的所有内容复制到dst目录下
            print(f"Source directory backed up to: {backup_dst_path}")
        else:  # 复制单个文件
            src_file_path = src_file_or_dir_path
            src_file_name, src_file_ext = os.path.splitext(os.path.basename(src_file_path))
            backup_dst_path = os.path.join(dst_dir, f"{src_file_name}_{timestamp}{src_file_ext}")
            shutil.copy2(src_file_path, backup_dst_path)
            print(f"Source file backed up to: {backup_dst_path}")
    except Exception as e:
        raise e


# endregion


# region accumulator
class Accumulator:
    """
    累加器
    """

    def __init__(self, n: int):
        """
        Args:
            n: 累加器的个数
        """
        self.data = [0.0] * n

    def add(self, *args: list[float]):
        """
        添加数据
        """
        self.data = [a + b for a, b in zip(self.data, args)]

    def reset(self):
        """
        重置累加器
        """
        self.data = [0.0] * len(self.data)


# endregion


# region visualization
class Animator:
    """
    可视化实时的训练过程。\n
    （动画会单独在一个widget里面渲染，不必担心影响单元格其他输出。）
    """

    def __init__(
        self,
        *,
        num_axes,
        num_epochs,
        ylim,
        legend,
        xlabel="epoch",
        ylabel="value",
        xscale="linear",
        yscale="linear",
        fmts=...,
        ax_size=...,
        line_width=1,
    ):
        """
        Args:
            num_axes: 绘制图表的个数
            num_epochs: 训练的轮次
            legend: 图例的列表
            fmts: 绘制图表的格式，默认是("-", "--", "-.", ":")无限循环
            ax_size: 每个图表的尺寸，默认是(len(legend)+14)/15*(3.5, 2.5)
        """
        # 根据legend的长度，确定fmts的长度
        if fmts is ...:
            fmts = ["-", "--", "-.", ":"] * (len(legend) // 4 + 1)
        if ax_size is ...:
            ax_size = [3.5 * (len(legend) + 14) / 15, 2.5 * (len(legend) + 14) / 15]

        self._set_backend_inline()
        self.fig, self.axes = plt.subplots(1, num_axes, figsize=(ax_size[0] * num_axes, ax_size[1]))
        self.axes = [self.axes] if num_axes == 1 else self.axes  # 确保axes是一个列表
        # 配置坐标轴的函数
        self._config_axes = lambda ax: self._config_axes_gen_func(ax, xlabel, ylabel, (1, num_epochs), ylim, xscale, yscale)
        self.legend = legend
        self.fmts = fmts
        self.num_legends = len(legend)  # 指标的个数
        self.matrices = []  # 使用列表，允许每个矩阵的shape不同。
        """
        每一行是一次训练结束后的数据：第一列是该训练周期，后面全是指标（in same order with self.legend）。
        """
        for _ in range(num_axes):
            self.matrices.append(torch.zeros(0, 1 + self.num_legends, dtype=torch.float32)) 
        self._plot_handle = display(self.fig, display_id=True)

    def add(self, ax_num: int, new_epoch: int, new_values: list[float]) -> None:
        """
        按照传入的legend的顺序添加新的训练数据点

        Args:
            ax_num: ax的编号（从0开始）
            new_epoch: 当前的训练轮次：随便一个数字。每个点表示该周期结束后的指标值
            new_values: 一维向量，包含该周期结束后的多个指标的值：一个可迭代对象。按照图例列表里面的顺序排列。
        """
        # 更新历史数据
        new_line = torch.tensor([[new_epoch, *new_values]], dtype=torch.float32)
        self.matrices[ax_num] = torch.concat([self.matrices[ax_num], new_line], dim=0)  # 性能开销微乎其微，完全不值得预分配空间来优化！ ！
        # 下面开始进行训练过程可视化
        # 没有用 for i, value in enumerate(new_values)，而是用矢量加速
        ax = self.axes[ax_num]
        ax.clear()
        for i in range(self.num_legends):
            ax.plot(self.matrices[ax_num][:, 0], self.matrices[ax_num][:, i + 1], self.fmts[i], label=self.legend[i])
        plt.tight_layout()

        self._config_axes(ax)

        update_display(self.fig, display_id=self._plot_handle.display_id)
        plt.close(self.fig)  # 避免一个块内的所有代码运行结束后 ipynb 自动再呈现一遍 fig

    def _set_backend_inline(self):
        """设置matplotlib的backend为inline"""
        backend_inline.set_matplotlib_formats("svg")

    def _config_axes_gen_func(self, ax, xlabel, ylabel, xlim, ylim, xscale, yscale):
        """这是一个设置matplotlib的坐标轴属性的母函数，用于生成一个设置坐标轴属性的函数"""
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend()
        ax.grid()


def show_images(imgs: Iterable[torch.Tensor | Image], num_rows: int, num_cols: int, titles: Iterable[str] = None, scale: float = 1.5):
    """
    绘制图像列表
    """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.cpu().detach().numpy())
        elif isinstance(img, Image):
            ax.imshow(img)
        else:
            raise ValueError(f"无法识别的图像类型: {type(img)}")
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


# endregion


# region model save & load
def save_model_state(model: torch.nn.Module, dir: str = STATE_DICTS_DIR) -> str:
    """
    保存模型状态字典
    """
    os.makedirs(dir, exist_ok=True)  # 确保目录存在
    class_name = model.__class__.__name__
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    f_name = f"{class_name}_{timestamp}.pth"
    path = os.path.join(dir, f_name)
    torch.save(model.state_dict(), path)
    print(f"模型状态字典已保存到 {path}")
    return path


def load_model_state(model: torch.nn.Module, path: str):
    """
    加载模型状态字典\n
    如何load: 你需要重新实例化一个模型（与被保存模型的结构完全相同）。\n
    以eval模式加载模型状态字典，之后记得使用with torch.inference_mode()来进行推理。
    """
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# endregion
