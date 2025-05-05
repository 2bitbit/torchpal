import torch
from torch.utils import data
from torchvision import datasets, transforms
from typing import Any
from .._config._constants import DATASETS_DIR



# region datasets
def load_dataset(name: str, batch_size: int, resize: Any = None) -> tuple[data.DataLoader, data.DataLoader]:
    """
    加载数据集，返回(训练集的 DataLoader, 测试集的 DataLoader)
    """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    train_dataset = getattr(datasets, name)(root=DATASETS_DIR, train=True, transform=trans, download=True)
    test_dataset = getattr(datasets, name)(root=DATASETS_DIR, train=False, transform=trans, download=True)
    return (data.DataLoader(train_dataset, batch_size, shuffle=True), data.DataLoader(test_dataset, batch_size, shuffle=False))


# endregion


# region Data Loader
def make_DataLoader(batch_size: int, is_test: bool, X: torch.Tensor, y: torch.Tensor = None) -> data.DataLoader:  # 分批次进行，以节省内存。
    """
    用法：直接传参X,y，返回一个训练或验证数据迭代器;只传参X,返回一个测试数据迭代器
    """
    if y is not None and is_test:  # 有y的测试集
        raise ValueError("y should not be provided when it's a test set")
    elif y is None and not is_test:  # 没有 y 的训练集或验证集
        raise ValueError("y must be provided when it's a training or validation set")
    else:
        if is_test:
            dataset = data.TensorDataset(X)
            return data.DataLoader(dataset, batch_size, shuffle=False)
        else:
            dataset = data.TensorDataset(X, y)
            return data.DataLoader(dataset, batch_size, shuffle=True)


# endregion
