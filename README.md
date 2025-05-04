# ✨ TorchPal - 您的PyTorch得力助手 ✨

**✨ 极度新手友好的 PyTorch 伴侣 ✨**

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-助手-FF6F00?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch助手"/>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/许可证-MIT-green?style=for-the-badge" alt="MIT许可证"/>
  <img src="https://img.shields.io/badge/版本-0.0.1-red?style=for-the-badge" alt="版本 0.0.1"/>
</p>

<p align="center">
  <b>🚀 加速您的深度学习工作流程 | 简化训练过程 | 提高开发效率 🚀</b>
</p>

## 📖 简介

**TorchPal** 是一个精心设计的PyTorch辅助工具包，旨在简化深度学习模型的开发、训练和评估流程。无论您是经验丰富的研究人员还是机器学习初学者，TorchPal都能为您提供一系列便捷工具，使您的深度学习之旅更加顺畅。

```python
# 简单易用的API
import torchpal as tp

# 使用自动化管理器轻松训练模型
manager = tp.train.RegressionAutoManager(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    net_cls=MyModel,
    net_params={"input_dim": 10, "hidden_dim": 64, "output_dim": 1},
    criterion_cls=nn.MSELoss,
    criterion_params={},
    optimizer_cls=torch.optim.Adam,
    optimizer_params={"lr": 0.001}
)

# 交互式可视化训练过程
manager.train_and_eval(k_folds=5, batch_size=32, num_epochs=100)
```

## ✨ 特色功能

### 🔄 自动化训练管理
- **`RegressionAutoManager`** - 用于回归任务的自动化训练管理器
- **`ClassificationAutoManager`** - 专为分类任务设计的训练管理器
- **K折交叉验证** - 内置支持，无需额外代码

### 📊 实时可视化
- **`Animator`** - 实时可视化训练过程，直观展示模型性能
- **`show_images`** - 轻松展示和比较图像数据

### 🛠️ 实用工具集
- **`Accumulator`** - 高效跟踪和累积训练指标
- **数据增强** - 通过`da`模块提供的工具增强您的数据
- **模型保存与加载** - 简化模型状态的保存和恢复

### 🔍 探索性训练
- 在小数据集上快速测试模型性能
- 轻松识别过拟合/欠拟合问题

## 🚀 快速开始

### 安装

```bash
pip install torchpal
```

### 基础使用

```python
import torchpal as tp
import torch
from torch import nn

# 1. 准备数据
X_train, y_train = ...  # 您的训练数据
X_test = ...  # 您的测试数据

# 2. 定义模型
class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# 3. 使用TorchPal训练
manager = tp.train.RegressionAutoManager(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    net_cls=MyModel,
    net_params={"input_dim": X_train.shape[1], "hidden_dim": 64, "output_dim": 1},
    criterion_cls=nn.MSELoss,
    criterion_params={},
    optimizer_cls=torch.optim.Adam,
    optimizer_params={"lr": 0.001}
)

# 探索性训练
manager.exploratory_train(subset_size=100, num_epochs=50)

# K折交叉验证训练
manager.train_and_eval(k_folds=5, batch_size=32, num_epochs=100)

# 最终训练
model = manager.final_train(batch_size=32, num_epochs=100)

# 预测
predictions = manager.predict(
    test_df=test_df,
    model=model,
    pred_col_name="prediction",
    model_path="model.pth"
)
```

## 📚 模块概览

TorchPal包含多个功能强大的模块：

- **`tp.train`** - 训练和评估模型的自动化管理器
- **`tp.utils`** - 实用工具函数和类（如`Animator`、`Accumulator`等）
- **`tp.data`** - 数据处理和加载工具
- **`tp.da`** - 数据增强技术

## 🔧 高级功能展示

### 自定义指标

```python
def custom_metric(y_hat, y):
    # 您的自定义指标逻辑
    return ((y_hat - y).abs() < 0.5).float().sum().item()

# 在训练中使用自定义指标
manager.metric_map["custom"] = custom_metric
manager.train_and_eval(
    k_folds=5, 
    batch_size=32, 
    num_epochs=100, 
    metric_names=["loss", "custom"]
)
```

### 模型状态保存与加载

```python
# 保存模型状态
tp.utils.save_model_state(model)

# 加载模型状态
model = MyModel(input_dim=10, hidden_dim=64, output_dim=1)
model = tp.utils.load_model_state(model, "path/to/model.pth")
```

## 🤝 贡献

我们欢迎并鼓励社区贡献！如果您有任何改进建议或发现了bug，请随时提交问题或拉取请求。

欢迎各种形式的贡献！如果你有任何建议、发现 bug 或想改进代码，请随时：

1.  Fork 本仓库
2.  创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3.  提交你的更改 (`git commit -m 'Add some AmazingFeature'`)
4.  推送到分支 (`git push origin feature/AmazingFeature`)
5.  提交一个 Pull Request

也欢迎在 [Issues](...) 页面提出问题或建议。

## 📜 许可证

本项目采用MIT许可证 - 详情请参阅[LICENSE](LICENSE)文件。

## 🙏 致谢

特别感谢PyTorch团队和所有为深度学习社区做出贡献的开发者们。

---

<p align="center">
  <b>TorchPal - 让PyTorch使用更加轻松愉快！</b><br>
  <i>用❤️打造</i>
</p>
