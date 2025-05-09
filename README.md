<p align="center">
  <b style="font-size: 2.2em;">torchpal</b><br>
  <span style="font-size: 1em;"> 极度新手友好的 PyTorch 伙伴；加速机器学习探索之旅！</span>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-助手-FF6F00?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 助手"/>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/许可证-MIT-green?style=for-the-badge" alt="MIT 许可证"/>
  </a>
</p>

## 📖 简介

**TorchPal** 是一个专为新手设计的 PyTorch 辅助工具包，拥有简单易用的 API。  
能极大简化深度学习模型的开发、训练和评估流程中的样板代码，帮助您更专注快速地验证自己的想法。

## 🌟 特色功能

- **自动化训练与评估**: 为常见的回归和分类任务提供自动化管理器，无需手动编写训练循环、验证逻辑
- **实时可视化**: 实时可视化训练与评估过程，支持自定义指标进行绘制，直观展示模型性能
- **K 折交叉验证**: 内置支持 K 折交叉验证，方便评估模型性能和稳定性
- **探索性训练**: 支持在小型数据子集上快速运行训练，帮助初步验证模型架构或超参数设置的合理性
- **实用工具集**: 提供丰富的实用工具，如提供模型保存/加载、脚本备份、数据加载等常用辅助功能

## ⚡ 快速开始

### 安装

```bash
pip install torchpal
```

### 基础使用（以回归任务为例）
[点击查看完整示例代码](example.ipynb)


```python
# 定义 MAE 函数 (输入为批量的 y_hat 和 y)
def mean_absolute_error(y_hat, y):
    # 函数需返回该批量的指标 *总和*
    return (y_hat - y).abs().sum().item()

# 在训练/评估时通过 metric_names 传入指标名称
manager.train_and_eval(k_folds=5, batch_size=32, num_epochs=100, metric_names=["loss", "mae"])  # 加入自定义指标 "mae"
```

## 📚 模块概览

- **`tp.train`**: 包含 `RegressionAutoManager` 和 `ClassificationAutoManager`，用于自动化训练和评估流程。
- **`tp.utils`**: 提供实用工具，如：
  - `Animator`: 实时绘制训练曲线。
  - `Accumulator`: 累加训练指标。
  - `save_model_state` / `load_model_state`: 保存和加载模型状态。
  - `backup_script`: 备份指定文件或目录。
  - `show_images`: 展示图像。
- **`tp.data`**: 数据处理相关工具，如：
  - `make_DataLoader`: 从 Tensor 创建 DataLoader。
  - `load_dataset`: 加载 torchvision 常见数据集。
- **`tp.da`**: 简单的数据分析工具 (基于 Pandas)，如：
  - `describe_df`: 计算并保存 DataFrame 的描述性统计信息。
  - `save_df`: 将 DataFrame 保存为 CSV。

_提示：TorchPal 优化了类型提示，您可以在编码时利用 IDE 的代码补全功能查看各模块和函数的可用参数及说明。_

## 🤝 贡献

欢迎各种形式的贡献！

1.  **发现 Bug 或有功能建议？** 请在 [GitHub Issues](https://github.com/your_username/torchpal/issues) 提出 (请将 `your_username/torchpal` 替换为你的实际仓库地址)。
2.  **贡献代码？**
    - Fork 本仓库。
    - 创建特性分支 (`git checkout -b feature/YourAmazingFeature`)。
    - 提交更改 (`git commit -m 'Add some AmazingFeature'`)。
    - 推送到分支 (`git push origin feature/YourAmazingFeature`)。
    - 提交 Pull Request。

## 📜 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 感谢 PyTorch 团队。
- 感谢所有开源贡献者。

---

<p align="center">
  <b>TorchPal - 让 PyTorch 更简单，让想法更快落地！</b><br>
  <i>用 ❤️ 制作</i>
</p>
