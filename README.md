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

- **自动化训练与评估**: 为常见的回归和分类任务提供自动化管理器，无需手动编写训练循环、验证逻辑（内置 K 折交叉验证）
- **实时可视化**: 实时可视化训练与评估过程，直观展示模型性能；支持自定义指标进行绘制，
- **探索性训练**: 支持在小型数据子集上快速运行训练，帮助初步验证模型架构或超参数设置的合理性
- **实用工具集**: 提供丰富的实用工具，如提供模型保存/加载、脚本备份、图片展示等常用辅助功能

## ⚡ 快速开始

### 安装

```bash
pip install torchpal
```

### 基础使用（以回归任务为例）
[点击查看完整示例代码](example.ipynb)

## 📚 模块概览

- **`tp.train`**: 包含 `RegressionAutoManager` 和 `ClassificationAutoManager`，用于自动化训练和评估流程。
- **`tp.utils`**: 提供实用工具
- **`tp.data`**: 数据处理相关工具
- **`tp.da`**: 简单的数据分析工具 (基于 Pandas)


提示：TorchPal 优化了类型提示，您可以在编码时利用 IDE 的提示与代码补全功能轻松查看各模块的内容、函数的可用参数及说明。

## 🤝 贡献

欢迎各种形式的贡献！

1.  **发现 Bug 或有功能建议？** 请在 [GitHub Issues](https://github.com/2bitbit/torchpal/issues) 提出。
2.  **贡献代码：**
    - Fork 本仓库。
    - 创建特性分支 (`git checkout -b feature/YourAmazingFeature`)。
    - 提交更改 (`git commit -m 'Add some AmazingFeature'`)。
    - 推送到分支 (`git push origin feature/YourAmazingFeature`)。
    - 提交 Pull Request。

## 🙏 致谢

- 感谢 PyTorch 团队。
- 感谢所有开源贡献者。


<p align="center">
  <b>TorchPal - 让 PyTorch 更简单，让想法更快落地！</b><br>
  <i>用 ❤️ 制作</i>
</p>
