import os
from setuptools import setup

# 获取 setup.py 文件所在的目录的绝对路径
_HERE = os.path.abspath(os.path.dirname(__file__))

# 使用绝对路径读取 requirements.txt 文件内容
with open(os.path.join(_HERE, 'requirements.txt'), 'r', encoding='utf-8') as f:
    install_requires = f.read().splitlines()

setup(
    # 其他元数据可以从 pyproject.toml 自动读取，
    # 但 install_requires 需要在这里指定，
    # 因为它被声明为 dynamic。
    install_requires=install_requires,
    # 如果你在 pyproject.toml 中还将其他字段 (如 version) 声明为 dynamic，
    # 也需要在这里或 setup.cfg 中提供。
)