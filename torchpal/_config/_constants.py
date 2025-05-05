import os
import sys
from IPython import get_ipython

# region DIRS

# 获取main.py所在的顶层目录
if get_ipython():  # 在.ipynb文件中运行时
    MAIN_DIR = os.getcwd()  # 不要用"./"，以免工作目录改变后对main文件定位效果的丢失。
else:  # 在.py文件中运行时
    MAIN_DIR = os.path.dirname(sys.argv[0])

# 备份目录
BACKUPS_DIR = os.path.join(MAIN_DIR, "backups")
# 模型状态字典目录
STATE_DICTS_DIR = os.path.join(MAIN_DIR, "state_dicts")
# 提交目录
SUBMISSION_DIR = os.path.join(MAIN_DIR, "submissions")
# 处理后的数据的目录
PROCESSED_DATA_DIR = os.path.join(MAIN_DIR, "processed_data")
# 数据集目录
DATASETS_DIR = os.path.join(MAIN_DIR, "datasets")

# endregion
