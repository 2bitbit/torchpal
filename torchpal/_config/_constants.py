import os
import sys
from IPython import get_ipython

# region DIRS

MAIN_DIR = ...  # main.py所在的顶层目录
"""默认值为执行文件所在的目录"""

if get_ipython():  # 在.ipynb文件中运行时
    MAIN_DIR = os.getcwd()  # 不要用"./"，以免工作目录改变后对main文件定位效果的丢失。
else:  # 在.py文件中运行时
    MAIN_DIR = os.path.dirname(sys.argv[0])


BACKUPS_DIR = os.path.join(MAIN_DIR, "backups")  # 备份目录
"""默认值为 执行文件所在的目录/backups"""


STATE_DICTS_DIR = os.path.join(MAIN_DIR, "state_dicts")  # 模型状态字典目录
"""默认值为 执行文件所在的目录/state_dicts"""


SUBMISSION_DIR = os.path.join(MAIN_DIR, "submissions")  # 提交目录
"""默认值为 执行文件所在的目录/submissions"""


PROCESSED_DATA_DIR = os.path.join(MAIN_DIR, "processed_data")  # 处理后的数据的目录
"""默认值为 执行文件所在的目录/processed_data"""


DATASETS_DIR = os.path.join(MAIN_DIR, "datasets")  # 数据集目录
"""默认值为 执行文件所在的目录/datasets"""

# endregion
