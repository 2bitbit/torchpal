# data analysis
import pandas as pd
import os
from .._config._constants import PROCESSED_DATA_DIR



def describe_df(df: pd.DataFrame, path: str = PROCESSED_DATA_DIR) -> None:
    """
    打印数据的信息包括:Dtype,null_count,count,mean,std,min,25%,50%,75%,max\n
    并保存到指定路径（默认是 ./processed/）
    """
    os.makedirs(path, exist_ok=True)  # 确保目录存在
    df = pd.concat([df.dtypes, df.isnull().sum(), df.describe().T], axis=1)
    df.columns = ["Dtype", "null_count", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    f_name = os.path.join(path, "describe_data.csv")
    df.to_csv(f_name, index=False)
    print(f"数据信息已保存到 {f_name}")


def save_df(df: pd.DataFrame, output_file_name: str, output_dir: str = PROCESSED_DATA_DIR) -> None:  # 不同于 py 文件，ipynb 会将工作目录切换到正在执行的文件所在的目录，所以可用"./"
    """
    保存 DataFrame 数据到指定路径（默认是 ./processed/）
    """
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在
    path = os.path.join(output_dir, output_file_name)
    df.to_csv(path, index=False)
    print(f"数据已保存到 {path}")
