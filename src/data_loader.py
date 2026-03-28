import pandas as pd

def load_data(file_path):
    """
    加载数据，并设置时间戳为索引
    :param file_path: 数据文件路径
    :return: DataFrame
    """
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df