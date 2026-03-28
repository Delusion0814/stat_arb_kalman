import akshare as ak
import pandas as pd

def fetch_pair_data(symbol1, symbol2):
    df1 = ak.stock_zh_a_hist(symbol=symbol1, period='daily', start_date='20240101', end_date='20260328', adjust='qfq')
    df2 = ak.stock_zh_a_hist(symbol=symbol2, period='daily', start_date='20240101', end_date='20260328', adjust='qfq')
    s1 = df1[['日期', '收盘']].set_index('日期')
    s2 = df2[['日期', '收盘']].set_index('日期')
    
    data = pd.concat([s1, s2], axis=1, join='inner')
    data.columns = [symbol1, symbol2]
    print(f"成功通过 AkShare 获取 {symbol1}-{symbol2} 前复权数据，日期区间: 20240101-20260328")
    print(f"数据行数: {len(data)}，开始:{data.index.min()} 结束:{data.index.max()}")
    return data