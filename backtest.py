import pandas as pd
import numpy as np

def backtest_strategy(price_df, signals, beta=None, transaction_cost=0.001):
    """
    执行回测，计算策略的表现
    :param price_df: 股票价格 DataFrame
    :param signals: 交易信号
    :param beta: 可选动态beta序列（用于价差计算）
    :param transaction_cost: 交易成本
    :return: 日收益序列和累计净值序列
    """
    df = price_df.copy()
    df['position'] = signals

    # 兼容不同列名：优先使用 X, Y；否则使用第一、第二列作为对冲对
    if 'X' not in df.columns or 'Y' not in df.columns:
        if len(df.columns) >= 2:
            df = df.rename(columns={df.columns[0]: 'X', df.columns[1]: 'Y'})
        else:
            raise ValueError('price_df must have at least two columns for X and Y prices.')

    df['return_x'] = df['X'].pct_change()
    df['return_y'] = df['Y'].pct_change()

    # 使用价差收益作为策略基准，避免直接使用单证券pct_change带来的波动失真
    if beta is not None:
        beta_s = pd.Series(beta, index=df.index).shift(1).ffill().fillna(1.0)
        df['pair_return'] = df['position'].shift(1) * (df['return_y'] - beta_s * df['return_x'])
    else:
        df['pair_return'] = df['position'].shift(1) * (df['return_y'] - df['return_x'])

    df['cost'] = abs(df['position'].diff().fillna(0)) * transaction_cost
    df['net_return'] = df['pair_return'] - df['cost']

    df['cum_return'] = (1 + df['net_return'].fillna(0)).cumprod()
    return df['net_return'], df['cum_return']