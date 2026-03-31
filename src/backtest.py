import pandas as pd

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
    # 统一列名
    cols = df.columns
    df = df.rename(columns={cols[0]: 'X', cols[1]: 'Y'})
    
    # 1. 准备收益率和 Beta
    df['ret_x'] = df['X'].pct_change()
    df['ret_y'] = df['Y'].pct_change()
    
    # 确保 beta 是 Series 且对齐
    if beta is not None:
        beta_s = pd.Series(beta, index=df.index)
    else:
        beta_s = pd.Series(1.0, index=df.index)
    
    # 2. 计算权重 (Weights)
    # position 为 1 代表做多价差 (买Y卖X)；-1 代表做空价差 (卖Y买X)
    # 权重计算：Y 的权重恒定为 1/|1+beta|，X 的权重为 -beta/|1+beta|
    total_value = 1.0 + beta_s.abs()
    weight_y = 1.0 / total_value
    weight_x = -beta_s / total_value
    
    # 实际持仓权重（结合交易信号）
    df['w_y'] = signals.shift(1) * weight_y.shift(1)
    df['w_x'] = signals.shift(1) * weight_x.shift(1)
    
    # 3. 计算组合收益 (Portfolio Return)
    # 此时收益率 = 权重X * 收益X + 权重Y * 收益Y
    df['pair_return'] = (df['w_y'] * df['ret_y']) + (df['w_x'] * df['ret_x'])
    
    # 4. 计算完整的交易成本 (包含信号变化 + Beta 调仓)
    # 今天的目标权重与昨天的实际权重之差
    target_w_y = signals * weight_y
    target_w_x = signals * weight_x
    
    # 计算 X 和 Y 各自的权重变化绝对值
    diff_y = abs(target_w_y - df['w_y'].fillna(0))
    diff_x = abs(target_w_x - df['w_x'].fillna(0))
    
    df['cost'] = (diff_y + diff_x) * transaction_cost
    
    # 5. 净收益与累计净值
    df['net_return'] = df['pair_return'] - df['cost']
    df['cum_return'] = (1 + df['net_return'].fillna(0)).cumprod()
    
    return df['net_return'], df['cum_return']
