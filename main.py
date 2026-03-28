import sys
import pandas as pd
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt
from src.cointegration import find_cointegrated_pairs
from src.kalman import kalman_estimate
from src.signals import generate_signal
from src.backtest import backtest_strategy
from src.metrics import sharpe_ratio, max_drawdown

def fetch_pair_data(symbol1, symbol2):
    df1 = ak.stock_zh_a_hist(symbol=symbol1, period='daily', start_date='20240101', end_date='20260328', adjust='qfq')
    df2 = ak.stock_zh_a_hist(symbol=symbol2, period='daily', start_date='20240101', end_date='20260328', adjust='qfq')
    s1 = df1[['日期', '收盘']].set_index('日期')
    s2 = df2[['日期', '收盘']].set_index('日期')
    
    data = pd.concat([s1, s2], axis=1, join='inner')
    data.columns = [symbol1, symbol2]
    print(f"✅ 成功通过 AkShare 获取 {symbol1}-{symbol2} 前复权数据，日期区间: 20240101-20260328")
    print(f"数据行数: {len(data)}，开始:{data.index.min()} 结束:{data.index.max()}")
    return data

def run_pair(symbol1, symbol2):
    print('=' * 50)
    print(f'回测标的: {symbol1} / {symbol2}')

    data = fetch_pair_data(symbol1, symbol2)
    window_df = data.copy()

    pairs = find_cointegrated_pairs(window_df)
    if not pairs:
        raise RuntimeError(f"{symbol1}/{symbol2} 未通过协整检验")

    pair = pairs[0]
    x = window_df[pair[0]]
    y = window_df[pair[1]]

    # 1. 计算 Kalman Beta
    beta_series, intercept_series = kalman_estimate(y, x)

    # 强制对齐索引，防止绘图错位
    beta_series = pd.Series(beta_series, index=y.index)
    intercept_series = pd.Series(intercept_series, index=y.index)

    # 2. 计算 Z-Score
    spread = y - beta_series * x - intercept_series
    lookback = 20
    mu = spread.rolling(lookback).mean()
    sigma = spread.rolling(lookback).std().replace(0, np.nan)
    zscore = ((spread - mu) / sigma).fillna(0)

    # 3. 回测
    signals = generate_signal(zscore, entry=2.5, exit=0.8, stoploss=3.0, max_hold=15)
    net_return, cum_return = backtest_strategy(window_df[[pair[0], pair[1]]], signals, beta=beta_series, transaction_cost=0.002)

    # --- 开始绘图逻辑 ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # 图像 1: Kalman Beta 曲线
    axes[0].plot(beta_series, label=f'Kalman Beta ({pair[1]}/{pair[0]})', color='orange')
    axes[0].set_title('Dynamic Hedge Ratio (Beta)')
    axes[0].legend()
    axes[0].grid(True)

    # 图像 2: Z-Score 曲线 (含阈值线)
    axes[1].plot(zscore, label='Z-Score', color='blue')
    axes[1].axhline(2.5, color='red', linestyle='--', alpha=0.6, label='Entry (Upper)')
    axes[1].axhline(-2.5, color='green', linestyle='--', alpha=0.6, label='Entry (Lower)')
    axes[1].axhline(0, color='black', linestyle='-', alpha=0.3)
    axes[1].set_title('Z-Score & Trading Thresholds')
    axes[1].legend()
    axes[1].grid(True)

    # 图像 3: 累计收益曲线
    axes[2].plot(cum_return, label='Equity Curve', color='green', linewidth=2)
    axes[2].set_title('Cumulative Returns')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
    # --- 绘图结束 ---

    # 计算各项指标
    daily_return = net_return.fillna(0)
    sharpe = sharpe_ratio(daily_return)
    max_dd = max_drawdown(cum_return)
    total_return = (cum_return.iloc[-1] - 1) * 100
    
    period_years = (cum_return.index[-1] - cum_return.index[0]).days / 365.25
    annual_return = ((cum_return.iloc[-1]) ** (1 / period_years) - 1) * 100 if period_years > 0 else 0
    total_profit = 100000 * (cum_return.iloc[-1] - 1)

    # 打印结果
    print(f"夏普比率: {sharpe:.4f}")
    print(f"最大回撤: {max_dd:.4%}")
    print(f"总收益率: {total_return:.2f}%")
    print(f"年化收益: {annual_return:.2f}%")
    print(f"实际收益: {total_profit:.2f} 元")
    print('=' * 50)

    return {
        'symbol1': symbol1, 'symbol2': symbol2,
        'sharpe': sharpe, 'max_drawdown': max_dd,
        'total_return': total_return, 'profit': total_profit
    }

if __name__ == '__main__':
    if len(sys.argv) >= 3:
        s1 = sys.argv[1]
        s2 = sys.argv[2]
    else:
        s1 = '600030'
        s2 = '600837'

    run_pair(s1, s2)

    

