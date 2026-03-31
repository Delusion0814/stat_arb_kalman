import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.cointegration import find_cointegrated_pairs
from src.kalman import kalman_estimate
from src.signals import generate_signal
from src.backtest import backtest_strategy
from src.metrics import sharpe_ratio, max_drawdown
from src.getdata import fetch_pair_data 
#from src.data_loader import load_data # 可选：如果不使用 AkShare 获取数据，而是从本地 CSV 文件加载数据，则需要引入这个模块

def run_pair(symbol1, symbol2, formation_period=252):
    print('=' * 50)
    print(f'回测标的: {symbol1} / {symbol2}')

    # 获取原始数据
    data = fetch_pair_data(symbol1, symbol2)
    #data = load_data("pair_data.csv")
    
    # --- 1. 前瞻偏差处理 ---
    # 仅使用“形成期”（前 formation_period 天）进行初次协整检验
    formation_df = data.iloc[:formation_period]
    trading_df = data.iloc[formation_period:] # 实际交易区间
    
    pairs = find_cointegrated_pairs(formation_df)
    if not pairs:
        # 如果初始不协整，可以选择跳过或记录
        raise RuntimeError(f"{symbol1}/{symbol2} 在形成期未通过协整检验")

    pair = pairs[0]
    # 使用全量数据进入 Kalman，但 Kalman 本身是流式的，不会预知未来
    x = data[pair[0]]
    y = data[pair[1]]

    # --- 2. 计算 Kalman Beta ---
    beta_list, intercept_list = kalman_estimate(y, x)
    
    beta_series = pd.Series(beta_list, index=y.index)
    intercept_series = pd.Series(intercept_list, index=y.index)

    # --- 3. Z-Score 计算 ---
    # 直接计算残差 (Spread)
    spread = y - (beta_series * x + intercept_series)
    
    # 均值(mu)直接设为 0, 因为 Kalman 的目标就是将残差均值收敛至 0
    mu = 0 
    sigma = spread.rolling(window=30).std()
    zscore = (spread / sigma).fillna(0)

    # --- 4. 回测 (仅在非形成期进行，或者全量回测但观察交易区间) ---
    signals_list = generate_signal(zscore, entry=2.5, exit=0.8, stoploss=3.0, max_hold=15)
    signals = pd.Series(signals_list, index=zscore.index)
    
    # 传入全量数据，但指标计算会涵盖整个区间
    net_return, cum_return = backtest_strategy(
        data[[pair[0], pair[1]]], 
        signals, 
        beta=beta_series, 
        transaction_cost=0.002
    )

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

    # 5. 指标计算
    daily_return = net_return.fillna(0)
    sharpe = sharpe_ratio(daily_return)
    max_dd = max_drawdown(cum_return)
    total_return = (cum_return.iloc[-1] - 1) * 100
    
    # 使用交易日频率 (252) 计算年化，比自然日计算更准
    trading_days = len(cum_return)
    annual_return = ((cum_return.iloc[-1]) ** (252 / trading_days) - 1) * 100
    
    # 打印结果
    print(f"夏普比率: {sharpe:.4f}")
    print(f"最大回撤: {max_dd:.4%}")
    print(f"总收益率: {total_return:.2f}%")
    print(f"年化收益: {annual_return:.2f}%")

    print('=' * 50)

    return {
        'symbol1': symbol1, 'symbol2': symbol2,
        'sharpe': sharpe, 'annual_return': annual_return
    }

if __name__ == '__main__':
    if len(sys.argv) >= 3:
        s1 = sys.argv[1]
        s2 = sys.argv[2]
    else:
        s1 = '600030' # 中信证券
        s2 = '600837' # 海通证券
        print(f"未提供命令行参数，默认使用 {s1} 和 {s2} 进行回测")

    run_pair(s1, s2)
