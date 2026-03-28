import numpy as np

def sharpe_ratio(returns, risk_free_rate=0.0):
    """
    计算夏普比率
    :param returns: 每日收益率
    :param risk_free_rate: 无风险利率
    :return: 夏普比率
    """
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

def max_drawdown(cum_return):
    """
    计算最大回撤
    :param cum_return: 累计回报
    :return: 最大回撤
    """
    cum_max = cum_return.cummax()
    drawdown = cum_return / cum_max - 1
    return drawdown.min()