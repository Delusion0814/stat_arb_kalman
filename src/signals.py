def generate_signal(zscore, entry=2.0, exit=0.5, stoploss=3.0, max_hold=20):
    """
    生成交易信号（含止盈/止损/最大持仓天数）
    :param zscore: z-score 序列
    :param entry: 开仓阈值
    :param exit: 平仓阈值
    :param stoploss: 止损阈值（绝对值）
    :param max_hold: 最大持仓周期
    :return: 交易信号列表
    """
    position = 0
    positions = []
    hold_days = 0

    for z in zscore:
        if position != 0:
            hold_days += 1

        if position == 0:
            if z > entry:
                position = -1
                hold_days = 0
            elif z < -entry:
                position = 1
                hold_days = 0
        elif position == 1:
            if z >= -exit or z > stoploss or hold_days >= max_hold:
                position = 0
                hold_days = 0
        elif position == -1:
            if z <= exit or z < -stoploss or hold_days >= max_hold:
                position = 0
                hold_days = 0

        positions.append(position)

    return positions