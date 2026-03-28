from statsmodels.tsa.stattools import coint

def find_cointegrated_pairs(price_df, pvalue_threshold=0.1):
    """
    寻找协整的股票对
    :param price_df: 股票价格 DataFrame
    :param pvalue_threshold: 协整检验的p值阈值
    :return: 可交易的股票对列表
    """
    pairs = []
    
    for i in range(len(price_df.columns)):
        for j in range(i+1, len(price_df.columns)):
            s1 = price_df.iloc[:, i]
            s2 = price_df.iloc[:, j]
            
            score, pvalue, _ = coint(s1, s2)
            
            if pvalue < pvalue_threshold:
                pairs.append((price_df.columns[i], price_df.columns[j]))
    
    return pairs