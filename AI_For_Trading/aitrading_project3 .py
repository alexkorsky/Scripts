import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_dollar_volume_weights(close, volume):
    """
    Generate dollar volume weights.

    Parameters
    ----------
    close : DataFrame
        Close price for each ticker and date
    volume : str
        Volume for each ticker and date

    Returns
    -------
    dollar_volume_weights : DataFrame
        The dollar volume weights for each ticker and date
    """
    
    assert close.index.equals(volume.index)
    assert close.columns.equals(volume.columns)
    
    #TODO: Implement function
    dollar_value = close * volume
    
    total_value = dollar_value.sum(axis=1)
    
    #print(dollar_value[:10])
    #print(total_value)
    
    
    return dollar_value.divide(total_value.values, axis = 'index')
    
def calculate_dividend_weights(dividends):
    """
    Calculate dividend weights.

    Parameters
    ----------
    dividends : DataFrame
        Dividend for each stock and date

    Returns
    -------
    dividend_weights : DataFrame
        Weights for each stock and date
    """
    #TODO: Implement function

    '''
    cum_div = dividends
    for x in range(len(dividends.index)):
        print(dividends.shift(x+1))
        cum_div = cum_div + dividends.shift(x+1).fillna(0)
    '''

    cum_div = dividends.cumsum()
    total_value = cum_div.sum(axis=1)
    
    return cum_div.divide(total_value.values, axis = 'index')
    
def generate_returns(prices):
    """
    Generate returns for ticker and date. Non-log. Simple

    Parameters
    ----------
    prices : DataFrame
        Price for each ticker and date

    Returns
    -------
    returns : Dataframe
        The returns for each ticker and date
    """
    #TODO: Implement function
    return (prices - prices.shift(1))/prices.shift(1)
    
def generate_weighted_returns(returns, weights):
    """
    Generate weighted returns.

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    weights : DataFrame
        Weights for each ticker and date

    Returns
    -------
    weighted_returns : DataFrame
        Weighted returns for each ticker and date
    """
    assert returns.index.equals(weights.index)
    assert returns.columns.equals(weights.columns)
    
    #TODO: Implement function

    return returns * weights

# this below is weird - i guess parameter 'rturns; would be already weighted by why do we add 1???
# this helps: 
#       https://money.stackexchange.com/questions/80179/how-to-calculate-the-return-over-a-period-from-daily-returns
    
def calculate_cumulative_returns(returns):
    """
    Calculate cumulative returns.

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date

    Returns
    -------
    cumulative_returns : Pandas Series
        Cumulative returns for each date
    """
    #TODO: Implement function
    
    print(returns)
    return (returns.sum(axis  = 1) + 1).cumprod()

    #print(returns.sum(axis = 1))
 
 def tracking_error(benchmark_returns_by_date, etf_returns_by_date):
    """
    Calculate the tracking error.

    Parameters
    ----------
    benchmark_returns_by_date : Pandas Series
        The benchmark returns for each date
    etf_returns_by_date : Pandas Series
        The ETF returns for each date

    Returns
    -------
    tracking_error : float
        The tracking error
    """
    assert benchmark_returns_by_date.index.equals(etf_returns_by_date.index)
    
    #TODO: Implement function
    sample_std = (etf_returns_by_date - benchmark_returns_by_date).std()

    return sample_std * 15.87450786638754 # how the fuck do we do math.sqrt(252)
    
def get_covariance_returns(returns):
    """
    Calculate covariance matrices.

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date

    Returns
    -------
    returns_covariance  : 2 dimensional Ndarray
        The covariance of the returns
    """
    #TODO: Implement function
    nonNAreturns = returns.fillna(0)
    cov = np.cov(nonNAreturns.values.T)
        
    return cov