import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_original = pd.read_csv('eod-quotemedia.csv', parse_dates=['date'], index_col=False)

# Add TB sector to the market
df = df_original
#df = pd.concat([df] + project_helper.generate_tb_sector(df[df['ticker'] == 'AAPL']['date']), ignore_index=True)

close = df.reset_index().pivot(index='date', columns='ticker', values='adj_close')
high = df.reset_index().pivot(index='date', columns='ticker', values='adj_high')
low = df.reset_index().pivot(index='date', columns='ticker', values='adj_low')

def get_high_lows_lookback(high, low, lookback_days):
    """
    Get the highs and lows in a lookback window.
    
    Parameters
    ----------
    high : DataFrame
        High price for each ticker and date
    low : DataFrame
        Low price for each ticker and date
    lookback_days : int
        The number of days to look back
    
    Returns
    -------
    lookback_high : DataFrame
        Lookback high price for each ticker and date
    lookback_low : DataFrame
        Lookback low price for each ticker and date
    """
    #TODO: Implement function
    #Don't forget to shift 1 day to avoid counting current day. T
    return high.shift(1).rolling(lookback_days).max(), low.shift(1).rolling(lookback_days).min()

def get_long_short(close, lookback_high, lookback_low):
    """
    Generate the signals long, short, and do nothing.
    
    Parameters
    ----------
    close : DataFrame
        Close price for each ticker and date
    lookback_high : DataFrame
        Lookback high price for each ticker and date
    lookback_low : DataFrame
        Lookback low price for each ticker and date
    
    Returns
    -------
    long_short : DataFrame
        The long, short, and do nothing signals for each ticker and date
    """
    #TODO: Implement function   
    pos_signal_df = close // lookback_high;
    pos_signal_df = pos_signal_df.applymap(lambda x: 1 if x >= 1 else 0)
    
    neg_signal_df = lookback_low // close * -1;
    neg_signal_df = neg_signal_df.applymap(lambda x: -1 if x <= -1 else 0)
        
    return_df = pos_signal_df + neg_signal_df
    
    #another way of doing this is very cool:
    #return ((close>lookback_high)*1-(close<lookback_low)*1).astype(int)
    
    return return_df

def clear_signals(signals, window_size):
    """
    Clear out signals in a Series of just long or short signals.
    
    Remove the number of signals down to 1 within the window size time period.
    
    Parameters
    ----------
    signals : Pandas Series
        The long, short, or do nothing signals
    window_size : int
        The number of days to have a single signal       
    
    Returns
    -------
    signals : Pandas Series
        Signals with the signals removed from the window size
    """
    # Start with buffer of window size
    # This handles the edge case of calculating past_signal in the beginning
    clean_signals = [0]*window_size
    
    for signal_i, current_signal in enumerate(signals):
        # Check if there was a signal in the past window_size of days
        has_past_signal = bool(sum(clean_signals[signal_i:signal_i+window_size]))
        # Use the current signal if there's no past signal, else 0/False
        clean_signals.append(not has_past_signal and current_signal)
        
    # Remove buffer
    clean_signals = clean_signals[window_size:]

    # Return the signals as a Series of Ints
    return pd.Series(np.array(clean_signals).astype(np.int), signals.index)


def filter_signals(signal, lookahead_days):
    """
    Filter out signals in a DataFrame.
    
    Parameters
    ----------
    signal : DataFrame
        The long, short, and do nothing signals for each ticker and date
    lookahead_days : int
        The number of days to look ahead
    
    Returns
    -------
    filtered_signal : DataFrame
        The filtered long, short, and do nothing signals for each ticker and date
    """
    #TODO: Implement function
    signal_t = signal.transpose()
    return_df = pd.DataFrame(index = signal_t.index, columns = signal_t.columns)
    k = 0
    for i, signal_series in signal_t.iterrows():           
        
        # first  get signal_series of 1s and 'clear' them
        pos_signal_series = (signal_series > 0) * 1
        clear_pos_signal_series = clear_signals(pos_signal_series, lookahead_days)

        # second  get signal_series of -1s and 'clear' them
        neg_signal_series = (signal_series < 0) * -1
        clear_neg_signal_series = clear_signals(neg_signal_series, lookahead_days)
        
        clear_signal_series = clear_pos_signal_series + clear_neg_signal_series

        '''       
        if k == 0:
            print(i)
            print(signal_series)
            print(pos_signal_series)
            print(clear_pos_signal_series)
            print(neg_signal_series)
            print(clear_neg_signal_series)
            
        k = k + 1
        '''    
        return_df.loc[i] = clear_signal_series
    
    #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")   
    #print(return_df)
    
    return return_df.transpose()
    
def get_lookahead_prices(close, lookahead_days):
    """
    Get the lookahead prices for `lookahead_days` number of days.
    
    Parameters
    ----------
    close : DataFrame
        Close price for each ticker and date
    lookahead_days : int
        The number of days to look ahead
    
    Returns
    -------
    lookahead_prices : DataFrame
        The lookahead prices for each ticker and date
    """
    #TODO: Implement function
    
    
    return close.shift(-1 * lookahead_days)

def get_return_lookahead(close, lookahead_prices):
    """
    Calculate the log returns from the lookahead days to the signal day.
    
    Parameters
    ----------
    close : DataFrame
        Close price for each ticker and date
    lookahead_prices : DataFrame
        The lookahead prices for each ticker and date
    
    Returns
    -------
    lookahead_returns : DataFrame
        The lookahead log returns for each ticker and date
    """
    #TODO: Implement function
    
    return np.log(lookahead_prices / close)

def get_signal_return(signal, lookahead_returns):
    """
    Compute the signal returns.
    
    Parameters
    ----------
    signal : DataFrame
        The long, short, and do nothing signals for each ticker and date
    lookahead_returns : DataFrame
        The lookahead log returns for each ticker and date
    
    Returns
    -------
    signal_return : DataFrame
        Signal returns for each ticker and date
    """
    #TODO: Implement function
    #print(signal[:4])
    #print("....")
    #print(lookahead_returns[:4])
    
    return signal * lookahead_returns
    
from scipy.stats import kstest


def calculate_kstest(long_short_signal_returns):
    """
    Calculate the KS-Test against the signal returns with a long or short signal.
    
    Parameters
    ----------
    long_short_signal_returns : DataFrame
        The signal returns which have a signal.
        This DataFrame contains two columns, "ticker" and "signal_return"
    
    Returns
    -------
    ks_values : Pandas Series
        KS static for all the tickers
    p_values : Pandas Series
        P value for all the tickers
    """
    #TODO: Implement function
    
    ticker_groups = long_short_signal_returns.groupby('ticker')
    return_ks_values = pd.Series()
    retrn_p_values = pd.Series()
    all_returns = long_short_signal_returns['signal_return']
    normal_args = [np.mean(all_returns), np.std(all_returns, ddof=0)]
    for ticker, ticker_group in ticker_groups:
   
        #print(ticker_group['signal_return'])
    
        returns = ticker_group['signal_return']
    
        t_stat, p_value = kstest(returns, 'norm', normal_args)
        
        return_ks_values.loc[ticker] = t_stat
        retrn_p_values.loc[ticker] = p_value
        #print("Test statistic: {}, p-value: {}".format(t_stat, p_value))
        #print("Is the distribution Likely Normal? {}".format(p_value > 0.05))
    
    
    return return_ks_values, retrn_p_values
    
def find_outliers(ks_values, p_values, ks_threshold, pvalue_threshold=0.05):
    """
    Find outlying symbols using KS values and P-values
    
    Parameters
    ----------
    ks_values : Pandas Series
        KS static for all the tickers
    p_values : Pandas Series
        P value for all the tickers
    ks_threshold : float
        The threshold for the KS statistic
    pvalue_threshold : float
        The threshold for the p-value
    
    Returns
    -------
    outliers : set of str
        Symbols that are outliers
    """
    #TODO: Implement function
    returnset_ks = set()
    returnset_p = set()
    outliers_ks = ks_values.loc[ks_values > ks_threshold]
    outliers_p = p_values.loc[p_values < pvalue_threshold]
    
    return set(outliers_ks.index.intersection(outliers_p.index))
    
 # high.shift(1).rolling(lookback_days).max(), low.shift(1).rolling(lookback_days).min()
    
print(high[:6])
print(high.shift(1)[:6])
print(">>>")
print(high.shift(1).rolling(2))

