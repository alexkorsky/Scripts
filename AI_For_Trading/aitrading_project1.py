import pandas as pd


month = pd.to_datetime('02/01/2018')
close_month = pd.DataFrame(
    {
        'A': 1,
        'B': 12,
        'C': 35,
        'D': 3,
        'E': 79,
        'F': 2,
        'G': 15,
        'H': 59},
    [month])

print(close_month)

# get the Seires out of DataFrame first:
month_series = close_month.loc[month]
print(month_series)
print("\n")
print(month_series.nlargest(2))

#Project
price_df = pd.read_csv('prices.csv', names=['ticker', 'date', 'open', 'high', 'low',
                                             'close', 'volume', 'adj_close', 'adj_volume'])
print(price_df.index)
sector = pd.Series(("Sec1", "Sec2", "Sec3", "Sec4", "Sec5", "Sec6", "Sec7", "Sec8"),
                   ('ABC', 'EFG', 'XYZ', 'D', 'E', 'F', 'G', 'H'))
                   
print(sector) 
close_prices = price_df.pivot(index='date', columns='ticker', values='close')

print(close_prices)

date = "2017-09-05"
close_series = close_prices.loc[date]                  
print(close_series)

top_n = 2
print(close_series.nlargest(top_n))

# get the list of keys from nlargest Series
keys = close_series.nlargest(top_n).index
superkeys = close_series.index

#use keys as keyto Secotr Sries to get get series of sectors
secList = sector.loc[keys]

# just get the vlaues now to get the list of actual sectors
print(secList.values)

# convert it to set
print(set(secList.values))

# try to modify close_series in place -- sub largest with 1 and smallest with 0:
print("XXXXXXXXXXXXXXXXXXXXXXXXX")
print(close_series)
close_series.loc[keys] = 1
close_series.loc[~superkeys.isin(keys)] = 0
print(close_series)

#Key moment see of close_prices were modifed inplace:
print(close_prices)

def get_top_n(prev_returns, top_n):
    """
    Select the top performing stocks
    
    Parameters
    ----------
    prev_returns : DataFrame
        Previous shifted returns for each ticker and date
    top_n : int
        The number of top performing stocks to get
    
    Returns
    -------
    top_stocks : DataFrame
        Top stocks for each ticker and date marked with a 1
    """
    # TODO: Implement Function
    prev_returns2 = pd.DataFrame(dtype='int64', columns=prev_returns.columns)
    
    for i, date_series in prev_returns.iterrows():
        #date_series = prev_returns.loc[date]
        date_largest = date_series.nlargest(top_n)
        keys = date_largest.index # this is a list of top N stock names in this row
        superkeys = date_series.index
        
        #use keys to assign 1 to all TOP stocks
        s = pd.Series(dtype='int64', index = superkeys)        
        s.loc[superkeys] = int(0)
        s.loc[keys] = int(1)
        
        #use superkeys that are NOT IN keys to assing 0 to everyone else
        #s.loc[~superkeys.isin(keys)] = 0
        
        prev_returns2.loc[i] = s
                
    return prev_returns2.astype('int64')
#YES TEHY WERE!!!!!!!!!!!!!!