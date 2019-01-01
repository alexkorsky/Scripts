import pandas as pd

#price_df = pd.read_csv('prices.csv')
price_df = pd.read_csv('prices.csv', names=['ticker', 'date', 'open', 'high', 'low',
                                             'close', 'volume', 'adj_close', 'adj_volume'])

print(price_df)
print("\n")

# all columns median
print(price_df.median())
print("\n")

# medias grouped by ticker
print(price_df.groupby('ticker').median())
print("\n")

#print 11 first rows
print(price_df.iloc[:11])

#interestign PIVOT -- makes tickers colm headers abd dat row headers
open_prices = price_df.pivot(index='date', columns='ticker', values='open')
low_prices = price_df.pivot(index='date', columns='ticker', values='low')
high_prices = price_df.pivot(index='date', columns='ticker', values='high')
close_prices = price_df.pivot(index='date', columns='ticker', values='close')

print(open_prices)
#
#
#
# RESAMPLE Pandas usage: very cool
#
#
#
import numpy as np
import pandas as pd
print("\n----------------------\n")
dates = pd.date_range('10/10/2018', periods=11, freq='D')
print(dates)
print("\n")
close_prices = np.arange(len(dates))
print(close_prices)
print("\n")
close = pd.Series(close_prices, dates)
print(close)
print("\n")

# just every third day row
#This returns a DatetimeIndexResampler object. It's an intermediate object similar to the GroupBy object.
#Just like group by, it breaks the original data into groups. That means, we'll have 
#to apply an operation to these groups. Let's make it simple and get the first element from each group.
closeResample = close.resample('3D')
print(closeResample)
print("\n")
print(closeResample.first())

#The resample function shines when handling time and/or date specific tasks.
# In fact, you can't use this function if the index isn't a time-related class.

close = pd.Series(close_prices, dates)
w = pd.DataFrame({
    'days': close,
    'weeks': close.resample('W').first()})
    
print(w)

# to calculate returns it is cool tuse shift funciton. 
#Positive argument means shift into past, negtive menas shift into future
return_vals = (close - close.shift(1))/close.shift(1)
print("RETURNS:::::::")
print (return_vals)

#OHLC

print(close.resample('W').ohlc())

#open_prices_weekly = open_prices.resample('W').first()
#high_prices_weekly = high_prices.resample('W').max()
#low_prices_weekly = low_prices.resample('W').min()
#close_prices_weekly = close_prices.resample('W').last()