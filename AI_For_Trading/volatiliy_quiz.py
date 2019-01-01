import pandas as pd
import numpy as np

def get_most_volatile(prices):
    """Return the ticker symbol for the most volatile stock.
    
    Parameters
    ----------
    prices : pandas.DataFrame
        a pandas.DataFrame object with columns: ['ticker', 'date', 'price']
    
    Returns
    -------
    ticker : string
        ticker symbol for the most volatile stock
    """
    # TODO: Fill in this function.
    StockPrices = prices.pivot(index='date', columns='ticker', values='price')
    
    Aseries = StockPrices[['A']]  #dataframe
    Bseries = StockPrices[['B']]  #dataframe
    
    print(StockPrices.var().nlargest(1).index[0])
    
    pass


def test_run(filename='prices2.csv'):
    """Test run get_most_volatile() with stock prices from a file."""
    prices = pd.read_csv(filename, parse_dates=['date'])
    print("Most volatile stock: {}".format(get_most_volatile(prices)))


if __name__ == '__main__':
    test_run()

