import pandas as pd

from pricelevels.cluster import ZigZagClusterLevels
from pricelevels.visualization.levels_with_zigzag import plot_with_pivots

def support_resistence_levels(df):
    df.rename(columns = {'open' : 'Open','high':'High','low':'Low', 'close':"Close","volume": 'Volume', 'Date':"Datetime"}, inplace=True)

    zig_zag_percent = 0.8

    zl = ZigZagClusterLevels(peak_percent_delta=zig_zag_percent, merge_distance=None,
                            merge_percent=0.1, min_bars_between_peaks=10, peaks='All') # for window size 60
    zl.fit(df)

    return zl.levels

if __name__ == "__main__":
    df = pd.read_csv('robochartist/test/D_test.csv')
    df.rename(columns = {'open' : 'Open','high':'High','low':'Low', 'close':"Close","volume": 'Volume', 'Date':"Datetime"}, inplace=True)

    zig_zag_percent = 0.8

    zl = ZigZagClusterLevels(peak_percent_delta=zig_zag_percent, merge_distance=None,
                            merge_percent=0.1, min_bars_between_peaks=20, peaks='Low')

    zl.fit(df)

    plot_with_pivots(df['Close'].values, zl.levels, zig_zag_percent)