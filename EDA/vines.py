import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv('wine.csv')
profile = ProfileReport(df, title="Wine Pandas Profiling Report")
profile