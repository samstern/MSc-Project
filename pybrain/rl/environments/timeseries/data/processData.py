import pandas as pd

#read in csv file where the dates are the keys
data=pd.read_csv('modelInputs.csv',parse_dates=['DATE'],index_col='DATE')

#insert a percenate returns column
data['RETURNS']=data['S&P PRICE'].pct_change()
data.shift()

#make sure data is complete
data=data.dropna()

#rearrange the ordering of the columns so that returns is the first one (makes it easer when casting to a matrix later)
cols=data.columns.tolist()
cols=cols[-1:]+cols[:-1]
data=data[cols]
data=data.drop('S&P PRICE',1)

rets=data['RETURNS']

