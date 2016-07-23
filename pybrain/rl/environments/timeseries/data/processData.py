import pandas as pd

#read in csv file where the dates are the keys
data=pd.read_csv('data2.csv',parse_dates=['DATE'],index_col='DATE')

#insert a percenate returns column
data['RETURNS']=data['Price'].pct_change()
data.shift()

#make sure data is complete
data=data.dropna()

#rearrange the ordering of the columns so that returns is the first one (makes it easer when casting to a matrix later)
cols=data.columns.tolist()
cols=cols[-1:]+cols[:-1]
data=data[cols]
data=data.drop('Price',1)

rets=data['RETURNS']

