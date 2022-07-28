import numpy as np
import pandas as pd
from datetime import datetime,timedelta
data=pd.read_csv('/Users/zhongming/Desktop/风力发电/full.csv')


data =data[['TurbID','Day', 'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3','Prtv', 'Patv']]


data['Prtv'][data['Prtv']<0]=0
data['Patv'][data['Patv']<0]=0
#data['date'] = pd.to_datetime(data['date'])
conda=((data.Patv<=0)&(data.Wspd>2.5))|(data.Pab1>89)|(data.Pab2>89)|(data.Pab3>89)
indice=np.where(~conda)
list(*indice)
indice=np.where(~conda)
data_new=data.iloc[list(*indice)]
def pre_data(df):
   #去除异常值
    for col in df.columns.tolist():
        if   col != 'TurbID'  and col != 'Day' :
            df.drop(df[(df[col] > (df[col].mean() + 4 * df[col].std()))|(df[col] < (df[col].mean() - 4 * df[col].std()))].index,inplace=True)
    return df
df1=pre_data(data_new)
# #df_=df1[['TurbID', 'date', 'Day', 'Patv','Wspd', 'Prtv']]
# conda1=(df1.TurbID==24)|(df1.TurbID==25)|(df1.TurbID==38)|(df1.TurbID==40)|(df1.TurbID==61)|(df1.TurbID==68)|(df1.TurbID==121)|(df1.TurbID==122)
# indic=list(*np.where(~conda1))
# wind_data=df1.iloc[indic]
# t=wind_data.corr()
# print(t['Patv'].sort_values(ascending=False))
# features=['TurbID','Day','Wspd', 'Itmp', 'Prtv','Patv']
# wind_data=wind_data[features]
dd=df1.pivot_table(index='TurbID',columns='Day',aggfunc='mean')
dd.fillna(dd.mean(),inplace=True)



#dd.columns=[i+1 for i in range(len(dd.columns))]
data_len=85
start=0
data_sum={}
j=1
while data_len+start<=183:
    temp_df=pd.DataFrame()

    for i in range(10):
        tmp=dd.iloc[:,start+(i*183):start+(i*183)+data_len]

        temp_df=pd.concat([temp_df,tmp],axis=1)

    data_sum[j]=temp_df
    start+=7
    j+=1
