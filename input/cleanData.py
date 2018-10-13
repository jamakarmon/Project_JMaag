#Set snakemake
in_file = snakemake.input[0]
out_file = snakemake.output[0]
out_dir = snakemake.params[0]

#Load packages
import pandas as pd
from datetime import datetime

#Load dataframe. Note index_col=0 sets the first column to be the index
df = pd.read_csv(in_file,index_col=0)

#LISTS FOR LOOP

#List of all canton in Switzerland
#canton_list=df.sort_values(by=['canton'])['canton'].unique().tolist()
canton_list = ['Aargau','Appenzell Ausserrhoden', 'Appenzell Innerrhoden', 'Basel-Landschaft', 'Basel-Stadt', 'Bern', 'Freiburg', 'Genf', 'Glarus', 'Graubünden', 'Jura', 'Luzern', 'Neuenburg', 'Nidwalden', 'Obwalden', 'Sankt Gallen', 'Schaffhausen', 'Schwyz', 'Solothurn', 'Tessin', 'Thurgau', 'Uri', 'Waadt', 'Wallis', 'Zug', 'Zürich']
canton_abb_list = ['AG','AR','AI','BL','BS','BE','FR','GE','GL','GR','JU','LU','NE','NW','OW','SG','SH','SZ','SO','TI','TG','UR','VD','VS','ZG','ZH']
#canton_list[0:5]
#canton_abb_list[0:5]
#Number List
number_list = list(range(0,26))


#Try to loop the datacleaning, -manipulation process
#canton_raw = pd.DataFrame()
#empty canton_dataframe
df_by_canton = pd.DataFrame({'date': []})
for i in number_list:
    df_canton = df.loc[df['canton'] == canton_list[i]].sort_values(by=['date'])
    df_canton = df_canton.rename(columns={'search_wohnung' : canton_abb_list[i]})
    df_canton = df_canton.drop('canton', axis=1)
    df_by_canton = pd.merge(df_by_canton, df_canton, how='right', on=['date'])

#Set date to format and index
df_by_canton = df_by_canton.assign(dateindex=pd.to_datetime(df_by_canton['date'], format='%Y-%m-%d'))
df_by_canton.set_index(['dateindex'],inplace=True)
df_by_canton.sort_values(by='date')

#append new column with Date as integer from datetime package
start_date = min(pd.to_datetime(df_by_canton['date']))
df_by_canton['date_integer'] = pd.to_datetime(df_by_canton['date']) - start_date
df_by_canton.date_integer = df_by_canton.date_integer.dt.days + 1

#Change order of columns
cols = df_by_canton.columns.tolist()
n = int(cols.index('date'))
m = int(cols.index('date_integer'))
cols = [cols[n]] + [cols[m]] + cols[:n] + cols[n+1:m]
df_by_canton = df_by_canton[cols]

#Create two new datasets for ZH and BE
df_by_canton_ZH = df_by_canton[['date','date_integer','ZH']]
df_by_canton_BE = df_by_canton[['date','date_integer','BE']]

#Save new datasets to csv
df_by_canton.to_csv(out_file, sep=',')
df_by_canton_ZH.to_csv(out_dir + 'data_ZH.csv', sep=',')
df_by_canton_BE.to_csv(out_dir + 'data_BE.csv', sep=',')
