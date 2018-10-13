#Set snakemake
in_file = snakemake.input[0]
out_file = snakemake.output[0]
out_dir = snakemake.params[0]

#Import all packages and load file
import os
import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic, acf, pacf

from scipy.linalg import toeplitz

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
import matplotlib.dates

import datetime as dt
from datetime import datetime

import pickle


#Load data
df_search = pd.read_csv(in_file,index_col=0)


#Take rolling mean to reduce noise in data
df_search['ind_orig'] = df_search['BE']
df_search['ind_ma'] = df_search.rolling(window=7)['BE'].mean()
#Set first date back to 1
df_search['time'] = df_search['date_integer'] -6
df_search = df_search[6:]

#Create function to make summary tables, save plots as pdf, and tables as tex tabular

#Table Summary
def results_summary_to_df(results):
    '''This takes the result of a statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame({"pvals":pvals,
                               "coeff":coeff,
                               "conf_lower":conf_lower,
                               "conf_higher":conf_higher
                                })

    #Reordering...
    results_df = results_df[["coeff","pvals","conf_lower","conf_higher"]]
    return results_df

#Function to save table as tex tabular
def tableastxt(dftable,dfname,index_ft):
    dir = out_dir + 'BE/table'
    if not os.path.exists(dir):
            os.makedirs(dir)
    with open(dir + "/" + dfname + '.tex', 'w') as tf:
     tf.write(dftable.to_latex(index=index_ft))

#Function to save plot as pdf
def plotaspdf(figplot, figname):
    dir = out_dir + 'BE/plot'
    if not os.path.exists(dir):
            os.makedirs(dir)
    fig = figplot
    fig.savefig(dir + "/" + figname + ".pdf")

#Plot original index

#Plot
figOrig, axOrig = plt.subplots(1,figsize=(10, 6))

#Set plots
axOrig.plot(pd.to_datetime(df_search['date']), df_search.ind_ma, color = 'gray', label = 'Index Values')

axOrig.xaxis.set_major_locator(matplotlib.dates.YearLocator())
axOrig.xaxis.set_minor_locator(matplotlib.dates.MonthLocator((1,4,7,10)))

axOrig.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("\n%Y"))
axOrig.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter("%b"))
plt.setp(axOrig.get_xticklabels(), rotation=0, ha="center")

#Set labels
plt.title("Search Term 'Wohnung'")
plt.xlabel('time')
plt.ylabel('quantity')
plt.legend()
plt.ylim(ymin = 0, ymax = 50)

#Save original plot
plotaspdf(figOrig, 'figOrig')

#TREND
#Choose trend model based on lowest bic

#Set dataset for trend
df_search_trend = df_search.copy()
df_search_trend['time^2'] = df_search_trend.time**2


#Model 1
Ytrend = df_search_trend['ind_ma']
Xtrend1 = sm.add_constant(df_search_trend['time'])
trend1 = sm.OLS(Ytrend[:-7],Xtrend1[:-7]).fit()

#Model 2
Xtrend2 = sm.add_constant(df_search_trend[['time','time^2']])
trend2 = sm.OLS(Ytrend[:-7],Xtrend2[:-7]).fit()

#Model 3
Xtrend3 = sm.add_constant(df_search_trend['time^2'])
trend3 = sm.OLS(Ytrend[:-7],Xtrend3[:-7]).fit()
#print(model1.summary())

#Winnermodel selection
trendmodels = [(trend1,Xtrend1),(trend2,Xtrend2),(trend3,Xtrend3)]
wintrendmodel=trendmodels[np.argmin([x.bic for x in list(zip(*trendmodels))[0]])]
print("trend BIC", wintrendmodel[0].bic)

#Add modelled trend to data
#Subtract modelled trend from original to get detrended series
df_search["trend"]=wintrendmodel[0].predict(wintrendmodel[1])
df_search["ind_deT"]=df_search['ind_ma']-df_search["trend"]

#Inspect modelled Trend
#Table Summary
trend_table=results_summary_to_df(wintrendmodel[0])

#Save trend table
tableastxt(trend_table,'trend_table',True)

#Plot
figTrend, axTrend = plt.subplots(1,figsize=(10, 6))

#Set plots
axTrend.plot(pd.to_datetime(df_search['date'])[:-7], df_search.ind_ma[:-7], color = 'gray', label = 'Index Values')
axTrend.plot(pd.to_datetime(df_search['date'])[:-7], df_search.trend[:-7], color = 'black', label = 'Trend')

axTrend.xaxis.set_major_locator(matplotlib.dates.YearLocator())
axTrend.xaxis.set_minor_locator(matplotlib.dates.MonthLocator((1,4,7,10)))

axTrend.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("\n%Y"))
axTrend.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter("%b"))
plt.setp(axTrend.get_xticklabels(), rotation=0, ha="center")

#Set labels
plt.title("Search Term 'Wohnung'")
plt.xlabel('time')
plt.ylabel('quantity')
plt.legend()
plt.ylim(ymin = 0, ymax = 50)

#Save trend plot
plotaspdf(figTrend, 'figTrend')

#Save Trend Model to file
pickle.dump(wintrendmodel[0], open(out_dir + 'BE/trend_model.sav', 'wb'))

#Inspect data for season

#Set dataset for season
df_search_season = df_search.copy()

#First set year and create new dataset by year
df_search_season['year_integer'] = pd.to_datetime(df_search_season["date"]).dt.year
df_search_season['dayofyear_integer'] = pd.to_datetime(df_search_season["date"]).dt.dayofyear

year2015 = df_search_season.loc[df_search_season['year_integer'] == 2015]
year2016 = df_search_season.loc[df_search_season['year_integer'] == 2016]
year2017 = df_search_season.loc[df_search_season['year_integer'] == 2017]
year2018 = df_search_season.loc[df_search_season['year_integer'] == 2018][:-7]


figSeason, axSeason = plt.subplots(1,figsize=(10, 6))

#Set plots
axSeason.plot(year2015['dayofyear_integer'], year2015['ind_deT'], color = 'blue', label = 'year 2015')
axSeason.plot(year2016['dayofyear_integer'], year2016['ind_deT'], color = 'red', label = 'year 2016')
axSeason.plot(year2017['dayofyear_integer'], year2017['ind_deT'], color = 'green', label = 'year 2017')
axSeason.plot(year2018['dayofyear_integer'], year2018['ind_deT'], color = 'yellow', label = 'year 2018')
axSeason.plot(year2016['dayofyear_integer'], [0]*len(year2016['ind_deT']), color = 'black')


# Set ticks
major_ticks = list(year2016.loc[pd.to_datetime(year2016["date"]).dt.is_month_start == True].append(year2016[-1:])['dayofyear_integer'])
minor_ticks=list((np.array(major_ticks[:-1])+np.array(major_ticks[1:]))/2)
minor_ticks_label = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
axSeason.set_xticks(major_ticks)
axSeason.set_xticklabels('')
axSeason.tick_params(length=0, which = 'minor')
axSeason.set_xticks(minor_ticks, minor=True)
axSeason.set_xticklabels(minor_ticks_label, minor=True)

#Set labels
plt.title("Search Term 'Wohnung'")
plt.xlabel('time')
plt.ylabel('quantity')
plt.legend()
plt.ylim(ymin = -30, ymax = 30)


#Save season plot
plotaspdf(figSeason, 'figSeason')

#SEASON

#Set dataset for season
df_search_season = df_search.copy()

#Append new column, that gives the month as integer
df_search_season['month'] = pd.to_datetime(df_search_season["date"]).dt.month


#Add seasonal mean as a new column
df_search_season['season_mean']=df_search_season[:-7].groupby('month')['ind_deT'].transform('mean')
df_search_season['season_mean']=df_search_season.groupby('month')['season_mean'].transform('mean')

#Subtract modelled season from original to get deseasonalized series
df_search['season']=df_search_season['season_mean']
df_search['ind_deS']=df_search['ind_deT']-df_search['season']

#ERROR
#Check error component first for stationarity with plot

#Plot
figError, axError = plt.subplots(1,figsize=(10, 6))

#Set plots
axError.plot(pd.to_datetime(df_search['date'])[:-7], df_search.ind_deS[:-7], color = 'gray', label = 'Index Values')
axError.plot(pd.to_datetime(df_search['date'])[:-7], [0]*len(df_search.ind_deS[:-7]), color = 'black')

axError.xaxis.set_major_locator(matplotlib.dates.YearLocator())
axError.xaxis.set_minor_locator(matplotlib.dates.MonthLocator((1,4,7,10)))

axError.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("\n%Y"))
axError.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter("%b"))
plt.setp(axError.get_xticklabels(), rotation=0, ha="center")

#Set labels
plt.title("Search Term 'Wohnung'")
plt.xlabel('time')
plt.ylabel('quantity')
plt.legend()
plt.ylim(ymin = -15, ymax = 15)


#Save error plot
plotaspdf(figError, 'figError')

#Set dataset for error
df_search_error= df_search.copy()


#Vectorcorrelation Table
#Specify lags
lag_p=7
lag_q=7
#Specify dataset
ind_deS_val= df_search_error['ind_deS'][:-7].values
#Make table
vectorcorr_table = [[None for i in range(lag_q+1)] for j in range(lag_p+1)]
acf_list=list(acf(ind_deS_val, nlags=(lag_q+lag_p)))[1:]
acf_list = acf_list[::-1] + [1] + acf_list
ref_list_num=list(range(-(lag_q+lag_p),(lag_q+lag_p+1)))[::-1]
ref_list_denum=list(range(-(lag_q+lag_p),(lag_q+lag_p+1)))

for p in range(lag_p):
    for q in range(lag_q):
        val1=acf_list[ref_list_num.index(q+1):ref_list_num.index(q-p+1)+1]
        val2=acf_list[ref_list_denum.index(q+1):ref_list_denum.index(q+p+1)+1]
        val3=acf_list[ref_list_denum.index(0):ref_list_denum.index(p)+1]

        vectorcorr=((-1)**p)*np.linalg.det(toeplitz(val1, val2))/np.linalg.det(toeplitz(val3, val3))
        vectorcorr_table[p+1][q+1]=round(vectorcorr,3)
        vectorcorr_table[0][q+1]=str(q)
        vectorcorr_table[p+1][0]=str(p)
        vectorcorr_table[0][0]="p/q"
vectorcorr_table
vectorcorr_df=pd.DataFrame(vectorcorr_table,columns=vectorcorr_table[0]).iloc[1:]#.set_index("p/q")

#Save vectorcorr table
tableastxt(vectorcorr_df,'vectorcorr_df',False)

#Plot ACF and PACF
#Calculate acf as values
acf_results_deS = acf(df_search_error['ind_deS'][:-7], nlags=20)
pacf_results_deS = pacf(df_search_error['ind_deS'][:-7], nlags=20, method = 'ols')

#Plot ACF
figACF, axACF = plt.subplots(1,figsize=(10, 6))

axACF.plot(acf_results_deS, color = 'black')
axACF.axhline(y=0,color='gray')
axACF.axhline(y=-1.96/np.sqrt(len(df_search_error['ind_deS'][:-7])), color='gray',linestyle = 'dashed')
axACF.axhline(y=+1.96/np.sqrt(len(df_search_error['ind_deS'][:-7])), color='gray',linestyle = 'dashed')
plt.title('Autocorrelation')
plt.xlabel('lag')
plt.ylabel('ACF')
#Set ticks
axACF.set_xticks(list(range(1,len(acf_results_deS)+1)))
axACF.set_xticklabels([str(x) for x in list(np.array(range(1,len(acf_results_deS)+1))-1)])

figPACF, axPACF = plt.subplots(1,figsize=(10,6))
#Plot PACF
axPACF.plot(pacf_results_deS, color = 'black')
axPACF.axhline(y=0,color='gray')
axPACF.axhline(y=-1.96/np.sqrt(len(df_search_error['ind_deS'][:-7])), color='gray',linestyle = 'dashed')
axPACF.axhline(y=1.96/np.sqrt(len(df_search_error['ind_deS'][:-7])), color='gray',linestyle = 'dashed')
plt.title('Partial Autocorrelation')
plt.xlabel('lag')
plt.ylabel('PACF')
#Set ticks
axPACF.set_xticks(list(range(1,len(pacf_results_deS)+1)))
axPACF.set_xticklabels([str(x) for x in list(np.array(range(1,len(pacf_results_deS)+1))-1)])

#Save ACF and PACF of Error plot
plotaspdf(figACF, 'figACF')
plotaspdf(figPACF, 'figPACF')

#AICc Table
#Specify lags
order_p=3
order_q=3
#Specify dataset
ind_deS_df= df_search_error['ind_deS'][:-7]
#Make table
order_p=8
order_q=8
aicc_table = [[None for i in range(order_q)] for j in range(order_p)]

for p in range(1,order_p):
    for q in range(1,order_q):
        error = ARIMA(ind_deS_df, order=(p,0,q),freq='D')
        error_fit = error.fit(disp=0)
        aic=error_fit.aic
        correction=2*((p+q)*(p+q+1))/(len(ind_deS_df)-p-q-1)
        aicc_table[p][q]=round(aic+correction,2)
        #aicc_table[p][q]=error_fit.aic
        aicc_table[0][q]=str(q)
        aicc_table[p][0]=str(p)
        aicc_table[0][0]="p/q"
aicc_table
aicc_df=pd.DataFrame(aicc_table,columns=aicc_table[0]).iloc[1:]#.set_index("p/q")

#Save AICc table
tableastxt(aicc_df,'aicc_df',False)

#Get p and q order with lowest AICc
min_set=[sublist[1:] for sublist in aicc_table][1:]
def min_value(min_list1):
    return min((n, i+1, j+1) for i, min_list2 in enumerate(min_list1)  for j, n in enumerate(min_list2))[1:]
print("ARMA orders", min_value(min_set))

#Calculate Error Model based on ACF, PACF Plot, Vectorcorr Table and AICc inspection
#rename ind_deS for output result
df_search_error['index']=df_search_error['ind_deS']
error = ARIMA(df_search_error['index'][:-7], order=(1,0,6),freq='D').fit(disp=0)

#Get error results as dataframe
error_table=results_summary_to_df(error)

#Save error results table
tableastxt(error_table,'error_table',True)

#Save error Model to file
pickle.dump(error, open(out_dir + 'BE/error_model.sav', 'wb'))

#Add modelled error to data
#Subtract modelled error from original to get white noise

#1. Get predicted values for insamples period
insample_values = error.predict()
#2. Get forecasted values for out of sample period
start_index = pd.to_datetime(df_search_error['date'][-7])
end_index = pd.to_datetime(df_search_error['date'][-1])
outsample_values = error.predict(start=start_index, end=end_index)
#3. Append modelled error to original dataframe
df_search['error'] = insample_values.append(outsample_values)
#4. Subtract modelled error from original to get white noise
df_search['wn']=df_search['ind_deS']-df_search['error']


#WHTE NOISE
#Check ACF and PACF of White Noise

#ACF
acf_results_WN = acf(error.resid, nlags=20)
pacf_results_WN = pacf(error.resid, nlags=20, method = 'ols')

#Plot ACF
figACF_WN, axACF_WN = plt.subplots(1,figsize=(10,6))

axACF_WN.plot(acf_results_WN, color = 'black')
axACF_WN.axhline(y=0,color='gray')
axACF_WN.axhline(y=-1.96/np.sqrt(len(error.resid)), color='gray',linestyle = 'dashed')
axACF_WN.axhline(y=+1.96/np.sqrt(len(error.resid)), color='gray',linestyle = 'dashed')
plt.title('Autocorrelation')
plt.xlabel('lag')
plt.ylabel('ACF')
#Set ticks
axACF_WN.set_xticks(list(range(1,len(acf_results_WN)+1)))
axACF_WN.set_xticklabels([str(x) for x in list(np.array(range(1,len(acf_results_WN)+1))-1)])

figPACF_WN, axPACF_WN = plt.subplots(1,figsize=(10,6))
#Plot PACF
axPACF_WN.plot(pacf_results_WN, color = 'black')
axPACF_WN.axhline(y=0,color='gray')
axPACF_WN.axhline(y=-1.96/np.sqrt(len(error.resid)), color='gray',linestyle = 'dashed')
axPACF_WN.axhline(y=1.96/np.sqrt(len(error.resid)), color='gray',linestyle = 'dashed')
plt.title('Partial Autocorrelation')
plt.xlabel('lag')
plt.ylabel('PACF')
#Set ticks
axPACF_WN.set_xticks(list(range(1,len(pacf_results_WN)+1)))
axPACF_WN.set_xticklabels([str(x) for x in list(np.array(range(1,len(pacf_results_WN)+1))-1)])

#Save ACF and PACF of White Noise plot
plotaspdf(figACF_WN, 'figACF_WN')
plotaspdf(figPACF_WN, 'figPACF_WN')

#Add modelled values to dataset
df_search['ind_modelled']=df_search['trend']+df_search['season']+df_search['error']

#Plot
figModel, axModel = plt.subplots(1,figsize=(10,6))

#Set plots
axModel.plot(pd.to_datetime(df_search['date']), df_search.ind_ma, color = 'gray', label = 'Index Values')
axModel.plot(pd.to_datetime(df_search['date']), df_search.ind_modelled, color = 'red', label = 'Model Values')

axModel.xaxis.set_major_locator(matplotlib.dates.YearLocator())
axModel.xaxis.set_minor_locator(matplotlib.dates.MonthLocator((1,4,7,10)))

axModel.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("\n%Y"))
axModel.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter("%b"))
plt.setp(axTrend.get_xticklabels(), rotation=0, ha="center")

#Set labels
plt.title("Search Term 'Wohnung'")
plt.xlabel('time')
plt.ylabel('quantity')
plt.legend()
plt.ylim(ymin = 0, ymax = 50)


#Save Model plot
plotaspdf(figModel, 'figModel')

#Calculation of mean absolute percentage error for in and out of sample
dev=abs((df_search.ind_modelled-df_search.ind_ma)/df_search.ind_modelled)
insample_MAPE=100*(dev[:-7].mean())/len(dev[:-7])
outsample_MAPE=100*(dev[-7:].mean())/len(dev[-7:])
print("insample_MAPE", insample_MAPE)
print("outsample_MAPE", outsample_MAPE)

#Save datafile to csv
df_search.to_csv(out_file, sep=',')
