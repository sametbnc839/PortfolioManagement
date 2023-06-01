import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pypfopt.risk_models import CovarianceShrinkage
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pypfopt.expected_returns import mean_historical_return
from pypfopt import expected_returns
from pypfopt import plotting
from pypfopt import risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation
from pypfopt.discrete_allocation import get_latest_prices
from pypfopt import HRPOpt
from pypfopt.efficient_frontier import EfficientCVaR
from scipy import stats
import yfinance as yf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


df = pd.read_csv(r'MainDataset.csv', parse_dates=True)

date = df['Date']
date = date[::-1]
date = pd.to_datetime(date,dayfirst = True)
# date2 = pd.to_datetime(date, format='%Y')

snp = df['S&P 500 Price']
snp = snp[::-1]
# fig, axsnp = plt.subplots()
# axsnp.plot(date,snp)
# axsnp.set_title('S&P 500')
# axsnp.set_xlabel('Date')
# axsnp.set_ylabel('S&P 500')

exxon = df['Exxon']
exxon = exxon[::-1]
# fig, axexxon = plt.subplots()
# axexxon.plot(date,exxon)
# axexxon.set_title('Exxon')
# axexxon.set_xlabel('Date')
# axexxon.set_ylabel('Exxon')

chevron = df['Chevron']
chevron = chevron[::-1]
# fig, axchevron = plt.subplots()
# axchevron.plot(date,chevron)
# axchevron.set_title('Chevron')
# axchevron.set_xlabel('Date')
# axchevron.set_ylabel('Chevron')


riotinto = df['Rio Tinto Price']
riotinto = riotinto[::-1]
# fig, axriotinto = plt.subplots()
# axriotinto.plot(date,riotinto)
# axriotinto.set_title('Rio Tinto Price')
# axriotinto.set_xlabel('Date')
# axriotinto.set_ylabel('Rio Tinto Price')

bhp = df['BHP']
bhp = bhp[::-1]
# fig, axbhp = plt.subplots()
# axbhp.plot(date,bhp)
# axbhp.set_title('BHP')
# axbhp.set_xlabel('Date')
# axbhp.set_ylabel('BHP')

glencore = df['Glencore']
glencore = glencore[::-1]
# fig, axglencore = plt.subplots()
# axglencore.plot(date,glencore)
# axglencore.set_title('Glencore')
# axglencore.set_xlabel('Date')
# axglencore.set_ylabel('Glencore')

brentoil = df['Brent Oil']
brentoil = brentoil[::-1]
# fig, axbrentoil = plt.subplots()
# axbrentoil.plot(date,brentoil)
# axbrentoil.set_title('Brent Oil')
# axbrentoil.set_xlabel('Date')
# axbrentoil.set_ylabel('Brent Oil')

coal = df['Coal']
coal = coal[::-1]
# fig, axcoal = plt.subplots()
# axcoal.plot(date,coal)
# axcoal.set_title('Coal')
# axcoal.set_xlabel('Date')
# axcoal.set_ylabel('Coal')

ttfgas = df['TFF Gas']
ttfgas = ttfgas[::-1]
# fig, axtffgas = plt.subplots()
# axtffgas.plot(date,tffgas)
# axtffgas.set_title('TFF Gas')
# axtffgas.set_xlabel('Date')
# axtffgas.set_ylabel('TFF Gas')

nickel = df['Nickel Price']
nickel = nickel[::-1]
# fig, axnickel = plt.subplots()
# axnickel.plot(date,nickel)
# axnickel.set_title('Nickel Price')
# axnickel.set_xlabel('Date')
# axnickel.set_ylabel('Nickel Price')

copper = df['Copper Price']
copper = copper[::-1]
# fig, axcopper = plt.subplots()
# axcopper.plot(date,copper)
# axcopper.set_title('Copper Price')
# axcopper.set_xlabel('Date')
# axcopper.set_ylabel('Copper Price')

msciworld = df['MSCI World Index']
msciworld = msciworld[::-1]
# fig, axmsciworld = plt.subplots()
# axmsciworld.plot(date,msciworld)
# axmsciworld.set_title('MSCI World Index')
# axmsciworld.set_xlabel('Date')
# axmsciworld.set_ylabel('MSCI World Index')

msciemerging = df['MSCI Emerging Markets']
msciemerging = msciemerging[::-1]
# fig, axmsciemerging = plt.subplots()
# axmsciemerging.plot(date,msciemerging)
# axmsciemerging.set_title('MSCI Emerging Markets')
# axmsciemerging.set_xlabel('Date')
# axmsciemerging.set_ylabel('MSCI Emerging Markets')

russel2kv = df['Russel 2000 V']
russel2kv = russel2kv[::-1]
# fig, axrussel2kv = plt.subplots()
# axrussel2kv.plot(date,russel2kv)
# axrussel2kv.set_title('Russel 2000 V')
# axrussel2kv.set_xlabel('Date')
# axrussel2kv.set_ylabel('Russel 2000 V')

russel2kvgrowth = df['Russel 2000 Growth Index']
russel2kvgrowth = russel2kvgrowth[::-1]
# fig, axrussel2kvgrowth = plt.subplots()
# axrussel2kvgrowth.plot(date,russel2kvgrowth)
# axrussel2kvgrowth.set_title('Russel 2000 Growth Index')
# axrussel2kvgrowth.set_xlabel('Date')
# axrussel2kvgrowth.set_ylabel('Russel 2000 Growth Index')

msciusstock = df['MSCI US Stock Market']
msciusstock = msciusstock[::-1]
# fig, axmsciusstock = plt.subplots()
# axmsciusstock.plot(date,msciusstock)
# axmsciusstock.set_title('MSCI US Stock Market')
# axmsciusstock.set_xlabel('Date')
# axmsciusstock.set_ylabel('MSCI US Stock Market')

ftseemerging = df['FTSE Emerging']
ftseemerging = ftseemerging[::-1]
# fig, axftseemerging = plt.subplots()
# axftseemerging.plot(date,ftseemerging)
# axftseemerging.set_title('FTSE Emerging')
# axftseemerging.set_xlabel('Date')
# axftseemerging.set_ylabel('FTSE Emerging')

ftsedeveloped = df['FTSE Developed']
ftsedeveloped = ftsedeveloped[::-1]
# fig, axftsedeveloped = plt.subplots()
# axftsedeveloped.plot(date,ftsedeveloped)
# axftsedeveloped.set_title('FTSE Developed')
# axftsedeveloped.set_xlabel('Date')
# axftsedeveloped.set_ylabel('FTSE Developed')

ftseglobal = df['FTSE Global']
ftseglobal = ftseglobal[::-1]
# fig, axftseglobal = plt.subplots()
# axftseglobal.plot(date,ftseglobal)
# axftseglobal.set_title('FTSE Global')
# axftseglobal.set_xlabel('Date')
# axftseglobal.set_ylabel('FTSE Global')

amazon = df['Amazon']
amazon = amazon[::-1]
# fig, axamazon = plt.subplots()
# axamazon.plot(date,amazon)
# axamazon.set_title('Amazon')
# axamazon.set_xlabel('Date')
# axamazon.set_ylabel('Amazon')

microsoft = df['Microsoft']
microsoft = microsoft[::-1]
# fig, axmicrosoft = plt.subplots()
# axmicrosoft.plot(date,microsoft)
# axmicrosoft.set_title('Microsoft')
# axmicrosoft.set_xlabel('Date')
# axmicrosoft.set_ylabel('Microsoft')

# plt.plot(date,amazon,marker='o', markersize=1.7, color="green", alpha=0.5, label='Amazon')
# plt.plot(date,microsoft,marker='o', markersize=1.7, color="red", alpha=0.5, label='Microsoft')
# plt.title('Technology Equities')
# plt.legend(markerscale=8)

# nickel = nickel/100
# plt.plot(date,copper,marker='o', markersize=1.7, color="green", alpha=0.5, label='Copper Price')
# plt.plot(date,nickel,marker='o', markersize=1.7, color="red", alpha=0.5, label='Nickel Price')
# plt.plot(date,brentoil,marker='o', markersize=1.7, color="blue", alpha=0.5, label='Brent Oil')
# plt.plot(date,ttfgas,marker='o', markersize=1.7, color="yellow", alpha=0.5, label='TTF Gas')
# plt.plot(date,coal,marker='o', markersize=1.7, color="purple", alpha=0.5, label='Coal')
# plt.title('Commodities')
# plt.legend(markerscale=8)

# plt.plot(date,msciworld,marker='o', markersize=1.7, color="green", alpha=0.5, label='MSCI World Index')
# plt.plot(date,msciusstock,marker='o', markersize=1.7, color="red", alpha=0.5, label='MSCI US')
# plt.plot(date,msciemerging,marker='o', markersize=1.7, color="blue", alpha=0.5, label='MSCI Emerging Markets')
# plt.title('MSCI')
# plt.legend(markerscale=8)

# plt.plot(date,russel2kv,marker='o', markersize=1.7, color="green", alpha=0.5, label='Russel 2000 V')
# plt.plot(date,russel2kvgrowth,marker='o', markersize=1.7, color="red", alpha=0.5, label='Russel 2000 Growth Index')
# plt.title('Russel 2000 Index')
# plt.legend(markerscale=8)

# plt.plot(date,ftseglobal,marker='o', markersize=1.7, color="green", alpha=0.5, label='FTSE Global')
# plt.plot(date,ftseemerging,marker='o', markersize=1.7, color="red", alpha=0.5, label='FTSE Emerging')
# plt.plot(date,ftsedeveloped,marker='o', markersize=1.7, color="blue", alpha=0.5, label='FTSE Developed')
# plt.title('FTSE Index')
# plt.legend(markerscale=8)

# plt.bar(date,snp)
# plt.title("S&P 500 Bar Chart")
# plt.xlabel("Date")
# plt.ylabel("S&P 500")
# glencore = glencore/10
# snp = snp/100
# riotinto = riotinto/100
# plt.plot(date,exxon,marker='o', markersize=1.7, color="green", alpha=0.5, label='Exxon')
# plt.plot(date,chevron,marker='o', markersize=1.7, color="red", alpha=0.5, label='Chevron')
# plt.plot(date,snp,marker='o', markersize=1.7, color="blue", alpha=0.5, label='S&P 500')
# plt.plot(date,riotinto,marker='o', markersize=1.7, color="yellow", alpha=0.5, label='Rio Tinto')
# plt.plot(date,glencore,marker='o', markersize=1.7, color="gray", alpha=0.5, label='Glencore')
# plt.title('Market Indices')
# plt.legend(markerscale=8)

# plt.plot(date,exxon,marker='o', markersize=1.7, color="green", alpha=0.5, label='Exxon')
# plt.plot(date,chevron,marker='o', markersize=1.7, color="red", alpha=0.5, label='Chevron')
# plt.plot(date,glencore,marker='o', markersize=1.7, color="gray", alpha=0.5, label='Glencore')
# plt.title('Market Indices 2')
# plt.legend(markerscale=8)

# firstday = [exxon[0],chevron[0],glencore[0],riotinto[0],snp[0]]
# lastday = [exxon[2972],chevron[2972],glencore[2972],riotinto[2972],snp[2972]]
# labels = ['Exxon', 'Chevron', 'Glencore', 'Rio Tinto', 'S&P 500']
# plt.pie(firstday, labels=labels)
# plt.title('Market Indices at First Day of Dataset')
# plt.pie(lastday, labels=labels)
# plt.title('Market Indices at Last Day of Dataset')

# plt.bar(date,exxon)
# plt.plot(date,chevron)

# plt.boxplot([coal,copper,brentoil], labels=['Coal','Copper','Brent Oil'])
# plt.title('Commodities Boxplot')

# plt.boxplot([nickel],labels=['Nickel'])
# plt.title('Nickel Boxplot')

# plt.boxplot([snp],labels=['S&P 500'])
# plt.title('S&P 500 Boxplot')

# plt.boxplot([chevron,exxon],labels=['Chevron','Exxon'])
# plt.title('Chevron and Exxon Boxplot')

# plt.boxplot([riotinto,bhp],labels=['Rio Tinto','BHP'])
# plt.title('Rio Tinto and BHP Boxplot')

# plt.boxplot([glencore],labels=['Glencore'])
# plt.title('Glencore Boxplot')

# plt.boxplot([msciworld,msciemerging],labels=['MSCI World Index','MSCI Emerging Markets',])
# plt.title('MSCI World Index and MSCI Emerging Markets Boxplot')

# plt.boxplot([russel2kv,russel2kvgrowth],labels=['Russel 2000 V','Russel 2000 Growth Index',])
# plt.title('Russel 2000 V and Russel 2000 Growth Index Boxplot')

# plt.boxplot([msciusstock],labels=['MSCI US Stock Market'])
# plt.title('MSCI US Stock Market Boxplot')

# plt.boxplot([ftsedeveloped],labels=['FTSE Developed'])
# plt.title('FTSE Developed Boxplot')

# plt.boxplot([ftseglobal],labels=['FTSE Global'])
# plt.title('FTSE Global Boxplot')

# plt.boxplot([microsoft,amazon],labels=['Microsoft','Amazon',])
# plt.title('Microsoft and Amazon Boxplot')

###Oulier Detection###

no_outlier_snp= snp.copy()
no_outlier_exxon = exxon.copy()
no_outlier_chevron = chevron.copy()
no_outlier_riotinto = riotinto.copy()
no_outlier_bhp = bhp.copy()
no_outlier_glencore = glencore.copy()
no_outlier_brentoil = brentoil.copy()
no_outlier_coal = coal.copy()
no_outlier_ttfgas = ttfgas.copy()
no_outlier_nickel = nickel.copy()
no_outlier_copper = copper.copy()
no_outlier_msciworld = msciworld.copy()
no_outlier_msciemerging = msciemerging.copy()
no_outlier_russel2kv = russel2kv.copy()
no_outlier_russel2kvgrowth = russel2kvgrowth.copy()
no_outlier_mscius= msciusstock.copy()
no_outlier_ftseemerging = ftseemerging.copy()
no_outlier_ftsedeveloped = ftsedeveloped.copy()
no_outlier_ftseglobal = ftseglobal.copy()
no_outlier_amazon = amazon.copy()
no_outlier_microsoft = microsoft.copy()

for x in range(len(no_outlier_exxon)):
    if (no_outlier_exxon[x] < 42.67):
        no_outlier_exxon[x] = 42.67
    if (no_outlier_exxon[x] > 76.3):
        no_outlier_exxon[x] = 76.3
# plt.boxplot([exxon,no_outlier_exxon],labels=['Exxon','No Outlier Exxon'])
# plt.title('Exxon Boxplot Before and After The Outlier Detection')
# plt.plot(date,no_outlier_exxon)
# plt.title('No Outlier Exxon')

for x in range(len(no_outlier_chevron)):
    if (no_outlier_chevron[x] > 133.69):
        no_outlier_chevron[x] = 133.69
# plt.boxplot([chevron,no_outlier_chevron],labels=['Chevron','No Outlier Chevron'])
# plt.title('Chevron Boxplot Before and After The Outlier Detection')
# plt.plot(date,no_outlier_chevron)
# plt.title('No Outlier Chevron')

for x in range(len(no_outlier_riotinto)):
    if (no_outlier_riotinto[x] > 5826.41):
        no_outlier_riotinto[x] = 5826.41
# plt.boxplot([riotinto,no_outlier_riotinto],labels=['Rio Tinto','No Outlier Rio Tinto'])
# plt.title('Rio Tinto Boxplot Before and After The Outlier Detection')
# plt.plot(date,no_outlier_chevron)
# plt.title('No Outlier Rio Tinto')

for x in range(len(no_outlier_bhp)):
    if (no_outlier_bhp[x] > 2090.2):
        no_outlier_bhp[x] = 2090.2
    if (no_outlier_bhp[x] < 521.3):
        no_outlier_bhp[x] = 521.3
# plt.boxplot([bhp,no_outlier_bhp],labels=['BHP','No Outlier BHP'])
# plt.title('BHP Boxplot Before and After The Outlier Detection')
# plt.plot(date,no_outlier_chevron)
# plt.title('No Outlier BHP')

for x in range(len(no_outlier_glencore)):
    if (no_outlier_glencore[x] > 396.77):
        no_outlier_glencore[x] = 396.77
    if (no_outlier_glencore[x] < 107.59):
        no_outlier_glencore[x] = 107.59
# plt.boxplot([glencore,no_outlier_glencore],labels=['Glencore','No Outlier Glencore'])
# plt.title('Glencore Boxplot Before and After The Outlier Detection')
# plt.plot(date,no_outlier_glencore)
# plt.title('No Outlier Glencore')

for x in range(len(no_outlier_coal)):
    if (no_outlier_coal[x] > 173.2):
        no_outlier_coal[x] = 173.2
# plt.boxplot([coal,no_outlier_coal],labels=['Coal','No Outlier Coal'])
# plt.title('Coal Boxplot Before and After The Outlier Detection')
# plt.plot(date,no_outlier_brentoil)
# plt.title('No Outlier Coal')

for x in range(len(no_outlier_ttfgas)):
    if (no_outlier_ttfgas[x] > 173.2):
        no_outlier_ttfgas[x] = 173.2
# plt.boxplot([ttfgas,no_outlier_ttfgas],labels=['TTF Gas','No Outlier TTF Gas'])
# plt.title('TTF Gas Boxplot Before and After The Outlier Detection')
# plt.plot(date,no_outlier_ttfgas)
# plt.title('No Outlier TTF Gas')

for x in range(len(no_outlier_nickel)):
    if (no_outlier_nickel[x] > 27461.5):
        no_outlier_nickel[x] = 27461.5
# plt.boxplot([nickel,no_outlier_nickel],labels=['Nickel','No Outlier Nickel'])
# plt.title('Nickel Boxplot Before and After The Outlier Detection')
# plt.plot(date,no_outlier_nickel)
# plt.title('No Outlier Nickel')

for x in range(len(no_outlier_msciworld)):
    if (no_outlier_msciworld[x] > 3221.14):
        no_outlier_msciworld[x] = 3221.14
# plt.boxplot([msciworld,no_outlier_msciworld],labels=['MSCI World','No Outlier MSCI World'])
# plt.title('MSCI World Boxplot Before and After The Outlier Detection')
# plt.plot(date,no_outlier_msciworld)
# plt.title('No Outlier MSCI World')

for x in range(len(no_outlier_msciemerging)):
    if (no_outlier_msciemerging[x] > 1288.2):
        no_outlier_msciemerging[x] = 1288.2
    if (no_outlier_msciemerging[x] < 750.65):
        no_outlier_msciemerging[x] = 750.65
# plt.boxplot([msciemerging,no_outlier_msciemerging],labels=['MSCI Emerging','No Outlier MSCI Emerging'])
# plt.title('MSCI Emerging Boxplot Before and After The Outlier Detection')
# plt.plot(date,no_outlier_msciemerging)
# plt.title('No Outlier MSCI Emerging')

for x in range(len(no_outlier_russel2kvgrowth)):
    if (no_outlier_russel2kvgrowth[x] > 1684.01):
        no_outlier_russel2kvgrowth[x] = 1684.01
# plt.boxplot([russel2kvgrowth,no_outlier_russel2kvgrowth],labels=['Russel 2000 Growth Index','No Outlier Russel 2000 Growth Index'])
# plt.title('Russel 2000 Growth Index Boxplot Before and After The Outlier Detection')
# plt.plot(date,no_outlier_msciworld)
# plt.title('No Outlier Russel 2000 Growth Index')

for x in range(len(no_outlier_mscius)):
    if (no_outlier_mscius[x] > 1684.01):
        no_outlier_mscius[x] = 1684.01
# plt.boxplot([msciusstock,no_outlier_mscius],labels=['MSCI US Stock','No Outlier MSCI US Stock'])
# plt.title('MSCI US Stock Boxplot Before and After The Outlier Detection')
# plt.plot(date,no_outlier_msciworld)
# plt.title('No Outlier Russel 2000 Growth Index')

##normalization##
scaler = MinMaxScaler()
no_outlier_snp = no_outlier_snp.values.reshape(-1, 1)
normalised_snp = scaler.fit_transform(no_outlier_snp)

no_outlier_exxon = no_outlier_exxon.values.reshape(-1, 1)
normalised_exxon = scaler.fit_transform(no_outlier_exxon)

no_outlier_chevron = no_outlier_chevron.values.reshape(-1, 1)
normalised_chevron = scaler.fit_transform(no_outlier_chevron)

no_outlier_riotinto = no_outlier_riotinto.values.reshape(-1, 1)
normalised_riotinto = scaler.fit_transform(no_outlier_riotinto)

no_outlier_bhp = no_outlier_bhp.values.reshape(-1, 1)
normalised_bhp = scaler.fit_transform(no_outlier_bhp)

no_outlier_glencore = no_outlier_glencore.values.reshape(-1, 1)
normalised_glencore = scaler.fit_transform(no_outlier_glencore)

no_outlier_brentoil = no_outlier_brentoil.values.reshape(-1, 1)
normalised_brentoil = scaler.fit_transform(no_outlier_brentoil)

no_outlier_coal = no_outlier_coal.values.reshape(-1, 1)
normalised_coal = scaler.fit_transform(no_outlier_coal)

no_outlier_ttfgas = no_outlier_ttfgas.values.reshape(-1, 1)
normalised_ttfgas = scaler.fit_transform(no_outlier_ttfgas)

no_outlier_nickel = no_outlier_nickel.values.reshape(-1, 1)
normalised_nickel = scaler.fit_transform(no_outlier_nickel)

no_outlier_copper = no_outlier_copper.values.reshape(-1, 1)
normalised_copper = scaler.fit_transform(no_outlier_copper)

no_outlier_msciworld = no_outlier_msciworld.values.reshape(-1, 1)
normalised_msciworld = scaler.fit_transform(no_outlier_msciworld)

no_outlier_msciemerging = no_outlier_msciemerging.values.reshape(-1, 1)
normalised_msciemerging = scaler.fit_transform(no_outlier_msciemerging)

no_outlier_russel2kv = no_outlier_russel2kv.values.reshape(-1, 1)
normalised_russel2kv = scaler.fit_transform(no_outlier_russel2kv)

no_outlier_russel2kvgrowth = no_outlier_russel2kvgrowth.values.reshape(-1, 1)
normalised_russel2kvgrowth = scaler.fit_transform(no_outlier_russel2kvgrowth)

no_outlier_mscius = no_outlier_mscius.values.reshape(-1, 1)
normalised_mscius = scaler.fit_transform(no_outlier_mscius)

no_outlier_ftseemerging = no_outlier_ftseemerging.values.reshape(-1, 1)
normalised_ftseemerging = scaler.fit_transform(no_outlier_ftseemerging)

no_outlier_ftsedeveloped = no_outlier_ftsedeveloped.values.reshape(-1, 1)
normalised_ftsedeveloped = scaler.fit_transform(no_outlier_ftsedeveloped)

no_outlier_ftseglobal = no_outlier_ftseglobal.values.reshape(-1, 1)
normalised_ftseglobal = scaler.fit_transform(no_outlier_ftseglobal)

no_outlier_amazon = no_outlier_amazon.values.reshape(-1,1)
normalised_amazon = scaler.fit_transform(no_outlier_amazon)

no_outlier_microsoft = no_outlier_microsoft.values.reshape(-1,1)
normalised_microsoft = scaler.fit_transform(no_outlier_microsoft)

date = date.values.reshape(-1,1)
date = date.flatten()
normalised_snp =normalised_snp.flatten()
normalised_exxon = normalised_exxon.flatten()
normalised_chevron = normalised_chevron.flatten()
normalised_riotinto = normalised_riotinto.flatten()
normalised_bhp = normalised_bhp.flatten()
normalised_glencore = normalised_glencore.flatten()
normalised_brentoil = normalised_brentoil.flatten()
normalised_coal = normalised_coal.flatten()
normalised_ttfgas = normalised_ttfgas.flatten()
normalised_nickel = normalised_nickel.flatten()
normalised_copper = normalised_copper.flatten()
normalised_msciworld = normalised_msciworld.flatten()
normalised_msciemerging = normalised_msciemerging.flatten()
normalised_russel2kv = normalised_russel2kv.flatten()
normalised_russel2kvgrowth = normalised_russel2kvgrowth.flatten()
normalised_mscius = normalised_mscius.flatten()
normalised_ftseemerging = normalised_ftseemerging.flatten()
normalised_ftsedeveloped = normalised_ftsedeveloped.flatten()
normalised_ftseglobal = normalised_ftseglobal.flatten()
normalised_amazon = normalised_amazon.flatten()
normalised_microsoft = normalised_microsoft.flatten()

no_outlier_snp =no_outlier_snp.flatten()
no_outlier_exxon = no_outlier_exxon.flatten()
no_outlier_chevron = no_outlier_chevron.flatten()
no_outlier_riotinto = no_outlier_riotinto.flatten()
no_outlier_bhp = no_outlier_bhp.flatten()
no_outlier_glencore = no_outlier_glencore.flatten()
no_outlier_brentoil = no_outlier_brentoil.flatten()
no_outlier_coal = no_outlier_coal.flatten()
no_outlier_ttfgas = no_outlier_ttfgas.flatten()
no_outlier_nickel = no_outlier_nickel.flatten()
no_outlier_copper = no_outlier_copper.flatten()
no_outlier_msciworld = no_outlier_msciworld.flatten()
no_outlier_msciemerging = no_outlier_msciemerging.flatten()
no_outlier_russel2kv = no_outlier_russel2kv.flatten()
no_outlier_russel2kvgrowth = no_outlier_russel2kvgrowth.flatten()
no_outlier_mscius = no_outlier_mscius.flatten()
no_outlier_ftseemerging = no_outlier_ftseemerging.flatten()
no_outlier_ftsedeveloped = no_outlier_ftsedeveloped.flatten()
no_outlier_ftseglobal = no_outlier_ftseglobal.flatten()
no_outlier_amazon = no_outlier_amazon.flatten()
no_outlier_microsoft = no_outlier_microsoft.flatten()

alldata_notnormalised = pd.DataFrame({'S&P 500 Price': no_outlier_snp,
                                     'Exxon': no_outlier_exxon,
                                     'Chevron': no_outlier_chevron,
                                     'Rio Tinto': no_outlier_riotinto,
                                     'BHP': no_outlier_bhp,
                                     'Glencore': no_outlier_glencore,
                                     'Brent Oil': no_outlier_brentoil,
                                     'Coal': no_outlier_coal,
                                     'TTF Gas': no_outlier_ttfgas,
                                     'Nickel': no_outlier_nickel,
                                     'Copper': no_outlier_copper,
                                     'MSCI World': no_outlier_msciworld,
                                     'MSCI Emerging': no_outlier_msciemerging,
                                     'Russel 2000 V': no_outlier_russel2kv,
                                     'Russel 2000 V Growth': no_outlier_russel2kvgrowth,
                                     'MSCI US': no_outlier_mscius,
                                     'FTSE Emerging': no_outlier_ftseemerging,
                                     'FTSE Developed': no_outlier_ftsedeveloped,
                                     'FTSE Global': no_outlier_ftseglobal,
                                     'Amazon' : no_outlier_amazon,
                                     'Microsoft': no_outlier_microsoft})


alldata = pd.DataFrame({'S&P 500 Price': normalised_snp,
                        'Exxon': normalised_exxon,
                        'Chevron': normalised_chevron,
                        'Rio Tinto': normalised_riotinto,
                        'BHP': normalised_bhp,
                        'Glencore': normalised_glencore,
                        'Brent Oil': normalised_brentoil,
                        'Coal': normalised_coal,
                        'TTF Gas': normalised_ttfgas,
                        'Nickel': normalised_nickel,
                        'Copper': normalised_copper,
                        'MSCI World': normalised_msciworld,
                        'MSCI Emerging': normalised_msciemerging,
                        'Russel 2000 V': normalised_russel2kv,
                        'Russel 2000 V Growth': normalised_russel2kvgrowth,
                        'MSCI US': normalised_mscius,
                        'FTSE Emerging': normalised_ftseemerging,
                        'FTSE Developed': normalised_ftsedeveloped,
                        'FTSE Global': normalised_ftseglobal,
                        'Amazon' : normalised_amazon,
                        'Microsoft': normalised_microsoft})


for column in alldata.columns:
        zero_indices = alldata[alldata[column] == 0].index
        alldata.loc[zero_indices, column] += 0.000001


# pd.describe_option('display')
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 10)
# pd.set_option('display.max_colwidth', None)
# print(alldata)


#########Correaltion Matrix and Variance############
correlation_matrix = alldata.corr()
# variance_alldata = alldata.var()
# print(variance_alldata)
# print(correlation_matrix)
# sns.heatmap(data=correlation_matrix,annot=True, xticklabels=True, yticklabels=True)

# plt.plot(date,normalised_snp)
# plt.xlabel('Date')
# plt.ylabel('S&P 500')
# plt.title('Normalised S&P 500')

# plt.plot(date,normalised_exxon,marker='o', markersize=1.7, color="green", alpha=0.5, label='Normalised Exxon')
# plt.plot(date,normalised_chevron,marker='o', markersize=1.7, color="red", alpha=0.5, label='Normalised Chevron')
# plt.plot(date,normalised_bhp,marker='o', markersize=1.7, color="blue", alpha=0.5, label='Normalised BHP')
# plt.legend()
# plt.title('Normalised Prices')

# plt.plot(date,normalised_riotinto,marker='o', markersize=1.7, color="green", alpha=0.5, label='Normalised Rio Tinto')
# plt.plot(date,normalised_glencore,marker='o', markersize=1.7, color="red", alpha=0.5, label='Normalised Glencore')
# plt.legend()
# plt.title('Normalised Prices 2')

# plt.plot(date,normalised_copper,marker='o', markersize=1.7, color="green", alpha=0.5, label='Normalised Copper Price')
# plt.plot(date,normalised_nickel,marker='o', markersize=1.7, color="red", alpha=0.5, label='Normalised Nickel Price')
# plt.plot(date,normalised_brentoil,marker='o', markersize=1.7, color="blue", alpha=0.5, label='Normalised Brent Oil')
# plt.plot(date,normalised_ttfgas,marker='o', markersize=1.7, color="yellow", alpha=0.5, label='Normalised TTF Gas')
# plt.plot(date,normalised_coal,marker='o', markersize=1.7, color="purple", alpha=0.5, label='Normalised Coal')
# plt.title('Normalised Commodities')
# plt.legend()

# plt.plot(date,normalised_ttfgas,marker='o', markersize=1.7, color="green", alpha=0.5, label='Normalised TTF Gas')
# plt.plot(date,normalised_coal,marker='o', markersize=1.7, color="red", alpha=0.5, label='Normalised Coal')
# plt.title('Normalised Commodities 2')
# plt.legend()

# alldata.to_excel(r'C:\Users\USER\Desktop\export_dataframe.xlsx', index=False
################MEAN VARIANCE OPTIMIZATION###################


# mu = expected_returns.mean_historical_return(alldata_notnormalised)
# S=risk_models.sample_cov(alldata_notnormalised)
# cov_matrix  = CovarianceShrinkage(alldata_notnormalised).ledoit_wolf()
#
# ef = EfficientFrontier(mu,S,weight_bounds=(None,None))
# ef.add_constraint(lambda w: w[0]+w[1]+w[2]+w[3]+w[4]+w[5]+w[6]+w[7]+w[8]+w[9]+w[10]+w[11]+w[12]+w[13]+w[14]+w[15]+w[16]+w[17]+w[18]+w[19]+w[20]==1)
# weights = ef.max_sharpe()
# cl_weisghts = ef.clean_weights()
# print(cl_weisghts)
# ef.portfolio_performance(verbose=True)
#
# lp = get_latest_prices(alldata_notnormalised)
#
# DA = DiscreteAllocation(weights, lp, total_portfolio_value=1000000)
# allocation, leftover = DA.greedy_portfolio()
# print("Discrete allocation:", allocation)
# print("Funds remaining: ${:.2f}".format(leftover))
#######################EF Plot###############
########FIRST PLOT##############
# mu2 = expected_returns.mean_historical_return(alldata_notnormalised)
# S2=risk_models.sample_cov(alldata_notnormalised)
# cov_matrix  = CovarianceShrinkage(alldata_notnormalised).ledoit_wolf()
#
# ef = EfficientFrontier(mu2,S2,weight_bounds=(None,None))
# ef.add_constraint(lambda w: w[0]+w[1]+w[2]+w[3]+w[4]+w[5]+w[6]+w[7]+w[8]+w[9]+w[10]+w[11]+w[12]+w[13]+w[14]+w[15]+w[16]+w[17]+w[18]+w[19]+w[20]==1)
# plotting.plot_efficient_frontier(ef)
#########SECOND PLOT WITH 1000 PORTFOLIOS WITH RISK RAGE 0.1 AND 0.8
# mu2 = expected_returns.mean_historical_return(alldata_notnormalised)
# S2=risk_models.sample_cov(alldata_notnormalised)
# cov_matrix  = CovarianceShrinkage(alldata_notnormalised).ledoit_wolf()
#
# ef = EfficientFrontier(mu2,S2,weight_bounds=(None,None))
# ef.add_constraint(lambda w: w[0]+w[1]+w[2]+w[3]+w[4]+w[5]+w[6]+w[7]+w[8]+w[9]+w[10]+w[11]+w[12]+w[13]+w[14]+w[15]+w[16]+w[17]+w[18]+w[19]+w[20]==1)
# risk_range = np.linspace(0.1,0.8,1000)
# plotting.plot_efficient_frontier(ef,ef_param="risk",ef_param_range=risk_range,show_assets=True,showfig=True)
# ef.portfolio_performance(verbose=True)
# #########PLOT ALL POTFOLIOS#############
# mu2 = expected_returns.mean_historical_return(alldata_notnormalised)
# S2=risk_models.sample_cov(alldata_notnormalised)
# cov_matrix  = CovarianceShrinkage(alldata_notnormalised).ledoit_wolf()
#
# ef = EfficientFrontier(mu2,S2,weight_bounds=(None,None))
# ef.add_constraint(lambda w: w[0]+w[1]+w[2]+w[3]+w[4]+w[5]+w[6]+w[7]+w[8]+w[9]+w[10]+w[11]+w[12]+w[13]+w[14]+w[15]+w[16]+w[17]+w[18]+w[19]+w[20]==1)
#
# fig, axEF = plt.subplots()
# plotting.plot_efficient_frontier(ef,ax=axEF,show_assets=False)
# res, std_tan , ret_tan = ef.portfolio_performance(verbose=True)
# axEF.scatter(std_tan,ret_tan,marker='*',s=100,c='r',label='Max Sharpe')
#
# n=10000
# w=np.random.dirichlet(np.ones(len(mu2)),n)
# ret = w.dot(mu2)
# std = np.sqrt(np.diag(w @ S2 @ w.T))
# sharpe = ret/std
# axEF.scatter(std, ret, marker='.',c= sharpe, cmap='viridis_r')
# axEF.set_title('Efficient Frontier With 10000 Random Portfolios')
# axEF.legend()
# plt.tight_layout()

#############With formulas ###############
# del df["Date"]
# alldata_mean = alldata.mean()
# covmatrix = alldata.cov()
# weights = np.array([0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05])
# sum = np.sum(alldata_mean*weights)*252
# std = np.sqrt(np.dot(weights.T,np.dot(covmatrix,weights))) * np.sqrt(252)
# print(round(sum*100,2),round(std*100,2))
#############Hierarchical Risk Parity############
# returns = alldata_notnormalised.pct_change().dropna()
# hrp = HRPOpt(returns)
# hrp_weights = hrp.optimize()
# hrp.portfolio_performance(verbose=True)
# cl_hrp_weights = hrp.clean_weights()
# print(dict(cl_hrp_weights))
# lp= get_latest_prices(alldata_notnormalised)
# da_hrp = DiscreteAllocation(hrp_weights, lp, total_portfolio_value=1000000)
#
# allocation, leftover = da_hrp.greedy_portfolio()
# print("Discrete allocation (HRP):", allocation)
# print("Funds remaining (HRP): ${:.2f}".format(leftover))
#####normalised DAta######
# returns = alldata_notnormalised.pct_change().dropna()
# hrp = HRPOpt(returns)
# hrp_weights = hrp.optimize()
# hrp.portfolio_performance(verbose=True)
# cl_hrp_weights = hrp.clean_weights()
# print('Weights = ', dict(cl_hrp_weights))
# lp= get_latest_prices(alldata_notnormalised)
# da_hrp = DiscreteAllocation(hrp_weights, lp, total_portfolio_value=100000)
#
# allocation, leftover = da_hrp.greedy_portfolio()
# print("Discrete allocation (HRP):", allocation)
# print("Funds remaining (HRP): ${:.2f}".format(leftover))
#################Mean Conditional Value at Risk#########################
# mu = mean_historical_return(alldata)
# S = alldata.cov()
# ef_cvar = EfficientCVaR(mu, S)
# cvar_weights = ef_cvar.min_cvar()
#
# cleaned_weights = ef_cvar.clean_weights()
# print(dict(cleaned_weights))
# lp= get_latest_prices(alldata)
# da_cvar = DiscreteAllocation(cvar_weights, lp, total_portfolio_value=100000)
# allocation, leftover = da_cvar.greedy_portfolio()
# print("Discrete allocation (CVAR):", allocation)
# print("Funds remaining (CVAR): ${:.2f}".format(leftover))

pct = alldata_notnormalised.pct_change()
mean_pct = pct.mean()
covMatrix_pct = pct.cov()
weights = np.random.random(len(mean_pct))
weights /= np.sum(weights)
print(weights)

sims = 100
day = 252
mean =np.full(shape=(day,len(weights)),fill_value=mean_pct)
mean = mean.T
port_sims =  np.full(shape=(day,sims),fill_value=0)
first_portfolio = 100000
sum = 0
for s in range(0,sims):
    Z = np.random.normal(size=(day, len(weights)))
    L = np.linalg.cholesky(covMatrix_pct)
    daily_returns = mean + np.inner(L,Z)
    port_sims[:,s] =np.cumprod(np.inner(weights,daily_returns.T)+1)*first_portfolio

for s in range(0,sims):
    sum += port_sims[251][s]
print('Average Portfolio Value at 252th Day', sum/sims)
print('Net Return After 252 Days on This Portfolio ', round((sum/sims)-first_portfolio,2))
# plt.figure(figsize=(16,8))
# plt.plot(port_sims)
# plt.ylabel('Portfolio Value (USD)')
# plt.xlabel('Days')
# plt.title('Monte Carlo Simulation of Portfolio')
r= 0.01
SharpeR = (port_sims.mean()-r)/port_sims.std()
print(SharpeR)




#################Hierarchical Risk Parity######################
# returns = alldata_notnormalised.pct_change().dropna()
# hrp = HRPOpt(returns)
# hrp_weights = hrp.optimize()
# hrp.portfolio_performance(verbose=True)
# print(dict(hrp_weights))
#
# lp= get_latest_prices(alldata)
# da_hrp = DiscreteAllocation(hrp_weights, lp, total_portfolio_value=100000)
# allocation, leftover = da_hrp.greedy_portfolio()
# print("Discrete allocation (HRP):", allocation)
# print("Funds remaining (HRP): ${:.2f}".format(leftover))

# stock_returns = alldata_notnormalised.pct_change()
#
# weights = {'S&P 500 Price':0.05,'Exxon':0.05,'Chevron': 0.05,'Rio Tinto' :0.05,'BHP': 0.05,'Glencore':0.05,'Brent Oil':0.05,'Coal':0.05,'TTF Gas':0.05,'Nickel':0.05,'Copper':0.05,'MSCI World':0.05,'MSCI Emerging': 0.05,'Russel 2000 V':0.05,'Russel 2000 V Growth':0.05,'MSCI US':0.05,'FTSE Emerging':0.05,'FTSE Developed':0.05,'FTSE Global':0.05,'Amazon' :0.05,'Microsoft' :0.05}
# #
# port_return = ([weights.get(x) for x in stock_returns.columns]*stock_returns).sum(axis=1)
# port_return = pd.DataFrame(port_return)
# port_return.rename(columns={port_return.columns[0]: 'Main Portfolio'}, inplace=True)
# port_return_var = port_return['Main Portfolio']
# print(port_return.describe())

# comp_data = yf.download("SPY ARKK XLE XLF GLD QQQ",start='2011-05-19',end='2023-03-13')
# comp_data_returns = comp_data['Adj Close'].pct_change()[1:]
# spy = comp_data_returns['SPY']
# arkk = comp_data_returns['ARKK']
# xle = comp_data_returns['XLE']
# xlf = comp_data_returns['XLF']
# gld = comp_data_returns['GLD']
# qqq = comp_data_returns['QQQ']
# calc_dataframe= pd.DataFrame({'SPY':spy,
#                          'ARKK':arkk,
#                          'XLE':xle,
#                          'XLF':xlf,
#                          'GLD':gld,
#                          'QQQ':qqq,
#                          'Main Portfolio':port_return_var})
# print(comp_data_returns.describe())

# print(calc_dataframe)
# alldata_notnormalised = pd.DataFrame({'Date':date,
#                                      'S&P 500 Price': no_outlier_snp,
#                                      'Exxon': no_outlier_exxon,
#                                      'Chevron': no_outlier_chevron,
#                                      'Rio Tinto': no_outlier_riotinto,
#                                      'BHP': no_outlier_bhp,
#                                      'Glencore': no_outlier_glencore,
#                                      'Brent Oil': no_outlier_brentoil,
#                                      'Coal': no_outlier_coal,
#                                      'TTF Gas': no_outlier_ttfgas,
#                                      'Nickel': no_outlier_nickel,
#                                      'Copper': no_outlier_copper,
#                                      'MSCI World': no_outlier_msciworld,
#                                      'MSCI Emerging': no_outlier_msciemerging,
#                                      'Russel 2000 V': no_outlier_russel2kv,
#                                      'Russel 2000 V Growth': no_outlier_russel2kvgrowth,
#                                      'MSCI US': no_outlier_mscius,
#                                      'FTSE Emerging': no_outlier_ftseemerging,
#                                      'FTSE Developed': no_outlier_ftsedeveloped,
#                                      'FTSE Global': no_outlier_ftseglobal,
#                                      'Amazon' : no_outlier_amazon,
#                                      'Microsoft': no_outlier_microsoft,})
#
# alldata_notnormalised = alldata_notnormalised.set_index('Date')
#
# returns = alldata_notnormalised.pct_change()
# mean_returns = returns.mean()
# covMatrix_returns = returns.cov()
# weights = np.random.random(len(mean_returns))
# weights /= np.sum(weights)
#
# portfolio_return = returns.dot(weights)
# var_matrix = returns.cov()*252
#
# portfolio_var = np.transpose(weights)@var_matrix@weights
#
# portfolio_vol = np.sqrt(portfolio_var)
# print('Portfolio Variance: ',portfolio_var)
# print('Portfolio Volatility: ',portfolio_vol)
#
# portfolio_returns = []
# portfolio_volatilies = []
# portfolio_weights = []
# asset_nums =len(alldata_notnormalised.columns)
# num_of_portfolios = 1000
# indv_rets = alldata_notnormalised.resample('Y').last().pct_change().mean()
#
# for portfolio in range(num_of_portfolios):
#     var = var_matrix.mul(weights,axis=0).mul(weights,axis=1).sum().sum()
#     std = np.sqrt(var)
#     vol = std*np.sqrt(252)
#     portfolio_volatilies.append(vol)
#     weights = np.random.random(asset_nums)
#     weights = weights/np.sum(weights)
#     portfolio_weights.append(weights)
#     returns = np.dot(weights,indv_rets)
#     portfolio_returns.append(returns)
#
#
# output = pd.DataFrame({'Returns':portfolio_returns,
#                        'Volatility':portfolio_volatilies})
#
# output.plot.scatter(x='Volatility', y='Returns', marker='o',color='red',s=30, alpha=0.6)
# plt.xlabel('Volatility')
# plt.ylabel('Returns')
# plt. title('Return - Volatility Of Modern Portfolio Theory')
#
# output_volatility = output['Volatility']
# output_returns = output['Returns']
# r =0.05
#
# min_vol=output.iloc[output['Volatility'].idxmin()]
# print((min_vol))
#
# best_ShrapeR = output.iloc[((output_returns-r)/output_volatility).idxmax()]
# print(best_ShrapeR)
#
# plt.subplots()
# plt.scatter(output_volatility,output_returns,alpha=0.6,s=30,color='red')
# plt.scatter(min_vol[1],min_vol[0],color='y',marker='*',s=300,label='Min Volatility')
# plt.scatter(best_ShrapeR[1],best_ShrapeR[0],color='b',marker='*',s=300,label='Best Sharpe Ratio')
# plt.xlabel('Volatility')
# plt.ylabel('Returns')
# plt.legend()
# plt.title('Return - Volatility Of Modern Portfolio Theory With Max Sharpe Ratio and Minimum Volatility')

plt.show()

