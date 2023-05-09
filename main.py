import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from sklearn.preprocessing import MinMaxScaler
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
fig, axmicrosoft = plt.subplots()
axmicrosoft.plot(date,microsoft)
axmicrosoft.set_title('Microsoft')
axmicrosoft.set_xlabel('Date')
axmicrosoft.set_ylabel('Microsoft')


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

alldata = pd.DataFrame({'Date': date,
                        'S&P 500 Price': normalised_snp,
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
                        'FTSE Global': normalised_ftseglobal})

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
#
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

plt.show()

