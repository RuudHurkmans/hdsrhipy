# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 09:26:27 2021

@author: hurkmans

Script to merge groundwtaer observation database
"""
import imod
import numpy as np
import os
import pandas as pd
from datetime import datetime
from scipy import stats
import warnings
#from stats import getGXG
from pathlib import Path


# merge data
csv_path = r'D:\4569.10\validatiedata\groundwater'
file1 = Path(csv_path) / 'GrondwaterMeetpunten_reeksen.csv'
file1_meta = Path(csv_path) / 'GrondwaterMeetpunten_aanwezig.csv'

file2 = Path(csv_path) /  'data_freatisch_2021.csv'
file2_meta = Path(csv_path) /  'GrondwaterMeetpunten.csv'

ds1 = pd.read_csv(file1,sep=",")
ds1.index = [pd.Timestamp(ds1.iloc[i,0]) for i in range(len(ds1.iloc[:,0]))]
ds1.drop(ds1.columns[0], axis=1, inplace=True)

if file1_meta is not None:    
    try:
        md1 = pd.read_csv(file1_meta, sep=",")
        ds1.columns = [md1[md1['PutFilter']==ID]['Name'].to_string(index=False) for ID in list(md1['PutFilter'])]        
    except:
        pass
    
ds2 = pd.read_csv(file2,sep=",", skiprows=3)
ds2.index = [pd.Timestamp(ds2.iloc[i,0]) for i in range(len(ds2.iloc[:,0]))]
ds2.drop(ds2.columns[0], axis=1, inplace=True)

if file2_meta is not None:    
    md2 = pd.read_csv(file2_meta, sep=",")
          
mdf = pd.DataFrame(np.zeros( (ds1.shape[0]+ds2.shape[0], len(set(list(ds1.columns)+list(ds2.columns)))))*np.nan)
mdf.columns = set(list(ds1.columns)+list(ds2.columns))
mdf.index = list(ds1.index) + list(ds2.index)

mdm  = md1
md1['PutFilter']= md1['Name']

for col in ds1.columns:
    if col in ds2.columns:
        ts1 = ds1[col]
        ts2 = ds2[col]
        mdf[col] = ts1.append(ts2)
    else:
        mdf.at[ds1.index, col] = ts1
for col in ds2.columns:
    if col in ds1.columns:
        ts1 = ds1[col]
        ts2 = ds2[col]
        mdf[col] = ts1.append(ts2)
    else:
        mdf.at[ds2.index, col] = ts2          
    row = md2[md2['LOC_ID']==col]
    newrow = [0.0, col,col,1.0,'2021-01-01','2021-07-31',float(row['X']), float(row['Y']), float(row['BKFILTNAP']), float(row['OKFILTNAP']),'',float(row['GHG']), float(row['GLG']), np.nan, np.nan]
    newrowdf = pd.DataFrame(newrow).T
    newrowdf.columns = mdm.columns
    mdm = mdm.append(newrowdf)        

mdm.index = range(0,mdm.shape[0])
mdf.to_csv(Path(csv_path) / 'alle_reeksen.csv', sep=",")
mdm.to_csv(Path(csv_path) / 'alle_aanwezig.csv', sep=",")
