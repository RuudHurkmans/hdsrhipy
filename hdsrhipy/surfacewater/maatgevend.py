# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 10:05:03 2021

@author: hurkmans
"""
import sys
import os

package_path = os.path.abspath('../../')
sys.path.append(package_path) 
import pandas as pd
import geopandas as gpd
from hdsrhipy.model import util
import shutil
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as td
from tqdm.auto import tqdm
from pathlib import Path

import matplotlib.pyplot as plt

class Maatgevend: 
   
    def __init__(self, model_path=None, name=None, export_path=None, afw_shape=None):        
        
        if export_path is None:
            export_path = r'D:\4569.10\results'#%%       
       
        if afw_shape is None:
            afw_shape = r'D:\4569.10\python\hdsrhipy\resources\Afwateringseenheden.shp'
        self.export_path = Path(export_path) / 'Maatgevend'
        self.export_path.mkdir(parents=True, exist_ok=True)
        
        if name is None:
            name='Huidig250'
        self.name=name                     
            
        self.afw = gpd.read_file(afw_shape)
        self.afw['CODENR'] = self.afw.index + 1
        
    def get_laterals(self):                       
        self.laterals = pd.read_csv(os.path.join(self.export_path,'..','Laterals_'+self.name+'.csv'), sep=",")
        #self.laterals.columns = 
        #%%

    def plot_location(self, nr):
        times = [pd.Timestamp(self.laterals.iloc[i,0]) for i in range(self.laterals.shape[0])]
        years = set([pd.Timestamp(self.laterals.iloc[i,0]).year for i in range(self.laterals.shape[0])])
        afwid = self.afw.loc[self.afw.CODENR==int(nr),'CODE'].to_string(index=False)
        print('Plotting '+afwid)
        ts = self.laterals.loc[:,str(nr)]              
        ts_af = ts.copy(deep=True)
        ts_af[ts_af < 0.] = 0.               
        rts1 = ts_af.rolling(24, closed='both').mean()
        rts1[np.isnan(rts1)] = 0.
        ts_aan = ts.copy(deep=True)
        ts_aan[ts_aan > 0. ] = 0
        ts_aan = ts_aan * -1.        
        rts10 = ts_aan.rolling(240, closed='both').mean()           
        rts10[np.isnan(rts10)] = 0.         
        fig, axs = plt.subplots(3)
        fig.suptitle('Maatgevende aan/afvoer voor '+afwid)
        fig.set_size_inches(8,8)
        axs[0].plot(times, ts, color='blue',label='Aan/afvoer')        
        axs[0].set_ylabel('Afvoer [m3/s]')
        axs[0].legend(ncol=2)
        mqaf = rts1.sort_values(ascending=True).iloc[int(-1.5*len(years))]        
        axs[1].plot(range(len(times)), rts1.sort_values(ascending=True),label='Afvoer gesorteerd')
        axs[1].plot(len(times)-int(1.5*len(years)), mqaf, 'bo', label='Maatgevende afvoer [m3/s]')
        axs[1].legend(ncol=2)
        axs[1].set_ylabel('Afvoer [m3/s]')        
        axs[2].plot(range(len(times)), rts10.sort_values(ascending=True),color='red',label='Aanvoer gesorteerd')
        axs[2].plot(len(times)-int(0.1*len(years)), rts10.sort_values(ascending=True).iloc[int(-0.1*len(years))], 'o',color='red', label='Maatgevende aanvoer [m3/s]')
        axs[2].set_ylabel('Aanvoer [m3/s]')
        axs[2].legend(ncol=2)
        plt.savefig(os.path.join(self.export_path, 'MG_'+afwid+'_'+self.name+'.png'))
                   

    def get_q_norm(self, dataset=None):
        if dataset is None:
            dataset = self.laterals
        times = [pd.Timestamp(dataset.iloc[i,0]) for i in range(dataset.shape[0])]
        years = set([pd.Timestamp(dataset.iloc[i,0]).year for i in range(dataset.shape[0])])
        mg_q = self.afw.copy(deep=True)
        keep = ['CODE','geometry','CODENR']
        for k in self.afw.columns:
            if k not in keep:
                mg_q.drop([k],axis=1,inplace=True) 
        mg_q['MQAF_M3S'] = np.nan
        mg_q['MQAF_MMD'] = np.nan
        mg_q['MQAAN_M3S'] = np.nan
        mg_q['MQAAN_MMD'] = np.nan
        mg_q.index = self.afw.index
        #%%
        for ind,i in tqdm(self.afw.iterrows(), total=len(self.afw)):
            if str(i.CODENR) in dataset.columns.to_list():        
                ts = dataset.loc[:,str(i.CODENR)]                      
                ts_af = ts.copy(deep=True)
                ts_af[ts_af < 0.] = 0.                
                # maatgevende afvoer                
                rts1 = ts_af.rolling(24, closed='both').mean()
                mqaf = rts1.sort_values(ascending=True).iloc[int(-1.5*len(years))]         
                mg_q.loc[mg_q['CODENR']==i.CODENR, 'MQAF_M3S'] = mqaf
                mg_q.loc[mg_q['CODENR']==i.CODENR, 'MQAF_MMD'] = (mqaf / i.geometry.area)*1e3*86400.
                                
                ts_aan = ts.copy(deep=True)
                ts_aan[ts_aan > 0. ] = 0
                ts_aan = ts_aan * -1.
                rts10 = ts_aan.rolling(240, closed='both').mean()
                ts_aan[np.isnan(ts_aan)] = 0.
                mqaan = ts_aan.sort_values(ascending=True).iloc[int(-0.1*len(years))]   
                mg_q.loc[mg_q['CODENR']==i.CODENR, 'MQAAN_M3S'] = mqaan
                mg_q.loc[mg_q['CODENR']==i.CODENR, 'MQAAN_MMD'] = (mqaan / i.geometry.area)*1e3*86400.
        return mg_q
        
        
        
# mg = Maatgevend()
# mg.get_laterals()
# qnorm = mg.get_q_norm()

# lat_samples = sample_nauwkeurigheid(mg.laterals, 0.01, n=2)

# (mins, maxs)  = MonteCarlo('normative', lat_samples, bootstrap_n=2, n=2)

# mg.plot_location(1828)

# mg.normative.to_file(os.path.join(mg.export_path, 'maatgevend.shp'))        

