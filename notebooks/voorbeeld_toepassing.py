import os
import sys
import imod
import numpy as np
import pandas as pd
import geopandas as gpd
import shutil
from shapely.geometry import Point
package_path = os.path.abspath('../')
sys.path.append(package_path)  
from pathlib import Path
import rasterio
import rioxarray as rio

from hdsrhipy import Meteorology
from hdsrhipy import Hydromedah
from hdsrhipy import Groundwater
from hdsrhipy import Maatgevend
from hdsrhipy import Runoff

from hdsrhipy.core.nauwkeurigheid import *

meteo = Meteorology()

#%%
# Download climate scenario from the dataset from the KNMI data platform
# meteo_path = os.path.join(r'E:\scenarios\\')
# datasets = ['RD85WL', 'EV85WL','RD85WH','EV85WH','RD85GH','EV85GH']
# for dat in [datasets[5]]:
#     if dat.startswith('RD'):
#         var = 'precipitation'
#     elif dat.startswith('EV'):
#         var = 'evaporation'
#     for y in range(2070,2100):        
#         print('Downloading '+dat+'for '+str(y)+'...')
#         start = 'INTER_OPER_R___'+dat+'__L3__'+str(y)+'0101T080000_'+str(y)+'0102T080000_0014.nc'
#         meteo.download_from_KNMI(variable=var, dataset_name=dat, dataset_version='014', max_keys=366, start_after=start, download_path = Path(meteo_path,dat))



export_path = r'D:\4569.10\results'
#%%

# Download HDSR forcing (precip, Makkink refernce evap) from WIWB
credentials = ('hkv.lijninwater', 'mVxIbDqmsWpK40P2WafY')   
hydromedah_path = os.path.join(r'E:\Hydromedah')

# 2010 2020
#meteo.download_from_WIWB(credentials=credentials, datasource='Meteobase.Precipitation', variable='precipitation', start='20100101000000', end='20190101000000', timestep='1D', download_path = Path(data_path))
#meteo.download_from_WIWB(credentials=credentials, datasource='Meteobase.Evaporation.Makkink', variable='evaporation', start='20100101000000', end='20210101000000', timestep='1D', download_path = Path(hydromedah_path))
#meteo.download_from_WIWB(credentials=credentials, datasource='Knmi.International.Radar.Composite.Final.Reanalysis', variable='precipitation', start='20210101000000', end='20211001000000', timestep='1D', download_path = Path(hydromedah_path))


#%%


#%%
export_path = r'D:\4569.10\results'

#peil_hagbov = [3.25, 5.14, 6.30] # # gemiddeld 3.65 T1, T10, T100
#Berekening voor berekend/opgegeven niveau  3.25 m+NAP
#Berekening voor berekend/opgegeven niveau  5.14 m+NAP
#Berekening voor berekend/opgegeven niveau  6.30 m+NAP

# for i in range(len(offsets)):
offsets = [0.0]+[x-3.65 for x in [3.25, 5.14, 6.30]]
offsetnames = ['ref','T1','T10','T100']



# print('s')
#%%
# ds = rasterio.open(os.path.join(pad,'peil_laag1_1.tif'))             
# ds = ds.rio.set_crs(28992, inplace=True)       
# #pixels,_ = mask(dataset=dat.to_dataset(), shapes=[json.loads(owshp.to_json())['features'][0]['geometry']], crop=True)
# pixels,_ = mask(dataset=ds,shapes=[json.loads(owshp2.to_json())['features'][0]['geometry']], crop=True)
#%%


#% 2021
nc_path = Path(r'D:\4569.10\nonapi_forcering')
# meteo.download_from_WIWB(credentials=credentials, api_key=None, datasource='Knmi.International.Radar.Composite.Final.Reanalysis', variable='precipitation', start='20210101000000', end='20210101000000', timestep='1D', download_path = Path(hydromedah_path))
# meteo.download_from_WIWB(credentials=credentials, datasource='Meteobase.Evaporation.Makkink', variable='evaporation', start='20210101000000', end='20220101000000', timestep='1D', download_path = Path(hydromedah_path))
# meteo.from_netcdf(nc_path, hydromedah_path, 'evaporation', 'Days', dsvar='Evaporation', scaling_factor= 0.01)

#%%
##Geul (temporary)
#meteo.download_from_WIWB(credentials=credentials, datasource='Knmi.International.Radar.Composite.Final.Reanalysis', variable='precipitation', start='20210831000000', end='20210902000000', timestep='1H', extent=[10725,102725,450793,635793],download_path = Path(r'D:\3663.10\WS2021'))
#forecasts
#meteo.download_from_WIWB(credentials=credentials, datasource='Knmi.Harmonie.Evaluatie', variable='absolute_difference', start=f'20210101000000', end=f'20210102000000', timestep='6H', extent=[10725,102725,450793,635793],download_path = Path(r'D:\3663.10\eval'))
# #%%

msw_vars = ['ETact','S01','Ssd01', 'qinf']
# hh = 'REF'
# name = 'Huidig250'
# validatiepad = r'D:\4569.10\validatiedata'
# hydromedah = Hydromedah(data_path=hydromedah_path, name=name, precip_path=hydromedah_path,evap_path=hydromedah_path)
# hydromedah.setup_model(start_date='2010-01-01', end_date='2021-10-01', resolution=250., add_surface_water=True, afw_shape = 'Afwateringseenheden', metaswap_vars = msw_vars)

# #shutil.copyfile(os.path.join(validatiepad, 'PEIL_LAAG_1_'+hh+'.IDF'), os.path.join(hydromedah_path, 'work', name, 'OPPERVLAKTEWATER','WINTER','PEIL_LAAG1_1.IDF'))
# # name = 'WH85250'
# # hydromedah = Hydromedah(data_path=hydromedah_path, name=name, precip_path=r'E:\scenarios\\RD85WH',evap_path=r'E:\scenarios\\EV85WH')
# # hydromedah.setup_model(start_date='2080-01-01', end_date='2089-12-31', resolution=250., add_surface_water=True, afw_shape = 'Afwateringseenheden', metaswap_vars = msw_vars)
# hydromedah.run_model(model_path = hydromedah_path)
# laterals = hydromedah.read_laterals(model_path=hydromedah_path, model_name=name, msw_file='sw_dtgw')
# laterals.to_csv(os.path.join(export_path,'laterals_'+name+'.csv'),sep=",")
# hydromedah.cleanup(model_path=hydromedah_path, name=name, 
#                       modflow_vars = ['head','bdgflf'],
#                       modflow_layers = [1],
#                       metaswap_files = ['sw_dtgw'])              


   
#%%
# name = 'Huidig250_2'
# r = Runoff(name=name)
# sd = r.get_msw_var('msw_Ssd01')4
# sd_mean = r.get_season_stat(sd, stat='mean')
# sd_samples = sample_nauwkeurigheid(sd, [-0.1,0.1], n=2)
# (sd_avmin, sd_avmax)  = MonteCarlo('metaswap_mean', sd_samples, bootstrap_n=2, n=2)
# sd_min = r.get_season_stat(sd, stat='min')
# (sd_minmin, sd_minmax)  = MonteCarlo('metaswap_min', sd_samples, bootstrap_n=2, n=2)

# inf = r.get_msw_var('msw_qinf')
# inf_mean = r.get_season_stat(inf, stat="mean")
# inf_samples = sample_nauwkeurigheid(sd,[-0.1,0.1], n=2)
# (inf_avmin, inf_avmax)  = MonteCarlo('metaswap_mean', inf_samples, bootstrap_n=2, n=2)
# inf_min = r.get_season_stat(inf, stat="min")
# (inf_minmin, inf_minmax)  = MonteCarlo('metaswap_min', inf_samples, bootstrap_n=2, n=2)

# gis_path = os.path.join(os.path.abspath('.'), '..','hdsrhipy','resources') 
# shapefile = os.path.join(gis_path, 'bod_clusters.shp')
# be_shp = gpd.read_file(shapefile)
# be_shp.drop(be_shp[be_shp['BODEMCODE']=='|g WATER'].index, axis=0, inplace=True)
# av_shp = gpd.read_file(os.path.join(gis_path, 'afvoergebieden.shp'))
# be_shp = gpd.clip(be_shp,av_shp)
# # #%%
# outdf = be_shp.copy(deep=True)
# outdf.drop([col for col in outdf.columns if col not in ['OBJECTID','BODEMCODE', 'geometry']], axis=1, inplace=True)
# outdf = r.aggregate_to_shapefile(sd_mean[0], shapefile=be_shp, output_df=outdf, outcolid='bav_S')  
# outdf = r.aggregate_to_shapefile(sd_avmin[0], shapefile=be_shp, output_df=outdf, outcolid='bav_l_S')  
# outdf = r.aggregate_to_shapefile(sd_avmax[0], shapefile=be_shp, output_df=outdf, outcolid='bav_u_S')  
# outdf = r.aggregate_to_shapefile(sd_min[0], shapefile=be_shp, output_df=outdf, outcolid='bmn_S')  
# outdf = r.aggregate_to_shapefile(sd_minmin[0], shapefile=be_shp, output_df=outdf, outcolid='bmn_l_S')  
# outdf = r.aggregate_to_shapefile(sd_minmax[0], shapefile=be_shp, output_df=outdf, outcolid='bmn_l_S')  
# outdf = r.aggregate_to_shapefile(sd_mean[1], shapefile=be_shp, output_df=outdf, outcolid='bav_W')  
# outdf = r.aggregate_to_shapefile(sd_avmin[1], shapefile=be_shp, output_df=outdf, outcolid='bav_l_W')  
# outdf = r.aggregate_to_shapefile(sd_avmax[1], shapefile=be_shp, output_df=outdf, outcolid='bav_u_W')  
# outdf = r.aggregate_to_shapefile(sd_min[1], shapefile=be_shp, output_df=outdf, outcolid='bmn_W')  
# outdf = r.aggregate_to_shapefile(sd_minmin[1], shapefile=be_shp, output_df=outdf, outcolid='bmn_l_W')  
# outdf = r.aggregate_to_shapefile(sd_minmax[1], shapefile=be_shp, output_df=outdf, outcolid='bmn_l_W')  
# outdf.to_file(os.path.join(export_path, 'runoff_karakteristieken_rzstor.shp'))
# outdf = be_shp.copy(deep=True)
# outdf.drop([col for col in outdf.columns if col not in ['OBJECTID','BODEMCODE', 'geometry']], axis=1, inplace=True)
# outdf = r.aggregate_to_shapefile(inf_mean[0], shapefile=be_shp, output_df=outdf, outcolid='iav_S')  
# outdf = r.aggregate_to_shapefile(inf_avmin[0], shapefile=be_shp, output_df=outdf, outcolid='iav_l_S')  
# outdf = r.aggregate_to_shapefile(inf_avmax[0], shapefile=be_shp, output_df=outdf, outcolid='iav_u_S')  
# outdf = r.aggregate_to_shapefile(inf_min[0], shapefile=be_shp, output_df=outdf, outcolid='imn_S')  
# outdf = r.aggregate_to_shapefile(inf_minmin[0], shapefile=be_shp, output_df=outdf, outcolid='imn_l_S')  
# outdf = r.aggregate_to_shapefile(inf_minmax[0], shapefile=be_shp, output_df=outdf, outcolid='imn_l_S')  
# outdf = r.aggregate_to_shapefile(inf_mean[1], shapefile=be_shp, output_df=outdf, outcolid='iav_W')  
# outdf = r.aggregate_to_shapefile(inf_avmin[1], shapefile=be_shp, output_df=outdf, outcolid='iav_l_W')  
# outdf = r.aggregate_to_shapefile(inf_avmax[1], shapefile=be_shp, output_df=outdf, outcolid='iav_u_W')  
# outdf = r.aggregate_to_shapefile(inf_min[1], shapefile=be_shp, output_df=outdf, outcolid='imn_W')  
# outdf = r.aggregate_to_shapefile(inf_minmin[1], shapefile=be_shp, output_df=outdf, outcolid='imn_l_W')  
# outdf = r.aggregate_to_shapefile(inf_minmax[1], shapefile=be_shp, output_df=outdf, outcolid='imn_l_W') 
# outdf.to_file(os.path.join(export_path, 'runoff_karakteristieken_inf.shp'))

#%%
# gw= Groundwater()

# #gw.get_validation_data()
# csvfile = r'D:\4569.10\validatiedata\groundwater\results\with LHM\statistics_measured_modeled.csv'
# summary = pd.read_csv(csvfile,sep=",")
# summary.dropna(axis=0,inplace=True, subset=['GHG'])
# afg = 'all'
# afgid  = r'D:\4569.10\validatiedata\surfacewater\afvoergebieden.shp'
# gdf_afg = gpd.read_file(afgid)

# if afg == 'all':
#     geom = gdf_afg.unary_union
# else:    
#     geom = gdf_afg[gdf_afg['GAFNAAM'] ==afg].geometry

# gdf = gpd.GeoDataFrame(summary, geometry=gpd.points_from_xy(summary.X_RD_CRD, summary.Y_RD_CRD))
# subset = gdf[gdf.geometry.within(geom)]
# means = [np.mean([i['GHG_ERROR'], i['GLG_ERROR'],i['GVG_ERROR']]) for _,i in subset.iterrows()]
# head_bandwidth = [np.percentile(means,17), np.percentile(means, 83)]

# name = 'Huidig250'
# gw = Groundwater(model_path=hydromedah_path, export_path=None, name=name)

# gw.get_seepage()
# means = gw.seepage_season_means()

# sp_samples = sample_nauwkeurigheid(gw.seepage, [-0.01, 0.01], n=1)
# (mins, maxs)  = MonteCarlo('seepage', sp_samples, bootstrap_n=5, n=1)

# months = ['W','L','Z','N','S6','W6']
# for i in range(6):
#     gw.export_raster(means[i], 'SP_best_'+months[i]+'_'+name+'.tif')
#     gw.export_raster(mins[i], 'SP_min_'+months[i]+'_'+name+'.tif')
#     gw.export_raster(maxs[i], 'SP_max_'+months[i]+'_'+name+'.tif')
   
# del gw.seepage, mins, maxs, means

# gw.get_heads_raster()
# means = gw.get_gxg_raster()
# means['gt'] = gw.get_gt_raster(gxg=means)

# #print(np.nanmin(gw.heads), np.nanmean(gw.heads), np.nanmax(gw.heads))
# head_samples = sample_nauwkeurigheid(gw.heads, head_bandwidth, n=2)            
# (mins, maxs) = MonteCarlo('gxg', head_samples, bootstrap_n=2, n=2)

# items = ['GHG','GLG','GVG','GT']
# for i in range(4):
#     gw.export_raster(means[items[i].lower()], items[i]+'_best_'+name+'.tif')
#     gw.export_raster(mins[i], items[i]+'_min_'+name+'.tif')
#     gw.export_raster(maxs[i], items[i]+'_max_'+name+'.tif')
    
    
#%% get bandwidth of surface water
import pandas as pd
q_afg = r'D:\4569.10\validatiedata\surfacewater\debieten_afvoergebieden.csv'
data = pd.read_csv(q_afg)
data[data==-999] = np.nan
data.drop([col for col in data.columns if 'quality' in col], axis=1, inplace=True)
afvoer = data.iloc[:,[i for i in range(len(data.columns)) if data.iloc[0,i] == 'H.B.u.d']]
aanvoer = data.iloc[:, [i for i in range(len(data.columns)) if data.iloc[0,i] == 'H.B.i.d']]
afvoer.drop([0], axis=0, inplace=True)
aanvoer.drop([0], axis=0, inplace=True)
afvoer.index = [pd.Timestamp(data.iloc[i,0]) for i in range(len(data.iloc[1:,0]))]
aanvoer.index  = [pd.Timestamp(data.iloc[i,0]) for i in range(len(data.iloc[1:,0]))]


wis_subloc = pd.read_csv(r'D:\4569.10\hdsrhipy\hdsrhipy\resources\oppvlwater_subloc.csv')


# 
# # #%%
# # mg = Maatgevend() 
# # mg.get_laterals(seepage_subtracted=False)
# # qnorm = mg.get_q_norm(dataset = mg.laterals)

# # mg.subtract_seepage(mean_seepage = True , model_path = hydromedah_path)
# # mg.get_laterals(seepage_subtracted=True, mean_seepage=True)
# # qnorm_nosp = mg.get_q_norm(dataset = mg.laterals_nosp)

# # mg.export_shp(qnorm, 'Qm_HUidig2050.shp')
# # mg.export_shp(qnorm_nosp, 'Qm_HUidig2050_no_seepage.shp')
# # #%%
# # mg.plot_location(567, seepage_subtracted=True)
# # plotting = range(1,1400,250)
# # for i in plotting:
# #      try:
# #          mg.plot_location(i, seepage_subtracted=True)
# #      except:
# #          print('Figure failed.')




# # #%% 
# # pad = r'E:\Hydromedah\work\Huidig250\OPPERVLAKTEWATER\WINTER'
# # test11 = imod.idf.open(os.path.join(pad,'PEIL_LAAG1_1.IDF'))
# # test12 = imod.idf.open(os.path.join(pad,'PEIL_LAAG1_2.IDF'))
# # test13 = imod.idf.open(os.path.join(pad,'PEIL_LAAG1_3.IDF'))
# # test21 = imod.idf.open(os.path.join(pad,'PEIL_LAAG2_1.IDF'))
# # test22 = imod.idf.open(os.path.join(pad,'PEIL_LAAG2_2.IDF'))
# # test23 = imod.idf.open(os.path.join(pad,'PEIL_LAAG2_3.IDF'))
# # import matplotlib.pyplot as plt
# # # plt.imshow(test1)
# # # plt.imshow(test2)
# # # plt.imshow(test3)

# # #%%
# # #



# # # #%%
# # # qaf_gaf = []
# # # qaan_gaf = []
# # # for _,afg in gdf_afg.iterrows():
# # #     geom = afg.geometry
# # #     subset = qnorm[qnorm.geometry.within(geom)]
# # #     qaf_gaf.append(subset['MQAF_MMD'].dropna().mean())
# # #     qaan_gaf.append(subset['MQAAN_MMD'].dropna().mean())
    
# # # #%% 


# # #lat_samples = sample_nauwkeurigheid(mg.laterals, head_bandwidth, n=1)
# # # (mins, maxs)  = MonteCarlo('normative', lat_samples, bootstrap_n=2, n=2)

# # # validate
   
# # # #%%
# # # # # # 
# # # name = 'Huidig250'
# # # gw = Groundwater(model_path=hydromedah_path, export_path=None, name=name)

# # # gw.get_seepage()
# # # means = gw.seepage_season_means()

# # # sp_samples = sample_nauwkeurigheid(gw.seepage, 0.01, n=5)
# # # (mins, maxs)  = MonteCarlo('seepage', sp_samples, bootstrap_n=5, n=10)

# # # months = ['W','L','Z','N','S6','W6']
# # # for i in range(6):
# # #     gw.export_raster(means[i], 'SP_best_'+months[i]+'_'+name+'.tif')
# # #     gw.export_raster(mins[i], 'SP_min_'+months[i]+'_'+name+'.tif')
# # #     gw.export_raster(maxs[i], 'SP_max_'+months[i]+'_'+name+'.tif')

   
# # # gw.get_heads_raster()
# # # means = gw.get_gxg_raster()
# # # means['gt'] = gw.get_gt_raster(gxg=means)

# # # #print(np.nanmin(gw.heads), np.nanmean(gw.heads), np.nanmax(gw.heads))
# # # head_samples = sample_nauwkeurigheid(gw.heads, 0.05, n=5)            
# # # (mins, maxs) = MonteCarlo('gxg', head_samples, bootstrap_n=5, n=10)


# # # items = ['GHG','GLG','GVG','GT']
# # # for i in range(4):
# # #     gw.export_raster(means[items[i].lower()], items[i]+'_best_'+name+'.tif')
# # #     gw.export_raster(mins[i], items[i]+'_min_'+name+'.tif')
# # #     gw.export_raster(maxs[i], items[i]+'_max_'+name+'.tif')
    
    



