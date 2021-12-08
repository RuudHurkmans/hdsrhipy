# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 17:08:51 2021

@author: hurkmans
"""
import os
import sys
package_path = os.path.abspath('../../')
sys.path.append(package_path)  
import imod
from hdsrhipy import Runfile
from hdsrhipy.model import rasters
import hdsrhipy
import pandas as pd
import geopandas as gpd
from hdsrhipy.model import util
import shutil
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as td
from tqdm import tqdm


import geopandas as gpd
resolution = 25.         
hydromedah_path = r'E:\Hydromedah'      
msw_vars = ['ETact','S01','Ssd01']
start_date='2010-01-01'
end_date='2010-12-31'
rf = Runfile(os.path.join(hydromedah_path,'hdsr_ns.run'), data_path='$DBASE$\\', evap_path=hydromedah_path, precip_path=hydromedah_path)
rf.to_version(4.3)
rf.update_metaswap(datadir=hydromedah_path, start_date=pd.to_datetime(start_date), end_date=pd.to_datetime(end_date), metaswap_vars=msw_vars)
# make the calculation grid smaller (from [105000,173000,433000,473000])
rf.data['XMIN'] = 105000.0
rf.data['XMAX'] = 173000.0
rf.data['YMIN'] = 433000.0
rf.data['YMAX'] = 473000.0       
rf.data['CSIZE'] = resolution    
rf.change_period(start_date, end_date)
gdf_pg = gpd.read_file(os.path.join(hydromedah_path, 'Afwateringseenheden.shp'))
gdf_pg['CODENR'] = gdf_pg.index + 1                  
util.add_simgro_surface_water(rf, gdf_pg=gdf_pg, run1period=False, datadir=hydromedah_path)