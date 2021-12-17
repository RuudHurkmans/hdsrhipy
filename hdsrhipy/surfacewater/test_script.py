# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 15:21:09 2021

@author: pezij
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import contextily as cx

from WatervraagAanbod import WatervraagAanbod

plt.close('all')

#%%
# definitie van een dictionary voor de bestandsnamen
fNames = dict()

# top10nl
fNames['top10nl'] = r'd:\projecten\hydrologische_informatieproducten\werkmap\data\top10nl\clip\top10nl_terrein_water_weg.gpkg'

# koppelcodes top10nl en Mozart
fNames['top10nl_koppeling'] = r'd:\projecten\hydrologische_informatieproducten\werkmap\data\top10nl\koppeling_top10nl_mozart.xlsx'

# LSW-polygonen
fNames['lsw'] = r'd:\projecten\hydrologische_informatieproducten\werkmap\data\geo\lsw\lsw_hdsr_large.shp'

# afwateringseenheden
fNames['afwateringseenheden'] = r'd:\projecten\hydrologische_informatieproducten\werkmap\data\GISdata\afwateringseenheden.shp'

# uitvoer mozart
fNames['mozart_out'] = r'd:\projecten\hydrologische_informatieproducten\werkmap\data\mozart\lswwaterbalans.out'

#%% inlezen data

# initieer watervraag- en aanbod class voor referentie en onzekerheidsanalyse
referentie = WatervraagAanbod(fNames)
ref_min5 = WatervraagAanbod(fNames)
ref_plus5 = WatervraagAanbod(fNames)

# bepaal interessegebieden
referentie.invoerdata['schaalgebied'] = referentie.invoerdata['afwateringseenheden']
ref_min5.invoerdata['schaalgebied'] = ref_min5.invoerdata['afwateringseenheden']
ref_plus5.invoerdata['schaalgebied'] = ref_plus5.invoerdata['afwateringseenheden']

# verkrijg de LSW-nummers
lswNrs = referentie.invoerdata['lsws']['LSWFINAL']

print("Uitvoeren schaling lsw's naar schaalgebieden")
for _, lsw_nr in tqdm(lswNrs[8:10].iteritems()):
    
    # voer referentieanalyse uit
    referentie.uitvoeren_schaling(lsw_nr, onzekerheid_opp=0)
    
    # definieer onzekerheid oppervlaktes top10nl
    ref_min5.uitvoeren_schaling(lsw_nr, onzekerheid_opp=-0.05)
    ref_plus5.uitvoeren_schaling(lsw_nr, onzekerheid_opp=0.05)
    
            
#%% schrijf weg naar csv
ref_dir = r'd:\projecten\hydrologische_informatieproducten\werkmap\resultaten\referentie\\'
min5_dir = r'd:\projecten\hydrologische_informatieproducten\werkmap\resultaten\min5\\'
plus5_dir = r'd:\projecten\hydrologische_informatieproducten\werkmap\resultaten\plus5\\'

# schrijf geschaalde mozartresultaten weg naar mappen
referentie.schaling_naar_csv(ref_dir)
ref_min5.schaling_naar_csv(min5_dir) 
ref_plus5.schaling_naar_csv(plus5_dir)

#%% overzichtskaart maken

periode_start = '2018-06-01'
periode_end = '2018-08-01'
gebruiker = 'landbouw'
aspect = 'watertekort'

referentie.invoerdata['schaalgebied'][f'{aspect}{gebruiker}'] = np.nan

for index, row in referentie.invoerdata['schaalgebied'].iterrows():
    key = row['CODE']
    
    try:
        df = pd.read_csv(fr'{ref_dir}\mozart_schaalgebied_{key}_geschaald.csv',
                        parse_dates=True,
                        index_col=['TIMESTART'])

        df_tekort = df.loc[periode_start:periode_end, f'watertekort_{gebruiker}']

        totaal_tekort = df_tekort.sum()

        referentie.invoerdata['schaalgebied'].loc[index, f'{aspect}{gebruiker}'] = totaal_tekort.item()
        
    except FileNotFoundError:
        print(f"{row['CODE']} not found")

fig, ax = plt.subplots(figsize=(20, 15))
cbar = referentie.invoerdata['schaalgebied'].to_crs(epsg=3857).plot(ax=ax, column=f'{aspect}{gebruiker}', legend=True)
cx.add_basemap(ax)
ax.set_title(f'Watertekort {gebruiker} in periode {periode_start} - {periode_end}')
# ax_cbar = fig.colorbar(cbar, ax=ax)
# ax_cbar.set_ylabel('Watertekort [$m^3$]')

#%% visualisatie van onzekerheid oppervlakte top10nl (b.v. watervraag landbouw)
code_afwateringsgebied = 'PG0566-1'
code_afwateringsgebied = 'PG0714-42'

df_ref = pd.read_csv(f'{ref_dir}\mozart_schaalgebied_{code_afwateringsgebied}_geschaald.csv', parse_dates=True, index_col=[0])
df_min5 = pd.read_csv(f'{min5_dir}\mozart_schaalgebied_{code_afwateringsgebied}_geschaald.csv', parse_dates=True, index_col=[0])
df_plus5 = pd.read_csv(f'{plus5_dir}\mozart_schaalgebied_{code_afwateringsgebied}_geschaald.csv', parse_dates=True, index_col=[0])

fig, ax = plt.subplots()

# df_ref['watervraag_landbouw'].plot(ax=ax, legend=False)
ax.fill_between(df_min5.index, df_min5['watervraag_landbouw'], df_plus5['watervraag_landbouw'])

#%% bepalen van klimaatscenarios
# definieer de scenarios
scenarios = ['REF2017', 'W2050']

# definieer de map waar de scenario-data staan
directory = r'd:\projecten\hydrologische_informatieproducten\werkmap\data\lhm_toekomst\\'

# bepaal factor d.m.v. klimaatscenarios
referentie.bepaal_factor_klimaatscenarios(directory, scenarios)

#%% visualisatie van onzekerheid klimaatscenarios

# definieer onzekerheid klimaatfactor
onzekerheid_klimaat = 0.05 # 5%

df_ref_watervraag_toekomst = df_ref*referentie.klimaatfactor['watervraag']
df_min5_watervraag_toekomst = df_ref*(referentie.klimaatfactor['watervraag']-onzekerheid_klimaat)
df_plus5_watervraag_toekomst = df_ref*(referentie.klimaatfactor['watervraag']+onzekerheid_klimaat)

fig, ax = plt.subplots()

df_ref['watervraag_landbouw'].plot(ax=ax, legend=False)
df_ref_watervraag_toekomst['watervraag_landbouw'].plot(ax=ax, legend=False)
ax.fill_between(df_min5.index, df_min5_watervraag_toekomst['watervraag_landbouw'], df_plus5_watervraag_toekomst['watervraag_landbouw'],
                color='grey')
