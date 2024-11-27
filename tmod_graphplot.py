import os
import pandas as pd
from datetime import datetime
import re

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pvlib

site_141 = 'site141.csv' 
site_142 = 'site142.csv'
df_141 = pd.read_csv(site_141, index_col=0, parse_dates=True)
df_142 = pd.read_csv(site_142, index_col=0, parse_dates=True)


# choosing windspeed and Tamb data
col1 = ['AvgTamb_1','AvgWindS_1']

df1_selected = df_141[col1].rename(columns = {
    'AvgTamb_1':'Tamb',
    'AvgWindS_1':'WindSp'} )

# choosing poa and Tmod data
col2 = ['AvgGmod09_2','AvgGmod09_3','AvgTmod_2','AvgTmod_3']

df2_selected = df_142[col2].rename(columns = {
    'AvgGmod09_2':'poa_2',
    'AvgGmod09_3':'poa_3'})


# concatenate into new dataframe
df_selected = pd.concat([df1_selected, df2_selected], axis=1)

# mask the outliers
filters = {
    'poa':[0, 1500],
    'AvgT':[0, 100],}

def apply_filters(df, filters):

    for substring, (min_val, max_val) in filters.items():

        # Find columns that contain the substring

        substring_columns = df.columns[df.columns.str.contains(substring)]

        for col in substring_columns:

            df.loc[:, col] = df[col].where((df[col] >= min_val) & (df[col] <= max_val), np.nan)
            # count null entries of the column
    return df


filtered_df = apply_filters(df_selected, filters)

# There is some missing data, infer the frequency from the first several data points
# freq = pd.infer_freq(filtered_df.index[:10])
 
# interpolate
# filtered_df['poa_3'] = rdtools.interpolate(filtered_df['poa_3'], freq)

# Specify the Metadata
meta1 = {"latitude": 14.49422,
        "longitude": 120.5986,
        "timezone": "Asia/Manila",
        "gamma_pdc": -0.284, 
        "azimuth": 180,
        "tilt": 9,
        "power_dc_rated": 1000.0,
        "temp_model_params":
        pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']}
meta2 = {"latitude": 14.49422,
        "longitude": 120.5986,
        "timezone": "Asia/Manila",
        "gamma_pdc": -0.284, 
        "azimuth": 180,
        "tilt": 9,
        "power_dc_rated": 1000.0,
        "temp_model_params":
        pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']}

filtered_df.index = filtered_df.index.tz_localize(meta1['timezone'])
 
# Calculate cell temperature
filtered_df['Tcell_1'] = pvlib.temperature.sapm_cell(filtered_df['poa_1'], filtered_df['Tamb'], filtered_df['WindSp'],
                                            **meta1['temp_model_params'])

filtered_df['Tcell_7'] = pvlib.temperature.sapm_cell(filtered_df['poa_7'], filtered_df['Tamb'], filtered_df['WindSp'],
                                           **meta1['temp_model_params'])

filtered_df['Tcell_20'] = pvlib.temperature.sapm_cell(filtered_df['poa_20'], filtered_df['Tamb'], filtered_df['WindSp'],
                                           **meta2['temp_model_params'])

filtered_df['diff_1'] = filtered_df['Tcell_1'] - filtered_df['AvgTmod_1']
filtered_df['diff_7'] = filtered_df['Tcell_7'] - filtered_df['AvgTmod_7']
filtered_df['diff_20'] = filtered_df['Tcell_20'] - filtered_df['AvgTmod_20']

daily_data = filtered_df.resample('D').mean()
daily_temp = daily_data[['Tcell_1','AvgTmod_1','diff_1','Tcell_7','AvgTmod_7','diff_7','Tcell_20','AvgTmod_20','diff_20']]

 

#df_selected.wind_speed

# For rooftop (102-112) please use ]['close_rack_glass_polymer'], 
# for BATLAB please use ‘open_rack_glass_polymer’
# In the degradation function,
# where no commented, please use glass/polymer | open rack     ['open_rack_glass_polymer']
# where Bifacial, please use the glass/glass   | open rack ['open_rack_glass_glass']
# where thin-film, please treat as glass/polymer open rack, ['open_rack_glass_polymer']

# daily mean scatter plot
plt.figure(figsize=(9, 6))
plt.plot(daily_temp.index, daily_temp['Tcell_20'], # change the columns for another solar panel
         linestyle='None', marker='o', alpha = 0.4, ms = 3, color='royalblue', label = f'Theoretical temperature')

plt.plot(daily_temp.index, daily_temp['AvgTmod_20'], # change the columns for another solar panel
        linestyle='None', marker='o', alpha = 0.4, ms = 3, color='green', label = 'Actual temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (Celsius)')
plt.title('Theorectical and Actual daily mean module temperatures (144-20)') # change graph labels
plt.grid(color ='grey', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.show()

# monthly diff bar plot
daily_temp['Date'] = daily_temp.index.to_period('M').astype(str)
daily_temp = daily_temp.sort_values(by = 'Date')

plt.figure(figsize=(12, 6))
sns.boxplot(data = daily_temp, x = 'Date', y = 'diff_20')
plt.title('Monthly Boxplots of the Difference between Theoretical and Actual Module Temperature (144-20)')
plt.xlabel('Month and Year')
plt.ylabel('Temperature Difference (Celsius)')
plt.grid(color ='grey', linestyle = '--', linewidth = 0.5)
plt.xticks(rotation=45, ha='right')
plt.show()
