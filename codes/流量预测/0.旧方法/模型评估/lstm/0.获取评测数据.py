import requests
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np


start_time = datetime(2025, 5, 28) 
end_time = datetime.now()

names = [
        'DLDZ_DQ200_SYSTEM_PI05.PV',
        'DLDZ_AVS_SYSTEM_PI05.PV',
        'DLDZ_DQ200_LLJ01_FI01.PV',
        'DLDZ_AVS_LLJ01_FI01.PV'
        ]

# 构建API请求参数
params = {
    "startTime": start_time.isoformat(timespec='milliseconds'),
    "endTime": end_time.isoformat(timespec='milliseconds'),
    "interval": 60000, 
    "valueonly": 0,
    "decimal": 2,
    "names": ','.join(names)
}

try:
    response = requests.get(
        "",
        params=params,
        timeout=10  
    )
    response.raise_for_status() 
except Exception as e:
    raise Exception(f"API请求失败: {str(e)}")

data = response.json()
items = data.get('items', [])

import pandas as pd


timestamp_dict = {}
columns = []

for item in items:
    var_name = item['name']
    columns.append(var_name)
    
    for val in item['vals']:
        timestamp = val['time']
        value = val['val']
        
        if timestamp not in timestamp_dict:
            timestamp_dict[timestamp] = {}
        
        timestamp_dict[timestamp][var_name] = value

df = pd.DataFrame([
    {'timestamp': ts, **values} for ts, values in timestamp_dict.items()
])


df['timestamp'] = pd.to_datetime(df['timestamp'])


def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return (series < lower_bound) | (series > upper_bound)

column = ['DLDZ_DQ200_SYSTEM_PI05.PV', 'DLDZ_AVS_SYSTEM_PI05.PV',
       'DLDZ_DQ200_LLJ01_FI01.PV', 'DLDZ_AVS_LLJ01_FI01.PV']


for column in df.select_dtypes(include=[np.number]).columns:
    outliers_mask = detect_outliers_iqr(df[column])
    df.loc[outliers_mask, column] = np.nan
    df[column] = df[column].interpolate(method='linear')


df_cleaned = df.dropna()
df_cleaned['瞬时流量'] = df_cleaned['DLDZ_DQ200_LLJ01_FI01.PV'] + df_cleaned['DLDZ_AVS_LLJ01_FI01.PV']


out_path = r''
df_cleaned.to_excel(out_path, index=False)