"""
用来获取最近days天的数据，并处理成完整的30分钟间隔数据
"""
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataFetcher")

def fetch_recent_data(days: int):
    """获取指定天数的完整数据（30分钟间隔）"""

    now = datetime.now()
    end_time = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(microseconds=1)
    start_time = (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    names = [
        'DLDZ_DQ200_SYSTEM_PI05.PV',
        'DLDZ_AVS_SYSTEM_PI05.PV',
        'DLDZ_DQ200_LLJ01_FI01.PV',
        'DLDZ_AVS_LLJ01_FI01.PV'
    ]
    
    params = {
        "startTime": start_time.isoformat(timespec='milliseconds'),
        "endTime": end_time.isoformat(timespec='milliseconds'),
        "interval": 1800000, 
        "valueonly": 0,
        "decimal": 2,
        "names": ','.join(names)
    }
    
    try:
        logger.info(f"请求数据范围: {start_time} 至 {end_time} (最近{days}天)")
        response = requests.get("", params=params, timeout=30)
        response.raise_for_status()
        logger.info("API请求成功，开始处理数据...")
    except Exception as e:
        logger.error(f"API请求失败: {str(e)}")
        raise
    
    data = response.json()
    
    df = pd.DataFrame()
    for item in data.get('items', []):
        temp_df = pd.DataFrame(item['vals'])
        temp_df['name'] = item['name']
        df = pd.concat([df, temp_df])
    
    if df.empty:
        logger.warning("API返回空数据集")
        return pd.DataFrame()
    
    df_pivot = df.pivot(index='time', columns='name', values='val').reset_index()
    df_pivot.columns.name = None
    df_pivot.rename(columns={'time': 'timestamp'}, inplace=True)
    
    df_pivot['timestamp'] = pd.to_datetime(df_pivot['timestamp'])
    
    full_range = pd.date_range(start=start_time, end=end_time, freq='30T')
    df_full = pd.DataFrame({'timestamp': full_range})
    
    df_merged = df_full.merge(df_pivot, on='timestamp', how='left')
    
    expected_points = len(full_range)
    actual_points = len(df_pivot)
    missing_points = expected_points - actual_points
    logger.info(f"预期数据点: {expected_points}, 实际获取: {actual_points}, 缺失点: {missing_points}")
    
    nan_count_before = df_merged.isna().sum().sum()
    logger.info(f"填充前NaN值总数: {nan_count_before}")
    
    numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_merged[col] = df_merged[col].interpolate(method='linear', limit_direction='both')
    
    nan_count_after = df_merged.isna().sum().sum()
    logger.info(f"填充后NaN值总数: {nan_count_after}")
    
    df_merged['瞬时流量'] = df_merged['DLDZ_DQ200_LLJ01_FI01.PV'] + df_merged['DLDZ_AVS_LLJ01_FI01.PV']
    df_merged['总压力'] = (df_merged['DLDZ_DQ200_SYSTEM_PI05.PV'] + df_merged['DLDZ_AVS_SYSTEM_PI05.PV']) / 2

    result_df = df_merged[['timestamp', '瞬时流量', '总压力']].copy()
    
    min_time = result_df['timestamp'].min().strftime("%Y-%m-%d %H:%M")
    max_time = result_df['timestamp'].max().strftime("%Y-%m-%d %H:%M")
    logger.info(f"最终数据时间范围: {min_time} 至 {max_time}")
    
    return result_df


def fetch_today_data():
    """获取调用当天的数据（从当天0点到当前时刻），数据可能不完整"""
    now = datetime.now()
    start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)  # 当天0点
    end_time = now 

    names = [
        'DLDZ_DQ200_SYSTEM_PI05.PV',
        'DLDZ_AVS_SYSTEM_PI05.PV',
        'DLDZ_DQ200_LLJ01_FI01.PV',
        'DLDZ_AVS_LLJ01_FI01.PV'
    ]

    params = {
        "startTime": start_time.isoformat(timespec='milliseconds'),
        "endTime": end_time.isoformat(timespec='milliseconds'),
        "interval": 1800000,  
        "valueonly": 0,
        "decimal": 2,
        "names": ','.join(names)
    }

    try:
        logger.info(f"请求数据范围: {start_time} 至 {end_time} (当天实时数据)")
        response = requests.get("", params=params, timeout=30)
        response.raise_for_status()
        logger.info("API请求成功，开始处理当天数据...")
    except Exception as e:
        logger.error(f"API请求失败: {str(e)}")
        raise

    data = response.json()

    df = pd.DataFrame()
    for item in data.get('items', []):
        temp_df = pd.DataFrame(item['vals'])
        temp_df['name'] = item['name']
        df = pd.concat([df, temp_df])

    if df.empty:
        logger.warning("API返回空数据集")
        return pd.DataFrame()

    df_pivot = df.pivot(index='time', columns='name', values='val').reset_index()
    df_pivot.columns.name = None
    df_pivot.rename(columns={'time': 'timestamp'}, inplace=True)
    df_pivot['timestamp'] = pd.to_datetime(df_pivot['timestamp'])

    full_range = pd.date_range(start=start_time, end=end_time, freq='30T')
    df_full = pd.DataFrame({'timestamp': full_range})

    df_merged = df_full.merge(df_pivot, on='timestamp', how='left')

    expected_points = len(full_range)
    actual_points = len(df_pivot)
    missing_points = expected_points - actual_points
    logger.info(f"当天预期数据点: {expected_points}, 实际获取: {actual_points}, 缺失点: {missing_points}")

    nan_count_before = df_merged.isna().sum().sum()
    logger.info(f"填充前NaN值总数: {nan_count_before}")

    numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_merged[col] = df_merged[col].interpolate(method='linear', limit_direction='both')

    nan_count_after = df_merged.isna().sum().sum()
    logger.info(f"填充后NaN值总数: {nan_count_after}")

    df_merged['瞬时流量'] = df_merged['DLDZ_DQ200_LLJ01_FI01.PV'] + df_merged['DLDZ_AVS_LLJ01_FI01.PV']
    df_merged['总压力'] = (df_merged['DLDZ_DQ200_SYSTEM_PI05.PV'] + df_merged['DLDZ_AVS_SYSTEM_PI05.PV']) / 2

    result_df = df_merged[['timestamp', '瞬时流量', '总压力']].copy()

    min_time = result_df['timestamp'].min().strftime("%Y-%m-%d %H:%M")
    max_time = result_df['timestamp'].max().strftime("%Y-%m-%d %H:%M")
    logger.info(f"当天数据时间范围: {min_time} 至 {max_time}")

    return result_df
