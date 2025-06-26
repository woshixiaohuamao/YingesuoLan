import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SundayDataFetcher")

def fetch_sunday_data(weeks: int, output_dir="sunday_data"):
    """
    获取指定周数的周日数据（30分钟间隔）
    
    参数:
        weeks (int): 需要获取的周数
        output_dir (str): 数据保存目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算最近的周日日期
    today = datetime.now()
    last_sunday = today - timedelta(days=(today.weekday() + 1) % 7)
    
    # 变量列表
    names = [
        'DLDZ_DQ200_LLJ01_FI01.PV',
        'DLDZ_AVS_LLJ01_FI01.PV'
    ]
    
    # 存储所有周日数据
    all_sunday_data = []
    
    # 遍历每一周
    for i in range(weeks):
        # 计算当前周日的日期
        sunday_date = last_sunday - timedelta(weeks=i)
        
        # 设置时间范围：周日00:00到23:30
        start_time = sunday_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = sunday_date.replace(hour=23, minute=30, second=0, microsecond=0)
        
        logger.info(f"获取 {sunday_date.strftime('%Y-%m-%d')} 的周日数据...")
        
        # API请求 - 获取周日数据
        params = {
            "startTime": start_time.isoformat(timespec='milliseconds'),
            "endTime": end_time.isoformat(timespec='milliseconds'),
            "interval": 1800000,  # 30分钟间隔 (30*60*1000ms)
            "valueonly": 0,
            "decimal": 2,
            "names": ','.join(names)
        }
        
        try:
            response = requests.get("http://8.130.25.118:8000/api/hisdata", params=params, timeout=30)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"获取 {sunday_date.strftime('%Y-%m-%d')} 数据失败: {str(e)}")
            continue
        
        # 解析响应数据
        data = response.json()
        
        # 构建DataFrame
        df = pd.DataFrame()
        for item in data.get('items', []):
            temp_df = pd.DataFrame(item['vals'])
            temp_df['name'] = item['name']
            df = pd.concat([df, temp_df])
        
        if df.empty:
            logger.warning(f"{sunday_date.strftime('%Y-%m-%d')} 无数据")
            continue
        
        # 数据透视
        df_pivot = df.pivot(index='time', columns='name', values='val').reset_index()
        df_pivot.columns.name = None
        df_pivot.rename(columns={'time': 'timestamp'}, inplace=True)
        
        # 转换为时间格式
        df_pivot['timestamp'] = pd.to_datetime(df_pivot['timestamp'])
        
        # 创建完整的30分钟时间索引 (周日应该有48个点)
        full_range = pd.date_range(start=start_time, end=end_time, freq='30T')
        df_full = pd.DataFrame({'timestamp': full_range})
        
        # 合并确保时间序列完整
        df_merged = df_full.merge(df_pivot, on='timestamp', how='left')
        
        # 检查数据完整性
        expected_points = 48
        actual_points = len(df_merged)
        if actual_points < expected_points:
            logger.warning(f"{sunday_date.strftime('%Y-%m-%d')} 数据不完整: {actual_points}/{expected_points}")
        
        # 填充缺失值
        numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_merged[col] = df_merged[col].interpolate(method='linear', limit_direction='both')
            df_merged[col] = df_merged[col].fillna(0)  # 确保无NaN值
        
        # 计算瞬时流量
        df_merged['瞬时流量'] = df_merged['DLDZ_DQ200_LLJ01_FI01.PV'] + df_merged['DLDZ_AVS_LLJ01_FI01.PV']
        
        # 只保留timestamp和瞬时流量列
        result_df = df_merged[['timestamp', '瞬时流量']].copy()
        
        # 添加到总数据
        all_sunday_data.append(result_df)
        
        logger.info(f"成功获取 {sunday_date.strftime('%Y-%m-%d')} 的 {len(result_df)} 个数据点")
    
    # 合并所有周日数据
    if not all_sunday_data:
        logger.error("未获取到任何周日数据")
        return
    
    combined_df = pd.concat(all_sunday_data, ignore_index=True)
    
    # 按时间戳排序
    combined_df.sort_values('timestamp', inplace=True)
    
    # 保存为Excel文件
    output_file = os.path.join(output_dir, f"sunday_data.xlsx")
    combined_df.to_excel(output_file, index=False)
    logger.info(f"已保存周日数据到: {output_file} ({len(combined_df)} 行)")
    
    return combined_df

if __name__ == "__main__":
    # 示例：获取最近8周的周日数据
    fetch_sunday_data(weeks=4)