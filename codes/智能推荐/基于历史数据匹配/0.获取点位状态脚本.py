import requests
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
import json

# 固定开始时间
start_time = datetime(2025, 5, 29 , 12 , 0 , 0)  # 2025年5月1日 00:00:00

# 结束时间为当前时间
end_time = datetime.now()

names = ['DLDZ_DQ200_KYJ01_YI01.PV','DLDZ_DQ200_KYJ02_YI02.PV','DLDZ_DQ200_KYJ03_YI02.PV',
         'DLDZ_DQ200_KYJ04_YI02.PV','DLDZ_DQ200_KYJ05_YI02.PV','DLDZ_DQ200_KYJ06_YI02.PV',
         'DLDZ_AVS_KYJ01_YI01.PV','DLDZ_AVS_KYJ02_YI01.PV','DLDZ_AVS_KYJ03_YI01.PV',
         'DLDZ_AVS_KYJ04_YI01.PV','DLDZ_AVS_KYJ05_YI01.PV',
         'DLDZ_DQ200_LLJ01_FI01.PV','DLDZ_AVS_LLJ01_FI01.PV'
        ]

# 构建API请求参数
params = {
    "startTime": start_time.isoformat(timespec='milliseconds'),
    "endTime": end_time.isoformat(timespec='milliseconds'),
    "interval": 3600000,  # 1小时
    "valueonly": 0,
    "decimal": 2,
    "names": ','.join(names)
}

# 发送GET请求
try:
    response = requests.get(
        "",
        params=params,
        timeout=10  # 设置超时时间
    )
    response.raise_for_status()  # 检查HTTP状态码
except Exception as e:
    raise Exception(f"API请求失败: {str(e)}")
# 解析响应数据
data = response.json()

# 创建主DataFrame
master_df = pd.DataFrame()

for item in data["items"]:
    # 确保每个测点有 'name' 和 'vals'
    if "name" not in item or "vals" not in item:
        print(f"无效的测点数据: {item}")
        continue
    
    site_name = item["name"]
    vals = item["vals"]
    
    # 检查 vals 是否非空且包含 'time' 和 'val'
    if not vals or any("time" not in entry or "val" not in entry for entry in vals):
        print(f"测点 {site_name} 的数据格式错误或缺失字段")
        continue
    
    # 转换为DataFrame
    df = pd.DataFrame(vals)
    
    # 验证列是否存在
    if "time" not in df.columns or "val" not in df.columns:
        print(f"测点 {site_name} 的 DataFrame 缺少 'time' 或 'val' 列")
        continue
    
    # 转换时间格式并设置索引
    try:
        # 直接转换为datetime对象，不转字符串
        df["time"] = pd.to_datetime(df["time"])  # 移除了.dt.strftime()
        df_single = df.set_index("time")[["val"]].rename(columns={"val": site_name})
    except Exception as e:
        print(f"处理测点 {site_name} 时出错: {str(e)}")
        continue
    
    # 合并到主表
    if master_df.empty:
        master_df = df_single
    else:
        master_df = master_df.join(df_single, how="outer")

# 重置索引并将时间移到第一列

df = master_df.reset_index().rename(columns={"index": "time"})
df["hour"] = df["time"].dt.hour  # 提取小时信息
df['all_gas'] = df['DLDZ_DQ200_LLJ01_FI01.PV'] + df['DLDZ_AVS_LLJ01_FI01.PV']

# 定义设备列和产气量列
device_cols = [
    'DLDZ_DQ200_KYJ01_YI01.PV','DLDZ_DQ200_KYJ02_YI02.PV','DLDZ_DQ200_KYJ03_YI02.PV',
    'DLDZ_DQ200_KYJ04_YI02.PV','DLDZ_DQ200_KYJ05_YI02.PV','DLDZ_DQ200_KYJ06_YI02.PV',
    "DLDZ_AVS_KYJ01_YI01.PV",
    "DLDZ_AVS_KYJ02_YI01.PV",
    "DLDZ_AVS_KYJ03_YI01.PV",
    "DLDZ_AVS_KYJ04_YI01.PV",
    "DLDZ_AVS_KYJ05_YI01.PV"
]
gas_col = "all_gas"  # 假设目标产气量列

hourly_config = {}

for hour in df["hour"].unique():
    hour_key = int(hour)
    hour_data = df[df["hour"] == hour]
    
    combinations = []
    for _, row in hour_data.iterrows():
        # 生成设备状态字典并转换为Python原生类型
        device_status = {col: int(row[col]) for col in device_cols}
        gas_value = float(row[gas_col])
        
        combinations.append({
            "devices": device_status,
            "gas": gas_value
        })
    
    if not combinations:
        continue
    
    # 按设备数量分组
    grouped_combinations = {}
    for combo in combinations:
        # 计算当前组合的设备数量
        device_count = 0
        for val in combo["devices"].values():
            if val != -9999:  # 跳过通信失败的点位
                device_count += int(val)  # 只累加有效设备的状态值
                
        if device_count not in grouped_combinations:
            grouped_combinations[device_count] = []
            
        grouped_combinations[device_count].append(combo)
    
    # 将数字键转换为字符串并存储最终结构
    hourly_config[str(hour_key)] = {
        str(k): v for k, v in grouped_combinations.items()
    }

# 保存为JSON配置文件
with open("hourly_config_new.json", "w") as f:
    json.dump(hourly_config, f, indent=2, ensure_ascii=False)
