import pandas as pd
import json
from datetime import datetime

# 读取数据并解析时间
file_path = r''
df = pd.read_excel(file_path, parse_dates=["time"])
df["hour"] = df["time"].dt.hour  # 提取小时信息

# 定义设备列和产气量列
device_cols = [
    "DLDZ_AVS_KYJ01_YI01.PV",
    "DLDZ_AVS_KYJ02_YI01.PV",
    "DLDZ_AVS_KYJ03_YI01.PV",
    "DLDZ_AVS_KYJ04_YI01.PV",
    "DLDZ_AVS_KYJ05_YI01.PV"
]
gas_col = "DLDZ_AVS_LLJ01_FI01.PV"  # 假设目标产气量列

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
        device_count = sum(combo["devices"].values())
        
        if device_count not in grouped_combinations:
            grouped_combinations[device_count] = []
            
        grouped_combinations[device_count].append(combo)
    
    # 将数字键转换为字符串并存储最终结构
    hourly_config[str(hour_key)] = {
        str(k): v for k, v in grouped_combinations.items()
    }

# 保存为JSON配置文件
with open("hourly_config.json", "w") as f:
    json.dump(hourly_config, f, indent=2, ensure_ascii=False)