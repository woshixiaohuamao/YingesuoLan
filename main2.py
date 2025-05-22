import yaml
import requests
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic_settings import BaseSettings

# 加载配置文件
with open("config.yml", "r", encoding="utf-8") as f:  
    config = yaml.safe_load(f)

# 从配置中提取参数
API_URL = config["api_config"]["url"]
API_PARAMS = config["api_config"]["params"]
VARIABLE_MAPPING = config["variable_mapping"]
MODEL_PATH = config["model_config"]["model_path"]
TIME_FEATURES = config["time_features"]
APP_INFO = config["app_info"]
# 处理 names 参数：将列表转为逗号分隔的字符串
if "names" in API_PARAMS and isinstance(API_PARAMS["names"], list):
    API_PARAMS["names"] = ",".join(API_PARAMS["names"])

# 初始化FastAPI应用
app = FastAPI()

# 加载模型
model = joblib.load(MODEL_PATH)

def fetch_realtime_data():
    """从API获取实时数据（使用YAML配置参数）"""
    # 时间范围计算
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    
    # 构建动态参数
    params = {
        "startTime": start_time.isoformat(timespec='milliseconds'),
        "endTime": end_time.isoformat(timespec='milliseconds'),
        **API_PARAMS  # 合并固定参数
    }

    # 发送请求
    try:
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
    except Exception as e:
        raise Exception(f"API请求失败: {str(e)}")
    
    # 解析响应
    data = response.json()
    if data['code'] != 0:
        raise ValueError(f"API返回错误码: {data['code']}")
    
    # 提取变量数据
    input_data = {}
    for var_api, var_model in VARIABLE_MAPPING.items():
        item = next((item for item in data['items'] if item['name'] == var_api), None)
        if not item:
            raise ValueError(f"未找到变量: {var_api}")
        if not item['vals']:
            raise ValueError(f"变量 {var_api} 数据为空")
        input_data[var_model] = item['vals'][-1]['val']
    
    # 计算差分值
    def calculate_diff(var_name):
        item = next(item for item in data['items'] if item['name'] == var_name)
        vals = item['vals']
        return round(vals[-1]['val'] - vals[-2]['val']) if len(vals) >=2 else 0
    
    input_data["DQ200差分值"] = calculate_diff("DLDZ_DQ200_LLJ01_FQ01.PV")
    input_data["AVS差分值"] = calculate_diff("DLDZ_AVS_LLJ01_FQ01.PV")
    
    # 时间特征生成（根据配置开关）
    if TIME_FEATURES["enabled"]:
        current_time = datetime.now()
        time_features = {
            "year": current_time.year,
            "month": current_time.month,
            "day": current_time.day,
            "hour": current_time.hour,
            "minute": current_time.minute,
            "second": current_time.second
        }
        if TIME_FEATURES["include_weekday"]:
            time_features["weekday"] = current_time.weekday()
        if TIME_FEATURES["include_is_weekend"]:
            time_features["is_weekend"] = 1 if current_time.weekday() in [5,6] else 0
        if TIME_FEATURES["include_quarter"]:
            time_features["quarter"] = (current_time.month - 1) // 3 + 1
        
        input_data.update(time_features)
    
    return input_data

# API端点保持不变
@app.post("/predict_realtime/")
async def predict_realtime():
    try:
        input_data = fetch_realtime_data()
        feature_array = np.array(list(input_data.values())).reshape(1, -1)
        prediction = model.predict(feature_array)
        return {"prediction": prediction.tolist()[0], "input_data": input_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"Health_check": "OK"}

@app.get("/info")
def read_info():
    return APP_INFO  # 直接返回配置中的信息