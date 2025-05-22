import requests
from datetime import datetime, timedelta
from fastapi import HTTPException
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd

def fetch_realtime_data():
    """
    调用数据库API获取实时数据，并转换为模型输入格式
    """
    # 获取最近一个小时的数据
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    
    # 构建API请求参数
    params = {
        "startTime": start_time.isoformat(timespec='milliseconds'),
        "endTime": end_time.isoformat(timespec='milliseconds'),
        "interval": 60000,   # 1分钟
        "valueonly": 0,
        "decimal": 2,
        "names": "DLDZ_DQ200_SYSTEM_PI05.PV,DLDZ_AVS_SYSTEM_PI05.PV,DLDZ_DQ200_LLJ01_FQ01.PV,DLDZ_AVS_LLJ01_FQ01.PV"
    }
    
    # 发送GET请求
    try:
        response = requests.get(
            "http://8.130.25.118:8000/api/hisdata",
            params=params,
            timeout=10  # 设置超时时间
        )
        response.raise_for_status()  # 检查HTTP状态码
    except Exception as e:
        raise Exception(f"API请求失败: {str(e)}")
    
    # 解析响应数据
    data = response.json()
    if data['code'] != 0:
        raise ValueError(f"API返回错误码: {data['code']}")
    
    # 变量映射关系
    variable_mapping = {
        "DLDZ_DQ200_SYSTEM_PI05.PV": "DQ200系统压力",
        "DLDZ_AVS_SYSTEM_PI05.PV": "AVS系统压力",
        "DLDZ_DQ200_LLJ01_FQ01.PV": "DQ200累积流量",
        "DLDZ_AVS_LLJ01_FQ01.PV": "AVS累积流量"
    }
    
    input_data = {}
    
    # 提取各变量的最新值
    for var_api, var_model in variable_mapping.items():
        item = next((item for item in data['items'] if item['name'] == var_api), None)
        if not item:
            raise ValueError(f"未找到变量: {var_api}")
        if not item['vals']:
            raise ValueError(f"变量 {var_api} 数据为空")
        input_data[var_model] = item['vals'][-1]['val']
    
    # 计算系统压力的差分值（最后两个数据点差值取整）
    def calculate_diff(var_name):
        item = next(item for item in data['items'] if item['name'] == var_name)
        vals = item['vals']
        # 计算最后两个点的数据差值
        if len(vals) >= 2:
            return round(vals[-1]['val'] - vals[-2]['val'])
        return 0
    
    input_data["DQ200差分值"] = calculate_diff("DLDZ_DQ200_LLJ01_FQ01.PV")
    input_data["AVS差分值"] = calculate_diff("DLDZ_AVS_LLJ01_FQ01.PV")
    
    # 生成时间特征（使用当前时间）
    current_time = datetime.now()
    input_data.update({
        "year": current_time.year,
        "month": current_time.month,
        "day": current_time.day,
        "weekday": current_time.weekday(),  # 0=Monday, 6=Sunday
        "hour": current_time.hour,
        "minute": current_time.minute,
        "second": current_time.second,
        "is_weekend": 1 if current_time.weekday() in [5, 6] else 0,
        "quarter": (current_time.month - 1) // 3 + 1
    })
    
    return input_data


app = FastAPI()
# 加载模型
model = joblib.load("xgb_reg_model.pkl")

# FastAPI端点示例
@app.post("/predict_realtime/")
async def predict_realtime():
    try:
        input_data = fetch_realtime_data()
        feature_array = np.array(list(input_data.values())).reshape(1, -1)
        prediction = model.predict(feature_array)
        return {"prediction": prediction.tolist()[0], "input_data": input_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# 根路径
@app.get("/")
def read_root():
    return {"Health_check": "OK"}

@app.get("/info")
def read_info():
    return {"Name":"英格索兰项目","Version":"1.0.0","Author":"赵泽宸","Description":"基于XGBoost的瞬时用气量预测模型"}
# 启动服务
# uvicorn main:app --reload