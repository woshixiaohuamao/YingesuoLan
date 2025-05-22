import requests
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import uvicorn

def fetch_realtime_data():
    """
    获取实时数据并计算特征
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    
    params = {
        "startTime": start_time.isoformat(timespec='milliseconds'),
        "endTime": end_time.isoformat(timespec='milliseconds'),
        "interval": 60000,   # 1分钟间隔
        "valueonly": 0,
        "decimal": 2,
        "names": "DLDZ_DQ200_SYSTEM_PI05.PV,DLDZ_AVS_SYSTEM_PI05.PV,DLDZ_DQ200_LLJ01_FQ01.PV,DLDZ_AVS_LLJ01_FQ01.PV"
    }
    
    try:
        response = requests.get(
            "http://8.130.25.118:8000/api/hisdata",
            params=params,
            timeout=10
        )
        response.raise_for_status()
    except Exception as e:
        raise Exception(f"API请求失败: {str(e)}")
    
    data = response.json()
    if data['code'] != 0:
        raise ValueError(f"API返回错误码: {data['code']}")

    # 获取原始数据
    var_dq200 = "DLDZ_DQ200_LLJ01_FQ01.PV"
    var_avs = "DLDZ_AVS_LLJ01_FQ01.PV"
    
    # 处理DQ200数据
    dq200_item = next((item for item in data['items'] if item['name'] == var_dq200), None)
    avs_item = next((item for item in data['items'] if item['name'] == var_avs), None)
    
    # 验证数据完整性
    if not dq200_item or not avs_item:
        raise ValueError("缺少必要变量数据")
    if len(dq200_item['vals']) < 2 or len(avs_item['vals']) < 2:
        raise ValueError("数据点不足，需要至少2个数据点")

    # 获取最新系统压力值
    dq200_pressure = dq200_item['vals'][-1]['val']  # DQ200系统压力
    avs_pressure = avs_item['vals'][-1]['val']     # AVS系统压力

    # 计算差分平均值
    def calculate_avg_diff(vals):
        diffs = [vals[i]['val'] - vals[i-1]['val'] for i in range(1, len(vals))]
        return round(np.mean(diffs), 2)
    
    dq200_flow = calculate_avg_diff(dq200_item['vals'])  # DQ200总流量
    avs_flow = calculate_avg_diff(avs_item['vals'])     # AVS总流量

    # 时间特征
    current_time = datetime.now()
    time_features = {
        "year": current_time.year,
        "month": current_time.month,
        "day": current_time.day,
        "weekday": current_time.weekday(),
        "hour": current_time.hour,
        "minute": current_time.minute,
        "second": current_time.second,
        "is_weekend": 1 if current_time.weekday() in [5, 6] else 0,
        "quarter": (current_time.month - 1) // 3 + 1
    }

    # 构建两个模型的特征字典
    features_avs = {
        "DQ200系统压力": dq200_pressure,
        "AVS系统压力": avs_pressure,
        "DQ200总流量": dq200_flow,
        **time_features
    }
    
    features_dq200 = {
        "DQ200系统压力": dq200_pressure,
        "AVS系统压力": avs_pressure,
        "AVS总流量": avs_flow,
        **time_features
    }

    return features_avs, features_dq200

app = FastAPI()

# 加载模型并定义特征顺序
model_avs = joblib.load("./Fastapi/xgb_avs_time.pkl")
model_dq200 = joblib.load("./Fastapi/xgb_dq200_part.pkl")

# 两个模型的特征顺序
AVS_FEATURES = [
    'DQ200系统压力', 'AVS系统压力', 'DQ200总流量', 
    'year', 'month', 'day', 'weekday', 
    'hour', 'minute', 'second', 'is_weekend', 'quarter'
]

DQ200_FEATURES = [
    'DQ200系统压力', 'AVS系统压力', 'AVS总流量',
    'year', 'month', 'day', 'weekday', 
    'hour', 'minute', 'second', 'is_weekend', 'quarter'
]

@app.post("/predict_realtime/")
async def predict_realtime():
    try:
        features_avs, features_dq200 = fetch_realtime_data()
        
        # AVS模型预测
        avs_input = np.array([features_avs[key] for key in AVS_FEATURES]).reshape(1, -1)
        avs_pred = model_avs.predict(avs_input)[0]
        
        # DQ200模型预测
        dq200_input = np.array([features_dq200[key] for key in DQ200_FEATURES]).reshape(1, -1)
        dq200_pred = model_dq200.predict(dq200_input)[0]
        
        return {
            "predictions": {
                "AVS_model": float(avs_pred),
                "DQ200_model": float(dq200_pred)
            },
            "input_features": {
                "AVS_model_input": features_avs,
                "DQ200_model_input": features_dq200
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 其他端点保持不变
@app.get("/")
def read_root():
    return {"Health_check": "OK"}

@app.get("/info")
def read_info():
    return {
        "Name": "英格索兰项目",
        "Version": "2.0.0",
        "Author": "赵泽宸",
        "Description": "双模型实时预测系统（DQ200 & AVS）"
    }

if __name__ == '__main__':
    uvicorn.run(app,port=8001)