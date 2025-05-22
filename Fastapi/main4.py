import requests
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import uvicorn

def fetch_realtime_data():
    """
    调用数据库API获取实时数据，生成两个模型所需的输入特征
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    
    params = {
        "startTime": start_time.isoformat(timespec='milliseconds'),
        "endTime": end_time.isoformat(timespec='milliseconds'),
        "interval": 60000,   # 1分钟间隔
        "valueonly": 0,
        "decimal": 2,
        "names": "DLDZ_DQ200_LLJ01_FQ01.PV,DLDZ_AVS_LLJ01_FQ01.PV"  # 同时获取两个变量
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

    # 获取两个变量数据
    var_dq200 = "DLDZ_DQ200_LLJ01_FQ01.PV"
    var_avs = "DLDZ_AVS_LLJ01_FQ01.PV"
    
    item_dq200 = next((item for item in data['items'] if item['name'] == var_dq200), None)
    item_avs = next((item for item in data['items'] if item['name'] == var_avs), None)
    
    if not item_dq200 or not item_avs:
        raise ValueError("未找到其中一个或多个变量")

    # 处理DQ200滞后特征
    raw_dq200 = [v['val'] for v in item_dq200['vals']]
    if len(raw_dq200) < 8:
        raise ValueError(f"DQ200需要至少8个数据点，当前只有{len(raw_dq200)}个")
    diffs_dq200 = [round(raw_dq200[i] - raw_dq200[i-1]) for i in range(-7, 0)]  # 取最后7个差分
    
    # 处理AVS滞后特征
    raw_avs = [v['val'] for v in item_avs['vals']]
    if len(raw_avs) < 8:
        raise ValueError(f"AVS需要至少8个数据点，当前只有{len(raw_avs)}个")
    diffs_avs = [round(raw_avs[i] - raw_avs[i-1]) for i in range(-7, 0)]

    # 公共时间特征
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
    
    # 构建两个输入字典
    input_dq200 = {**time_features, **{f"Lag_{i+1}": v for i, v in enumerate(diffs_dq200)}}
    input_avs = {**time_features, **{f"Lag_{i+1}": v for i, v in enumerate(diffs_avs)}}
    
    return input_dq200, input_avs

app = FastAPI()

# 加载两个模型
model_dq200 = joblib.load("./Fastapi/xgb_dq200_LAG.pkl")
model_avs = joblib.load("./Fastapi/xgb_avs_LAG.pkl")

# 特征顺序（两个模型结构相同）
FEATURE_ORDER = [
    'year', 'month', 'day', 'weekday', 'hour', 'minute', 'second',
    'is_weekend', 'quarter', 'Lag_1', 'Lag_2', 'Lag_3', 
    'Lag_4', 'Lag_5', 'Lag_6', 'Lag_7'
]

@app.post("/predict_realtime/")
async def predict_realtime():
    try:
        # 获取两个输入数据
        input_dq200, input_avs = fetch_realtime_data()
        
        # DQ200模型预测
        features_dq200 = np.array([input_dq200[key] for key in FEATURE_ORDER]).reshape(1, -1)
        pred_dq200 = model_dq200.predict(features_dq200)[0]
        
        # AVS模型预测
        features_avs = np.array([input_avs[key] for key in FEATURE_ORDER]).reshape(1, -1)
        pred_avs = model_avs.predict(features_avs)[0]
        
        return {
            "predictions": {
                "DQ200_model": float(pred_dq200),
                "AVS_model": float(pred_avs)
            },
            "input_features": {
                "DQ200": input_dq200,
                "AVS": input_avs
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