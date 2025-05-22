import requests
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import uvicorn

def fetch_realtime_data():
    """
    调用数据库API获取实时数据，并转换为模型输入格式
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    
    params = {
        "startTime": start_time.isoformat(timespec='milliseconds'),
        "endTime": end_time.isoformat(timespec='milliseconds'),
        "interval": 60000,   # 1分钟间隔
        "valueonly": 0,
        "decimal": 2,
        "names": "DLDZ_DQ200_LLJ01_FQ01.PV"  # 只需要获取用于计算滞后的变量
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

    input_data = {}
    
    # 处理滞后特征
    var_name = "DLDZ_DQ200_LLJ01_FQ01.PV"
    item = next((item for item in data['items'] if item['name'] == var_name), None)
    if not item:
        raise ValueError(f"未找到变量: {var_name}")
    
    # 获取原始数据值并验证
    raw_values = [v['val'] for v in item['vals']]
    if len(raw_values) < 8:
        raise ValueError(f"需要至少8个数据点，当前只有{len(raw_values)}个")
    
    # 计算差分序列（最新数据在列表最后）
    recent_values = raw_values[-8:]  # 取最后8个数据点
    diffs = [round(recent_values[i] - recent_values[i-1]) for i in range(1, 8)]
    
    # 填充滞后特征（Lag_1为最新差分）
    for i in range(7):
        input_data[f"Lag_{i+1}"] = diffs[-(i+1)]  # 倒序取最后7个差分
    
    # 生成时间特征
    current_time = datetime.now()
    input_data.update({
        "year": current_time.year,
        "month": current_time.month,
        "day": current_time.day,
        "weekday": current_time.weekday(),
        "hour": current_time.hour,
        "minute": current_time.minute,
        "second": current_time.second,
        "is_weekend": 1 if current_time.weekday() in [5, 6] else 0,
        "quarter": (current_time.month - 1) // 3 + 1
    })
    
    return input_data

app = FastAPI()
model = joblib.load("xgb_dq200_LAG.pkl")

# 确保特征顺序与模型一致
FEATURE_ORDER = [
    'year', 'month', 'day', 'weekday', 'hour', 'minute', 'second',
    'is_weekend', 'quarter', 'Lag_1', 'Lag_2', 'Lag_3', 
    'Lag_4', 'Lag_5', 'Lag_6', 'Lag_7'
]

@app.post("/predict_realtime/")
async def predict_realtime():
    try:
        input_data = fetch_realtime_data()
        # 按预定顺序构建特征数组
        feature_array = np.array([input_data[key] for key in FEATURE_ORDER]).reshape(1, -1)
        prediction = model.predict(feature_array)
        return {"prediction": prediction.tolist()[0], "input_data": input_data}
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
        "Version": "1.0.0",
        "Author": "赵泽宸",
        "Description": "基于XGBoost的瞬时用气量预测模型"
    }

#运行
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8001)