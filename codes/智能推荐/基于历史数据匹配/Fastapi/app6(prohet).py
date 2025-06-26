import torch
from fastapi import FastAPI, HTTPException, Query
import numpy as np
from joblib import load
from datetime import datetime, timedelta
import json
from pydantic import BaseModel
from device_recommender import DeviceRecommender
from pathlib import Path
from data_fetcher import fetch_realtime_data, fetch_realtime_data_for_P
from prophet_forecaster import get_prophet_forecast  

# 模型结构定义
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=60):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# 更新模型配置：添加误差统计路径
MODEL_CONFIGS = {
    "L": {
        "model_path": "lstm_144_60.pth",
        "scaler_path": "scaler_model.joblib",
        "stats_path": "error_stats_L.json",
        "data_names": ["DLDZ_DQ200_LLJ01_FQ01.PV", "DLDZ_AVS_LLJ01_FQ01.PV"]
    },
    "P": {
        "model_path": "lstm_144_60_pressure.pth",
        "scaler_path": "scaler_model_pressure.joblib",
        "stats_path": "error_stats_P.json",
        "data_names": ["DLDZ_DQ200_SYSTEM_PI05.PV", "DLDZ_AVS_SYSTEM_PI05.PV"]
    }
}

# 计算误差上下限
def load_error_stats(stats_path):
    """加载误差统计（带异常处理）"""
    try:
        with open(stats_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Error stats file {stats_path} not found")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {stats_path}")

# 动态加载模型和归一化器
def load_model_and_scaler(model_type):
    """修改后的加载函数"""
    config = MODEL_CONFIGS.get(model_type)
    if not config:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # 加载模型
    model = torch.load(config["model_path"], map_location="cpu", weights_only=False)
    model.eval()
    
    # 加载归一化器
    scaler = load(config["scaler_path"])
    
    return model, scaler, config 

# 预测函数
def predict_with_model(model_type):
    """修改后的预测函数"""
    model, scaler, config = load_model_and_scaler(model_type)
    
    # 获取输入数据
    if model_type == "P":
        diff_sum = fetch_realtime_data_for_P()
    else:
        diff_sum = fetch_realtime_data()
    
    if not diff_sum or len(diff_sum) < 144:
        raise ValueError("获取的数据不足144个点")
    
    # 数据预处理
    input_data = np.array(diff_sum).reshape(-1, 1)
    scaled_input = scaler.transform(input_data)
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0)
    
    # 执行预测
    with torch.no_grad():
        predictions = model(input_tensor).numpy()
    
    # 逆归一化
    predictions_inv = scaler.inverse_transform(predictions)
    timestamps = [datetime.now() + timedelta(minutes=i+1) for i in range(60)]
    values = predictions_inv.tolist()[0]
    
    # 加载并应用误差统计
    error_stats = load_error_stats(config["stats_path"])
    
    # 计算上下限（使用标准差法）
    upper = [v + 1.96 * error_stats["std_dev"] for v in values]
    lower = [v - 1.96 * error_stats["std_dev"] for v in values]
    
    # 构建返回结果
    return {
        "model_type": model_type,
        "prediction_start": datetime.now().isoformat(),
        "prediction_interval": "PT1M",
        "statistical_method": "95%_std_dev",
        "predictions": [
            {
                "timestamp": ts.isoformat(),
                "value": float(value),
                "upper_bound": float(upper[i]),
                "lower_bound": float(lower[i])
            }
            for i, (ts, value) in enumerate(zip(timestamps, values))
        ]
    }

# FastAPI 应用
app = FastAPI(title="Multi-Model Prediction API", version="3.0")

@app.post("/predict")
async def predict(model_type: str = Query(..., description="Model type: 'L' or 'P'")):
    try:
        result = predict_with_model(model_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 初始化推荐器（在相同目录）
# recommender = DeviceRecommender(str(Path(__file__).parent / "hourly_config.json"))
recommender = DeviceRecommender(str(Path(__file__).parent / "hourly_config_new.json"))
# 直接使用导入的Prophet函数
@app.get("/prophet_forecast")
async def prophet_forecast_endpoint(days: int = Query(7, description="预测天数", ge=1, le=365)):
    return get_prophet_forecast(days)

class RequestData(BaseModel):
    hour: int
    target_gas: float
    strategy: str = "efficient"

@app.get("/recommend")
async def get_recommendation(
    hour: int = Query(..., title="小时", ge=0, le=23),
    target_gas: float = Query(..., title="目标流量", gt=0),
    strategy: str = Query("efficient", title="策略", regex="^(accurate|efficient|safe)$")
):
    try:
        result = recommender.recommend(hour, target_gas, strategy)
        return result or {"error": "无可用配置"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) 
      
@app.get("/")
def read_root():
    return {"Health_check": "OK"}

@app.get("/info")
def read_info():
    return {
        "Name": "英格索兰项目-大连大众",
        "Version": "5.0.0",
        "Author": "赵泽宸",
        "Description": "预测模型 and 优化调度 and 多尺度预测"
    }

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="localhost", port=58888)
    
    # uvicorn.run(app, host="10.2.12.76", port=58888)
    uvicorn.run(app, host="0.0.0.0", port=58888)