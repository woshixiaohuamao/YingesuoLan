import os
from sarima_predictor import predict_sarima  # 星期日模型
from models import train_sarima_and_forecast  # 工作日模型
import uvicorn
from data_utils import fetch_recent_data
from datetime import datetime
import pandas as pd
from sunday_data import fetch_sunday_data
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List
from sarima_rf_predictor import predict_by_mode
from postprocessing_utils import add_prediction_bounds, interpolate_to_minute, convert_to_json_format

app = FastAPI(title="Flow Prediction API")

# 固定参数值
DEFAULT_DAYS = 2        # 固定使用最近2天的数据
DEFAULT_STEPS = 48       # 固定预测48步
CONFIDENCE_LEVEL = 0.95  # 固定置信水平95%

@app.get("/")
def read_root():
    return {"message": "欢迎使用预测API！调用 /predict_flow 进行流量预测，/predict_pressure 进行压力预测 /predict 流量预测SARIMIA+RF"}

class PredictionResponseItem(BaseModel):
    timestamp: str
    forecast: float
    lower_bound: float
    upper_bound: float
    metric: str
    model_type: str

@app.get("/predict", response_model=List[PredictionResponseItem])
def predict():
    """
    预测接口：默认执行'预测今天'，无需传参。
    """
    try:
        # 1. 默认模式为'T'（预测今天）
        mode = 'T'
        rf_preds, sarima_forecast, df_original = predict_by_mode(mode)

        # 2. 插值为1分钟
        df_minute = interpolate_to_minute(rf_preds)

        # 3. 添加上下限
        lower_bounds, upper_bounds = add_prediction_bounds(df_minute['value'].values)

        # 4. 转为JSON格式
        json_result = convert_to_json_format(
            df_minute,
            lower_bounds,
            upper_bounds,
            input_df=df_original,
            metric="瞬时流量"
        )

        return json_result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/predict_flow")
def predict_flow():
    """
    流量预测端点 - 无需参数
    - 周六: 使用周日模型预测流量
    - 其他日期: 使用工作日模型预测流量
    """
    current_date = datetime.now()
    is_saturday = current_date.weekday() == 5  # 0=周一, 5=周六
    model_type = "sunday" if is_saturday else "workday"
    
    if is_saturday:
        # 周日预测模型
        fetch_sunday_data(weeks=6)
        file_path = r''
        forecast_df = predict_sarima(file_path)  # 返回DataFrame        
        # 添加模型类型字段
        forecast_df["model_type"] = model_type
        return forecast_df.to_dict(orient='records')
    
    # 工作日预测
    return predict_workday(model_type)

def predict_workday(model_type: str):
    """
    工作日流量预测 - 使用固定参数
    """
    df = fetch_recent_data(DEFAULT_DAYS)
    # 使用固定参数进行预测
    forecast_df = train_sarima_and_forecast(
        df,
        target_column="瞬时流量",
        steps=DEFAULT_STEPS,
        confidence_level=CONFIDENCE_LEVEL
    )
    # 添加模型类型字段
    forecast_df["model_type"] = model_type
    
    return forecast_df.to_dict(orient='records')

@app.get("/predict_pressure")
def predict_pressure():
    """
    压力预测端点 - 无需参数
    """
    df = fetch_recent_data(DEFAULT_DAYS)    
    # 使用固定参数进行预测
    forecast_df = train_sarima_and_forecast(
        df,
        target_column="总压力",
        steps=DEFAULT_STEPS,
        confidence_level=CONFIDENCE_LEVEL
    )    
    return forecast_df.to_dict(orient='records')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=58888)
    #uvicorn.run(app, host="localhost", port=58888)