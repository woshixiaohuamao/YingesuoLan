from fastapi import FastAPI
import os
from sarima_predictor import predict_sarima  
from models import train_sarima_and_forecast 
import uvicorn
from data_utils import fetch_recent_data
from datetime import datetime
import pandas as pd
from sunday_data import fetch_sunday_data

app = FastAPI()

# 固定参数值
DEFAULT_DAYS = 2       
DEFAULT_STEPS = 48     
CONFIDENCE_LEVEL = 0.95 

@app.get("/")
def read_root():
    return {"message": "欢迎使用预测API！调用 /predict_flow 进行流量预测，/predict_pressure 进行压力预测"}

@app.get("/predict_flow")
def predict_flow():
    current_date = datetime.now()
    is_saturday = current_date.weekday() == 5  
    model_type = "sunday" if is_saturday else "workday"
    
    if is_saturday:
        fetch_sunday_data(weeks=4)
        file_path = r''
        forecast_df = predict_sarima(file_path)         
        forecast_df["model_type"] = model_type
        return forecast_df.to_dict(orient='records')
    
    return predict_workday(model_type)

def predict_workday(model_type: str):
    df = fetch_recent_data(DEFAULT_DAYS)
    forecast_df = train_sarima_and_forecast(
        df,
        target_column="瞬时流量",
        steps=DEFAULT_STEPS,
        confidence_level=CONFIDENCE_LEVEL
    )
    forecast_df["model_type"] = model_type
    
    return forecast_df.to_dict(orient='records')

@app.get("/predict_pressure")
def predict_pressure():
    df = fetch_recent_data(DEFAULT_DAYS)    
    forecast_df = train_sarima_and_forecast(
        df,
        target_column="总压力",
        steps=DEFAULT_STEPS,
        confidence_level=CONFIDENCE_LEVEL
    )    
    return forecast_df.to_dict(orient='records')


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)