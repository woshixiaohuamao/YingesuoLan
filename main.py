from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# 加载模型
model = joblib.load("xgb_reg_model.pkl")

@app.post("/predict/")
async def predict(input_data: dict):
    try:
        # 假设输入数据是标准化后的特征值，并且以字典形式传入
        feature_array = np.array(list(input_data.values())).reshape(1, -1)
        prediction = model.predict(feature_array)
        return {"prediction": prediction.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 根路径
@app.get("/")
def read_root():
    return {"Hello": "World"}
# 启动服务
# uvicorn main:app --reload