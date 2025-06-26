from pydantic import BaseModel
from datetime import datetime
from typing import List

class ForecastPoint(BaseModel):
    """单个预测点的数据结构"""
    timestamp: datetime
    forecast: float
    lower_bound: float
    upper_bound: float
    metric: str

class ForecastResult(BaseModel):
    """预测结果数据结构"""
    status: str
    message: str
    forecast_days: float
    data: List[ForecastPoint]