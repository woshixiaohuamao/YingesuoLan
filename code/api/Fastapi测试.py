import requests
import json

# 输入数据样例
input_data = {
  "DQ200系统压力": 6.07,
  "AVS系统压力": 6.4,
  "DQ200累积流量": 329084448,
  "AVS累积流量": 54116832,
  "DQ200差分值": 0,
  "AVS差分值": 8,
  "year": 2025,
  "month": 5,
  "day": 1,
  "weekday": 3,
  "hour": 0,
  "minute": 0,
  "second": 5,
  "is_weekend": 0,
  "quarter": 2
}


# 定义API地址和输入数据
API_URL = "http://localhost:8000/predict/"


# 发送POST请求
response = requests.post(API_URL, json=input_data)

# 解析响应
if response.status_code == 200:
    print("预测结果:", response.json()["prediction"])
else:
    print("请求失败:", response.json()["detail"])