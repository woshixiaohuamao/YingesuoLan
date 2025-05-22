import pandas as pd
import joblib

def predict_values(excel_path, model_path, num_samples):
    # 从Excel文件加载数据
    df = pd.read_excel(excel_path)
    
    # 确保按指定顺序选取数据，并限制为num_samples行
    df_sampled = df.iloc[:num_samples]
    
    # 分离特征与目标变量（假设目标变量名为 'target'，请根据实际情况调整）
    X = df_sampled.drop(columns=['时间',target])
    y_true = df_sampled[target]
    
    # 加载模型
    xgb_reg_loaded = joblib.load(model_path)
    
    # 预测
    y_pred = xgb_reg_loaded.predict(X)
    
    # 输出真实值和预测值
    for true, pred in zip(y_true, y_pred):
        print(f"真实值: {true}, 预测值: {pred}")

# 使用示例
target = 'DQ200总流量'
excel_file_path = r'D:\Aa中工互联\工作安排\英格索兰\data\预测\分钟数据-清洗-时间(节选).xlsx'  # 替换为你的Excel文件路径
model_file_path = r'D:\Aa中工互联\工作安排\英格索兰\Fastapi\xgb_dq200_part.pkl'  # 模型文件路径
number_of_samples = 10  # 指定要使用的数据条数

predict_values(excel_file_path, model_file_path, number_of_samples)