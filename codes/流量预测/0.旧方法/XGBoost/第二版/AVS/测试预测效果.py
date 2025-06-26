import pandas as pd
import joblib

def predict_values(excel_path, model_path, num_samples):
    # 从Excel文件加载数据
    df = pd.read_excel(excel_path)
    
    # 确保按指定顺序选取数据，并限制为num_samples行
    df_sampled = df.iloc[:num_samples]
    
    # 分离特征与目标变量
    X = df_sampled.drop(columns=['时间',target])
    y_true = df_sampled[target]
    
    # 加载模型
    xgb_reg_loaded = joblib.load(model_path)
    
    # 预测
    y_pred = xgb_reg_loaded.predict(X)
    
    # 输出真实值和预测值
    for true, pred in zip(y_true, y_pred):
        print(f"真实值: {true}, 预测值: {pred}")


target = 'AVS总流量'
excel_file_path = ''  
model_file_path = ''  
number_of_samples = 10  # 指定要使用的数据条数

predict_values(excel_file_path, model_file_path, number_of_samples)