from prophet import Prophet
import pandas as pd
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ParameterGrid


# Step 1: 准备数据
df = pd.read_excel() 

df_prophet = df.rename(columns={'timestamp': 'ds', '瞬时流量': 'y'})

df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])


df_prophet = df_prophet[['ds', 'y']]

# 示例数据划分（取最后 100 个时间点作为测试集）
train_size = len(df_prophet) - 100
train_df = df_prophet.iloc[:train_size]
test_df = df_prophet.iloc[train_size:]

# 定义参数网格
param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1],
    'seasonality_prior_scale': [0.1, 1.0, 10.0],
    'holidays_prior_scale': [1.0, 10.0]
}

# 存储结果
best_score = float('inf')
best_model = None
results = []

# 网格搜索
for params in ParameterGrid(param_grid):  
    model = Prophet(
        changepoint_prior_scale=params['changepoint_prior_scale'],
        seasonality_prior_scale=params['seasonality_prior_scale'],
        holidays_prior_scale=params['holidays_prior_scale']
    )
    model.add_country_holidays(country_name='China')
    model.fit(train_df)

    future = model.make_future_dataframe(
        periods=len(test_df), freq='T', include_history=False
    )
    forecast = model.predict(future)

    # 提取预测值与真实值对齐
    forecast = forecast.set_index('ds').reindex(test_df['ds'])
    y_true = test_df.set_index('ds')['y'].values
    y_pred = forecast['yhat'].values

    # 计算评估指标
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    results.append({
        **params,
        'mae': mae,
        'rmse': rmse
    })

    # 更新最佳模型
    if mae < best_score:
        best_score = mae
        best_model = model

# 打印最佳参数和得分
print("Best Parameters:", best_model.__dict__['params']) 
print("Best MAE:", best_score)

# 可视化预测结果（真实值 vs 预测值）
plt.figure(figsize=(12, 6))
plt.plot(test_df['ds'], test_df['y'], label='True', color='blue')
plt.plot(test_df['ds'], forecast['yhat'], label='Predicted', color='red', linestyle='--')
plt.fill_between(test_df['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='pink', alpha=0.3)
plt.legend()
plt.title('True vs Predicted (Test Set)')
plt.show()

# 可视化调参结果
results_df = pd.DataFrame(results)
results_df.sort_values('mae', inplace=True)
plt.figure(figsize=(10, 6))
plt.bar(range(len(results_df)), results_df['mae'], tick_label=range(len(results_df)))
plt.title('MAE for Different Parameter Combinations')
plt.xlabel('Parameter Index')
plt.ylabel('MAE')
plt.show()

# 保存最佳模型
import pickle
with open('best_prophet_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)