import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import json

# 忽略警告信息
warnings.filterwarnings('ignore')

# 加载数据
file_name = ''
df = pd.read_excel(file_name)

# 选择目标变量
target = 'Actual'

# 处理缺失值和异常值
df_clean = df.dropna(subset=[target]).reset_index(drop=True)
df_clean[target] = df_clean[target].replace([np.inf, -np.inf], np.nan).fillna(method='ffill')

# 准备Prophet需要的数据格式（仅保留 ds 和 y）
prophet_df = df_clean[['date', target]].copy()
prophet_df.columns = ['ds', 'y']  # Prophet要求的时间列名和目标列名

# 确保 ds 列为 datetime 类型
prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

# 时间序列分割（按时间顺序）
points_per_day = 24 * 60 // 30  # 30分钟间隔，一天48个点
val_days = 7  # 验证集取7天
val_size = val_days * points_per_day  # 验证集总点数

total_size = len(prophet_df)
test_start_idx = total_size - points_per_day  # 测试集起始位置
val_start_idx = test_start_idx - val_size    # 验证集起始位置

# 划分数据集
train_df = prophet_df.iloc[:val_start_idx]        # 训练集
val_df = prophet_df.iloc[val_start_idx:test_start_idx]  # 验证集
test_df = prophet_df.iloc[test_start_idx:]         # 测试集

print(f"数据集大小: 总样本{total_size}, 训练集{len(train_df)}, 验证集{len(val_df)}, 测试集{len(test_df)}")

# 创建Prophet模型（不添加任何回归量）
model = Prophet(
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=False,
    changepoint_prior_scale=0.01,
    seasonality_prior_scale=5.0
)

# 训练模型
model.fit(train_df)

# 创建未来数据框（仅保留 ds 列）
future = model.make_future_dataframe(periods=len(val_df) + len(test_df), freq='30Min')  # 30分钟间隔
future = future[future['ds'] >= train_df['ds'].max()]  
print("future 缺失值处理后大小:", len(future))

# 进行预测
forecast = model.predict(future)

# 合并预测结果
results = pd.merge(
    prophet_df, 
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
    on='ds', 
    how='left'
)
results['yhat'] = results['yhat'].fillna(method='ffill')

# 分割结果回各数据集
train_results = results.iloc[:len(train_df)]
val_results = results.iloc[len(train_df):len(train_df)+len(val_df)]
test_results = results.iloc[len(train_df)+len(val_df):]

# 评估函数
def evaluate_model(df, name):
    """评估模型并返回指标"""
    y_true = df['y']
    y_pred = df['yhat']
    
    # 检查 NaN
    if y_true.isnull().any() or y_pred.isnull().any():
        print(f"警告：{name}集存在 NaN，跳过评估！")
        return None, 0.0, 0.0, 0.0, 0.0  
    
    mse = mean_squared_error(y_true, y_pred) 
    rmse = np.sqrt(mse)  
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-5, None))) * 100
    
    print(f"\n{name}集评估结果:")
    print(f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}%")
    
    return y_pred, mse, rmse, mae, mape  

# 在三个数据集上评估
_, train_mse, train_rmse, train_mae, train_mape = evaluate_model(train_results, "训练")
_, val_mse, val_rmse, val_mae, val_mape = evaluate_model(val_results, "验证")
test_pred, test_mse, test_rmse, test_mae, test_mape = evaluate_model(test_results, "测试")

# 可视化结果
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为"宋体"

# 1. 整体预测图
fig1 = model.plot(forecast)
plt.title('Prophet整体预测')
plt.xlabel('日期')
plt.ylabel('值')
plt.savefig('prophet_forecast.png', dpi=300)
print("整体预测图已保存为: prophet_forecast.png")

# 2. 分量图（趋势和季节性）
fig2 = model.plot_components(forecast)
plt.savefig('prophet_components.png', dpi=300)
print("分量图已保存为: prophet_components.png")

# 3. 测试集预测对比（仅 Prophet）
plt.figure(figsize=(15, 6))
test_sample = test_results.iloc[:100]  # 取前100个样本展示

# 真实值
plt.plot(test_sample['ds'], test_sample['y'], 'b-', label='真实值', linewidth=2)

# Prophet 预测值
plt.plot(
    test_sample['ds'], 
    test_sample['yhat'], 
    'r--', 
    label='Prophet 预测', 
    linewidth=2
)

# Prophet 置信区间
plt.fill_between(
    test_sample['ds'], 
    test_sample['yhat_lower'], 
    test_sample['yhat_upper'],
    color='gray', alpha=0.2, label='Prophet 置信区间'
)

# 图表设置
plt.legend()
plt.title('测试集预测效果对比（仅 Prophet）')
plt.xlabel('时间')
plt.ylabel('值')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('test_comparison_prophet.png', dpi=300)
print("测试集预测对比图（仅 Prophet）已保存为: test_comparison_prophet.png")

# 4. 残差分析
plt.figure(figsize=(12, 6))
residuals = test_results['y'] - test_results['yhat']
plt.scatter(test_results['yhat'], residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('预测值与残差关系')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.savefig('residual_analysis.png', dpi=300)
print("残差分析图已保存为: residual_analysis.png")

# 最终评估报告
print("\n" + "="*50)
print("Prophet模型评估报告（仅使用历史值）")
print("="*50)

# 训练集
if train_mse is not None:
    print(f"训练集: MSE={train_mse:.4f} | RMSE={train_rmse:.4f} | MAE={train_mae:.4f} | MAPE={train_mape:.2f}%")
else:
    print("训练集: 评估失败（存在 NaN）")

# 验证集
if val_mse is not None:
    print(f"验证集: MSE={val_mse:.4f} | RMSE={val_rmse:.4f} | MAE={val_mae:.4f} | MAPE={val_mape:.2f}%")
else:
    print("验证集: 评估失败（存在 NaN）")

# 测试集
if test_mse is not None:
    print(f"测试集: MSE={test_mse:.4f} | RMSE={test_rmse:.4f} | MAE={test_mae:.4f} | MAPE={test_mape:.2f}%")
else:
    print("测试集: 评估失败（存在 NaN）")

# 保存模型
with open('prophet_model_no_features.json', 'w') as fout:
    json.dump(model_to_json(model), fout)
print("模型已保存为: prophet_model_no_features.json")