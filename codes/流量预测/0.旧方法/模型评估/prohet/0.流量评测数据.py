import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime

model_path = ''
file_path = ''
# 1. 加载模型
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# 2. 加载真实数据集
df = pd.read_excel(file_path)
df['ds'] = pd.to_datetime(df['timestamp'])
df['y'] = df['瞬时流量']

# 3. 准备测试数据
test_df = df[['ds']].copy()

# 4. 使用模型进行预测
forecast = model.predict(test_df)

# 5. 合并预测结果和真实值
results = pd.merge(df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

# 6. 计算评估指标
r2 = r2_score(results['y'], results['yhat'])
mse = mean_squared_error(results['y'], results['yhat'])
mae = mean_absolute_error(results['y'], results['yhat'])
rmse = np.sqrt(mse)

# 7. 打印评估结果
print("="*50)
print("Prophet 模型性能评估")
print("="*50)
print(f"R² (决定系数): {r2:.4f}")
print(f"MSE (均方误差): {mse:.4f}")
print(f"RMSE (均方根误差): {rmse:.4f}")
print(f"MAE (平均绝对误差): {mae:.4f}")
print("="*50)

# 8. 可视化真实值和预测值对比
plt.rcParams['font.sans-serif'] = ['SimSun']      
plt.figure(figsize=(14, 8))

# 主图：真实值与预测值对比
plt.subplot(2, 1, 1)
plt.plot(results['ds'], results['y'], 'b-', label='真实值', alpha=0.8)
plt.plot(results['ds'], results['yhat'], 'r-', label='预测值', alpha=0.8)
plt.fill_between(results['ds'], results['yhat_lower'], results['yhat_upper'], 
                 color='gray', alpha=0.2, label='置信区间')
plt.title('Prophet模型预测结果 vs 真实值')
plt.xlabel('日期')
plt.ylabel('值')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)

# 残差图
plt.subplot(2, 1, 2)
residuals = results['y'] - results['yhat']
plt.plot(results['ds'], residuals, 'g-', alpha=0.7)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)
plt.fill_between(results['ds'], -rmse, rmse, color='gray', alpha=0.1)
plt.title('预测残差 (真实值 - 预测值)')
plt.xlabel('日期')
plt.ylabel('残差')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('./模型评估/prophet_performance.png', dpi=300)
plt.show()

# 9. 添加残差列并保存评估结果
results['residual'] = residuals
results.to_csv('./模型评估/prohet/prophet_evaluation_results.csv', index=False)

# 10. 打印性能最好的和最差的日期
best_pred = results.loc[results['residual'].abs().idxmin()]
worst_pred = results.loc[results['residual'].abs().idxmax()]

print("\n最佳预测日期:")
print(f"日期: {best_pred['ds'].strftime('%Y-%m-%d')}")
print(f"真实值: {best_pred['y']:.2f}, 预测值: {best_pred['yhat']:.2f}, 误差: {best_pred['residual']:.2f}")

print("\n最差预测日期:")
print(f"日期: {worst_pred['ds'].strftime('%Y-%m-%d')}")
print(f"真实值: {worst_pred['y']:.2f}, 预测值: {worst_pred['yhat']:.2f}, 误差: {worst_pred['residual']:.2f}")

# 11. 季节性分析
if hasattr(model, 'plot_components'):
    fig = model.plot_components(forecast)
    fig.savefig('./模型评估/prophet_components.png', dpi=300)