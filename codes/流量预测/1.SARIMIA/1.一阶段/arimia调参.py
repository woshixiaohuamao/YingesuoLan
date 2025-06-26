import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

# 1. 准备数据

df = pd.read_excel()
series = df['瞬时流量']

# 2. 划分训练集和测试集
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# 3. 自动寻找最优ARIMA参数
auto_model = auto_arima(
    train,
    start_p=1,    # p的最小值
    start_q=1,    # q的最小值
    max_p=5,      # p的最大值
    max_q=5,      # q的最大值
    d=None,       # 自动检测最优d
    seasonal=False, # 非季节性数据（如果是季节性数据改为True并设置m参数）
    stepwise=True, # 使用逐步搜索（更快）
    trace=True,    # 打印搜索过程
    error_action='ignore',
    suppress_warnings=True
)

print("最优模型参数:", auto_model.order)  # 输出(p,d,q)

# 4. 用最优参数训练模型
model = auto_model  # auto_arima已自动训练
forecast_steps = len(test)
forecast = model.predict(n_periods=forecast_steps)

# 5. 评估
mse = mean_squared_error(test, forecast)
print(f'Mean Squared Error: {mse:.4f}')

# 6. 可视化
plt.figure(figsize=(12,6))
plt.plot(train.index[-50:], train.values[-50:], label='Train')
plt.plot(test.index, test.values, label='Actual')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.title('ARIMA Auto-tuning Forecasting')
plt.legend()
plt.show()