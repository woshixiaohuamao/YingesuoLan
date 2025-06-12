import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error


FILE_PATH = r'D:\Aa中工互联\工作安排\英格索兰\code\预测\lstm\data\data_0601-0609.xlsx'
TARGET_COLUMN = '瞬时流量'
SAMPLING_INTERVAL = 10
DATE_COLUMN = 'timestamp'

# 读取数据（假设文件为制表符分隔，时间戳列为 index）
df = pd.read_excel(FILE_PATH)
# 检查并转换日期列
if DATE_COLUMN not in df.columns or TARGET_COLUMN not in df.columns:
    raise ValueError(f"数据缺少必要的'{DATE_COLUMN}'或'{TARGET_COLUMN}'列")

# 转换日期并设置为索引
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df.set_index(DATE_COLUMN, inplace=True)
# 重采样为指定间隔
data = df[TARGET_COLUMN].resample(f'{SAMPLING_INTERVAL}T').mean()
# 计算每天有多少个数据点
points_per_day = 24 * 60 // SAMPLING_INTERVAL

# 划分训练集和测试集（示例：取最后12个样本作为测试集）
test_size = points_per_day  # 1天的数据点数量
test_data = data[-test_size:]
train_data = data[:-test_size]

# 定义并训练 SARIMA 模型（可根据你的数据调整参数）
model = SARIMAX(train_data, order=(0, 0, 0), seasonal_order=(0, 2, 2, points_per_day))
results = model.fit(disp=False)

# 预测测试集
preds = results.get_forecast(steps=points_per_day).predicted_mean

# 评估模型
mse = mean_squared_error(test_data, preds)
print(f"Test MSE: {mse:.2f}")