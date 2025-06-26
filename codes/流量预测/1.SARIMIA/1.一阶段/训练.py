import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error


FILE_PATH = ''
TARGET_COLUMN = '瞬时流量'
SAMPLING_INTERVAL = 30
DATE_COLUMN = 'timestamp'


df = pd.read_excel(FILE_PATH)
if DATE_COLUMN not in df.columns or TARGET_COLUMN not in df.columns:
    raise ValueError(f"数据缺少必要的'{DATE_COLUMN}'或'{TARGET_COLUMN}'列")


df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df.set_index(DATE_COLUMN, inplace=True)
data = df[TARGET_COLUMN].resample(f'{SAMPLING_INTERVAL}T').mean()
points_per_day = 24 * 60 // SAMPLING_INTERVAL


test_size = points_per_day 
test_data = data[-test_size:]
train_data = data[:-test_size]


model = SARIMAX(train_data, order=(0, 2, 1), seasonal_order=(1, 1, 0, points_per_day))
results = model.fit(disp=False)

preds = results.get_forecast(steps=points_per_day).predicted_mean

mse = mean_squared_error(test_data, preds)
print(f"Test MSE: {mse:.2f}")