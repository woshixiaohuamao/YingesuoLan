import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import joblib
import warnings
from datetime import datetime, timedelta
from data_utils import fetch_recent_data, fetch_today_data

warnings.filterwarnings("ignore")  # 忽略警告信息

MODEL_PATH = r''

REQUIRED_FEATURES = [
    'hour', 'day_of_week', 'day_of_month', 'month', 'weekend',
    'lag_1', 'lag_2', 'lag_3', 'lag_12', 'lag_24', 'lag_48',
    'rolling_6h_mean', 'rolling_12h_std'
]

def load_true_data_for_date(date_str):
    """
    严格加载指定日期的完整真实数据，不完整则抛错
    """
    df = fetch_recent_data(3)  # 获取最近3天数据
    if df.empty:
        raise ValueError("API未返回数据")

    df['date_only'] = df['timestamp'].dt.date.astype(str)
    day_data = df[df['date_only'] == date_str].copy()
    if day_data.empty:
        raise ValueError(f"没有找到 {date_str} 的真实数据")

    points_per_day = 48  # 30分钟间隔一天48个点
    if len(day_data) < points_per_day:
        raise ValueError(f"{date_str} 数据不完整")

    day_data.drop(columns=['date_only'], inplace=True)
    return day_data


def check_data_validity_for_today_prediction(df):
    """
    当天预测专用校验，放宽昨天和前天数据完整性要求。
    只要数据存在且点数>一定阈值（比如20个点）即可，不强制48点全部。
    """
    dates = df.index.date
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    day_before_yesterday = today - timedelta(days=2)

    if (yesterday not in dates) or (day_before_yesterday not in dates):
        raise RuntimeError("缺少昨天或前天数据，无法预测当天")

    count_yesterday = np.sum(dates == yesterday)
    count_day_before = np.sum(dates == day_before_yesterday)

    min_points_threshold = 20  # 可以根据需求调整阈值

    if count_yesterday < min_points_threshold:
        raise RuntimeError(f"昨天 {yesterday} 数据点数过少 ({count_yesterday})，无法预测当天")

    if count_day_before < min_points_threshold:
        raise RuntimeError(f"前天 {day_before_yesterday} 数据点数过少 ({count_day_before})，无法预测当天")


def check_data_validity_for_prediction(mode, df, points_per_day=24):
    """
    mode: 'T' (预测今天) 或 'N' (预测明天)
    df: DataFrame，已按时间处理好
    points_per_day: 每天的数据点数，默认24（1小时粒度）
    """
    dates = df.index.date
    latest_data_date = df.index[-1].date()
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    day_before_yesterday = today - timedelta(days=2)

    if mode == 'N':  # 预测明天
        # 需要今天+昨天的完整数据
        if (today not in dates) or (yesterday not in dates):
            raise RuntimeError("缺少昨天或今天数据，无法预测明天。")
        if np.sum(dates == today) < points_per_day:
            raise RuntimeError(f"今天 {today} 数据不完整 ({np.sum(dates == today)}/{points_per_day})，无法预测明天。")
        if np.sum(dates == yesterday) < points_per_day:
            raise RuntimeError(f"昨天 {yesterday} 数据不完整 ({np.sum(dates == yesterday)}/{points_per_day})，无法预测明天。")

    elif mode == 'T':  # 预测今天
        # 需要昨天+前天的完整数据
        if (yesterday not in dates) or (day_before_yesterday not in dates):
            raise RuntimeError("缺少前天或昨天数据，无法预测今天。")
        if np.sum(dates == yesterday) < points_per_day:
            raise RuntimeError(f"昨天 {yesterday} 数据不完整 ({np.sum(dates == yesterday)}/{points_per_day})，无法预测今天。")
        if np.sum(dates == day_before_yesterday) < points_per_day:
            raise RuntimeError(f"前天 {day_before_yesterday} 数据不完整 ({np.sum(dates == day_before_yesterday)}/{points_per_day})，无法预测今天。")
    else:
        raise ValueError("mode参数必须为 'T' 或 'N'")

def load_and_process_data(df):
    df = df.set_index("timestamp")
    df = df.sort_values('timestamp')
    df = df.resample('1H').mean()
    df.fillna(method='ffill', inplace=True)
    df['瞬时流量'] = df['瞬时流量'].interpolate(method='linear')

    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['lag_1'] = df['瞬时流量'].shift(1)
    df['lag_2'] = df['瞬时流量'].shift(2)
    df['lag_3'] = df['瞬时流量'].shift(3)
    df['lag_12'] = df['瞬时流量'].shift(12)
    df['lag_24'] = df['瞬时流量'].shift(24)
    df['lag_48'] = df['瞬时流量'].shift(48)
    df['rolling_6h_mean'] = df['瞬时流量'].rolling(window=6).mean()
    df['rolling_12h_std'] = df['瞬时流量'].rolling(window=12).std()
    df.fillna(method='bfill', inplace=True)
    return df

def predict_by_mode(mode: str):
    if mode not in ['T', 'N']:
        raise ValueError("mode 参数必须是 'T' 或 'N'")

    df = fetch_recent_data(3)  # 最近3天数据
    if df.empty:
        raise RuntimeError("获取数据为空，无法预测")

    df = load_and_process_data(df)  # 特征工程
    check_data_validity_for_prediction(mode, df)  # 检查数据完整性

    values = df['瞬时流量'].values
    feature_values = df[REQUIRED_FEATURES].values

    forecast_steps = 24
    train_days = 2
    points_per_day = 24

    dates = df.index.date
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)

    if mode == 'N':  # 预测明天
        target_indices = np.where(dates == today)[0]  # 今天数据
        end_idx = target_indices[0]  # 今天数据起始
    else:  # 预测今天
        target_indices = np.where(dates == yesterday)[0]  # 昨天数据
        end_idx = target_indices[0]  # 昨天数据起始

    start_idx = end_idx - train_days * points_per_day
    if start_idx < 0:
        raise RuntimeError("历史数据不足以进行预测")

    sarima_input = values[start_idx: end_idx]
    feature_input = feature_values[end_idx: end_idx + forecast_steps]

    model = SARIMAX(sarima_input, order=(0, 2, 1), seasonal_order=(1, 1, 0, 24),
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    sarima_forecast = model_fit.forecast(steps=forecast_steps)

    rf_models = joblib.load(MODEL_PATH)
    rf_preds = []
    for t in range(forecast_steps):
        sarima_feat = sarima_forecast[t].reshape(-1, 1)
        extra_feat = feature_input[t, :].reshape(1, -1)
        X_rf = np.concatenate([sarima_feat, extra_feat], axis=1)
        y_pred = rf_models[t].predict(X_rf)
        rf_preds.append(y_pred[0])
    # 返回预测值，SARIMA预测值，以及用作预测的输入df（带时间戳的DataFrame）
    return np.array(rf_preds), sarima_forecast, df

def evaluate_prediction(rf_preds, mode):
    df = fetch_recent_data(2)
    df = df.set_index('timestamp').sort_index()

    latest_date = df.index[-1].date()
    dates = df.index.date

    if mode == 'N':  # 预测明天，真实值取最新完整一天
        target_date = latest_date
        true_df = df[df.index.date == target_date]
        if len(true_df) < 48:
            raise RuntimeError(f"{target_date} 数据不完整({len(true_df)}点)，无法评估明天预测")
        true_values = true_df['瞬时流量'].values
        timestamps = true_df.index
    else:  # 预测今天
        target_date = latest_date
        true_df = df[df.index.date == target_date]
        true_values = true_df['瞬时流量'].values
        timestamps = true_df.index

    min_len = min(len(rf_preds), len(true_values))
    rf_preds = rf_preds[:min_len]
    true_values = true_values[:min_len]
    timestamps = timestamps[:min_len]

    mse = mean_squared_error(true_values, rf_preds)
    print(f"MSE: {mse:.4f}")

    plt.figure(figsize=(12, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(timestamps, true_values, label='True Values', marker='o')
    plt.plot(timestamps, rf_preds, label='Predicted Values', marker='x')
    plt.title('True vs Predicted Values')
    plt.xlabel('Timestamp')
    plt.ylabel('瞬时流量')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("true_vs_predicted.png")
    plt.show()

    return mse
