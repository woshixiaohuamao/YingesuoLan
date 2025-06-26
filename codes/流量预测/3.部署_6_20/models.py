"""
正常星期的预测
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging
import warnings

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SARIMAModel")

# 忽略警告
warnings.filterwarnings("ignore")

# 定义指标的约束条件
METRIC_CONSTRAINTS = {
    "瞬时流量": {
        "min_value": 0,
        "max_value": 350,
        "max_upper_bound": 350,
        "min_lower_bound": 0
    },
    "总压力": {
        "min_value": 0,
        "max_value": 8,  # 假设压力最大值为10，请根据实际情况调整
        "max_upper_bound": 8,
        "min_lower_bound": 0
    }
}

def apply_constraints(forecast_mean, conf_int, metric_name):
    """
    应用业务约束到预测结果
    
    参数:
    forecast_mean: 预测均值序列
    conf_int: 置信区间DataFrame
    metric_name: 指标名称
    
    返回:
    约束后的预测均值和置信区间
    """
    constraints = METRIC_CONSTRAINTS.get(metric_name, {})
    
    if not constraints:
        return forecast_mean, conf_int
    
    # 应用预测值约束
    min_value = constraints.get("min_value", -np.inf)
    max_value = constraints.get("max_value", np.inf)
    forecast_mean = np.clip(forecast_mean, min_value, max_value)
    
    # 应用置信区间约束
    max_upper_bound = constraints.get("max_upper_bound", np.inf)
    min_lower_bound = constraints.get("min_lower_bound", -np.inf)
    
    conf_int.iloc[:, 0] = np.maximum(conf_int.iloc[:, 0], min_lower_bound)  # 下限
    conf_int.iloc[:, 1] = np.minimum(conf_int.iloc[:, 1], max_upper_bound)  # 上限
    
    # 确保置信区间有意义（下限 <= 预测值 <= 上限）
    for i in range(len(forecast_mean)):
        if conf_int.iloc[i, 0] > forecast_mean[i]:
            conf_int.iloc[i, 0] = forecast_mean[i] - 0.1 * abs(forecast_mean[i])
        
        if conf_int.iloc[i, 1] < forecast_mean[i]:
            conf_int.iloc[i, 1] = forecast_mean[i] + 0.1 * abs(forecast_mean[i])
    
    return forecast_mean, conf_int


def calculate_custom_bounds(forecast_mean: pd.Series, metric_name: str):
    """
    根据指标名称自定义计算上下限。
    瞬时流量：±20内的随机数
    总压力：±1.5内的随机数
    """
    if metric_name == "瞬时流量":
        noise = np.random.uniform(0, 20, size=len(forecast_mean))
        lower_bounds = forecast_mean - noise
        upper_bounds = forecast_mean + noise
    elif metric_name == "总压力":
        noise = np.random.uniform(0, 0.8, size=len(forecast_mean))
        lower_bounds = forecast_mean - noise
        upper_bounds = forecast_mean + noise
    else:
        lower_bounds = forecast_mean
        upper_bounds = forecast_mean
    return lower_bounds, upper_bounds


def interpolate_to_minute(forecast_df: pd.DataFrame) -> pd.DataFrame:

    # 确保时间戳为datetime格式
    forecast_df = forecast_df.copy()
    forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'])

    # 设置时间戳为索引
    forecast_df.set_index('timestamp', inplace=True)

    # 计算插值范围
    start_time = forecast_df.index.min()
    last_time = forecast_df.index.max()
    expected_last_time = last_time + pd.Timedelta(minutes=30)  # 补齐到23:30之后的24:00

    # 如果最后时间不是23:30则报错提醒
    if last_time.time() != pd.Timestamp("23:30").time():
        raise ValueError("最后一个时间点不是23:30，请检查原数据。")

    # 在原DataFrame中加入24:00这一行，数值采用23:30这一行（方便后续插值）
    new_row = forecast_df.loc[[last_time]].copy()
    new_row.index = [expected_last_time]
    forecast_df = pd.concat([forecast_df, new_row])

    # 重新生成完整1分钟索引
    full_index = pd.date_range(start=start_time, end=expected_last_time, freq='min')

    # 对数值列进行线性插值
    interpolated_df = pd.DataFrame(index=full_index)
    for col in ['forecast', 'lower_bound', 'upper_bound']:
        if col in forecast_df.columns:
            interpolated_df[col] = forecast_df[col].resample('min').interpolate('linear')

    # 对非数值列进行前向填充
    for col in forecast_df.columns:
        if col not in ['forecast', 'lower_bound', 'upper_bound']:
            interpolated_df[col] = forecast_df[col].resample('min').ffill()

    # 重置索引
    interpolated_df.reset_index(inplace=True)
    interpolated_df.rename(columns={'index': 'timestamp'}, inplace=True)

    return interpolated_df


def train_sarima_and_forecast(df: pd.DataFrame, target_column: str, steps: int = 48, confidence_level: float = 0.95, interpolate: bool = True):
    """
    训练SARIMA模型并预测"明天整天"48个时间点
    """
    if df.empty:
        logger.error("输入数据框为空")
        return pd.DataFrame()

    df = df.sort_values('timestamp').reset_index(drop=True)
    df.set_index('timestamp', inplace=True)
    train_data = df[target_column].dropna()

    if len(train_data) < 48:
        logger.warning(f"训练数据不足 ({len(train_data)}个点)，无法进行SARIMA建模")
        return pd.DataFrame()

    logger.info(f"开始训练SARIMA模型，数据量: {len(train_data)}，目标: {target_column}")

    try:
        model = SARIMAX(
            train_data,
            order=(0, 2, 1),
            seasonal_order=(1, 1, 0, 48),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False)
        logger.info("模型训练完成")

        forecast = model_fit.get_forecast(steps=steps)
        forecast_mean = forecast.predicted_mean

        # === 使用自定义上下限计算函数替代SARIMA置信区间 ===
        lower_bounds, upper_bounds = calculate_custom_bounds(forecast_mean, target_column)

        # === 应用业务约束 ===
        constraints = METRIC_CONSTRAINTS.get(target_column, {})
        min_value = constraints.get("min_value", -np.inf)
        max_value = constraints.get("max_value", np.inf)
        max_upper_bound = constraints.get("max_upper_bound", np.inf)
        min_lower_bound = constraints.get("min_lower_bound", -np.inf)

        lower_bounds = np.clip(lower_bounds, min_lower_bound, max_value)
        upper_bounds = np.clip(upper_bounds, min_value, max_upper_bound)

        # 确保下限 <= 预测值 <= 上限
        for i in range(len(forecast_mean)):
            if lower_bounds[i] > forecast_mean[i]:
                lower_bounds[i] = forecast_mean[i] - 0.1 * abs(forecast_mean[i])
            if upper_bounds[i] < forecast_mean[i]:
                upper_bounds[i] = forecast_mean[i] + 0.1 * abs(forecast_mean[i])

        today = pd.Timestamp.now().normalize()
        # next_day = today + pd.Timedelta(days=1)
        next_day = today  # 永远只能预测今天，因为明天数据不全
        forecast_dates = pd.date_range(start=next_day, periods=48, freq='30T')

        forecast_df = pd.DataFrame({
            'timestamp': forecast_dates,
            'forecast': forecast_mean.values,
            'lower_bound': lower_bounds,
            'upper_bound': upper_bounds
        })
        forecast_df['metric'] = target_column

        logger.info(f"预测完成，生成{len(forecast_df)}个预测点（明天整天）")
        
        if interpolate:
            forecast_df = interpolate_to_minute(forecast_df)

        return forecast_df

    except Exception as e:
        logger.error(f"SARIMA模型训练或预测失败: {str(e)}")
        return pd.DataFrame()
