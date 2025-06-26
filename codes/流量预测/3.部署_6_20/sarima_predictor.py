import pandas as pd
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging
import numpy as np
from models import interpolate_to_minute  # 导入插值函数

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义指标的约束条件
METRIC_CONSTRAINTS = {
    "瞬时流量": {
        "min_value": 0,
        "max_value": 350,
        "max_upper_bound": 350,
        "min_lower_bound": 0
    },
}

def calculate_custom_bounds(forecast_mean: pd.Series):
    """
    自定义计算上下限（针对瞬时流量）
    上下限为预测值 ± 0~20的随机数
    """
    noise = np.random.uniform(0, 20, size=len(forecast_mean))  # 生成0~20的随机扰动
    lower_bounds = forecast_mean - noise
    upper_bounds = forecast_mean + noise
    return lower_bounds, upper_bounds

def apply_custom_constraints(forecast_mean, lower_bounds, upper_bounds):
    """
    对上下限应用业务约束，并保证上下限与预测值关系合理
    参数：
      forecast_mean: 预测值，Series或ndarray
      lower_bounds: 初步下限，Series或ndarray
      upper_bounds: 初步上限，Series或ndarray
    返回：
      处理后的下限和上限 ndarray
    """
    # 先统一转换为numpy数组（如果是Series）
    if isinstance(forecast_mean, pd.Series):
        forecast_mean_np = forecast_mean.values
    else:
        forecast_mean_np = forecast_mean

    if isinstance(lower_bounds, pd.Series):
        lower_bounds_np = lower_bounds.values
    else:
        lower_bounds_np = lower_bounds

    if isinstance(upper_bounds, pd.Series):
        upper_bounds_np = upper_bounds.values
    else:
        upper_bounds_np = upper_bounds

    # 获取业务约束
    constraints = METRIC_CONSTRAINTS.get("瞬时流量", {})
    min_value = constraints.get("min_value", -np.inf)
    max_value = constraints.get("max_value", np.inf)
    max_upper_bound = constraints.get("max_upper_bound", np.inf)
    min_lower_bound = constraints.get("min_lower_bound", -np.inf)

    # 应用clip限制上下限范围
    lower_bounds_np = np.clip(lower_bounds_np, min_lower_bound, max_value)
    upper_bounds_np = np.clip(upper_bounds_np, min_value, max_upper_bound)

    # 确保下限 <= 预测值 <= 上限
    for i in range(len(forecast_mean_np)):
        if lower_bounds_np[i] > forecast_mean_np[i]:
            lower_bounds_np[i] = forecast_mean_np[i] - 0.1 * abs(forecast_mean_np[i])
        if upper_bounds_np[i] < forecast_mean_np[i]:
            upper_bounds_np[i] = forecast_mean_np[i] + 0.1 * abs(forecast_mean_np[i])

    return lower_bounds_np, upper_bounds_np


def predict_sarima(file_path: str, interpolate: bool = True) -> pd.DataFrame:
    """
    使用预训练数据预测周日流量（瞬时流量）
    """
    logger.info("开启周日预测")
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_excel(file_path)
        if 'timestamp' not in df.columns or '瞬时流量' not in df.columns:
            logger.error("数据文件缺少必要的列")
            return pd.DataFrame()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        train_data = df['瞬时流量'].dropna()
        if len(train_data) < 48:
            logger.warning(f"训练数据不足 ({len(train_data)}个点)，无法进行SARIMA建模")
            return pd.DataFrame()

        logger.info(f"开始训练周日SARIMA模型，数据量: {len(train_data)}")
        model = SARIMAX(
            train_data,
            order=(0, 2, 1),
            seasonal_order=(1, 1, 0, 48),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False)
        logger.info("周日模型训练完成")

        forecast = model_fit.get_forecast(steps=48)
        logger.info(f"forecast object: {forecast}")

        forecast_mean = forecast.predicted_mean
        logger.info(f"forecast_mean 类型: {type(forecast_mean)}, 长度: {len(forecast_mean)}")

        # 计算自定义上下限
        lower_bounds, upper_bounds = calculate_custom_bounds(forecast_mean)
        logger.info("calculate_custom_bounds 执行完成")

        # 应用业务约束
        lower_bounds, upper_bounds = apply_custom_constraints(forecast_mean, lower_bounds, upper_bounds)
        logger.info("apply_custom_constraints 执行完成")

        # 时间索引
        today = pd.Timestamp.now().normalize()
        # next_day = today + pd.Timedelta(days=1)
        next_day = today # 只能预测今天
        forecast_dates = pd.date_range(start=next_day, periods=len(forecast_mean), freq='30T')

        forecast_df = pd.DataFrame({
            'timestamp': forecast_dates,
            'forecast': forecast_mean.values,
            'lower_bound': lower_bounds,
            'upper_bound': upper_bounds,
            'metric': '瞬时流量'
        })

        if interpolate:
            logger.info("进行插值处理")
            forecast_df = interpolate_to_minute(forecast_df)
        else:
            logger.info("跳过插值处理")

        return forecast_df

    except Exception as e:
        logger.error(f"周日预测失败: {str(e)}")
        return pd.DataFrame()
