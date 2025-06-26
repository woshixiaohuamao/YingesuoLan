import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SARIMAModel")
warnings.filterwarnings("ignore")

METRIC_CONSTRAINTS = {
    "瞬时流量": {
        "min_value": 0,
        "max_value": 350,
        "max_upper_bound": 350,
        "min_lower_bound": 0
    },
    "总压力": {
        "min_value": 0,
        "max_value": 10, 
        "max_upper_bound": 10,
        "min_lower_bound": 0
    }
}

def apply_constraints(forecast_mean, conf_int, metric_name):
    constraints = METRIC_CONSTRAINTS.get(metric_name, {})
    
    if not constraints:
        return forecast_mean, conf_int
    
    min_value = constraints.get("min_value", -np.inf)
    max_value = constraints.get("max_value", np.inf)
    forecast_mean = np.clip(forecast_mean, min_value, max_value)
    
    max_upper_bound = constraints.get("max_upper_bound", np.inf)
    min_lower_bound = constraints.get("min_lower_bound", -np.inf)
    
    conf_int.iloc[:, 0] = np.maximum(conf_int.iloc[:, 0], min_lower_bound)  # 下限
    conf_int.iloc[:, 1] = np.minimum(conf_int.iloc[:, 1], max_upper_bound)  # 上限
    
    for i in range(len(forecast_mean)):
        if conf_int.iloc[i, 0] > forecast_mean[i]:
            conf_int.iloc[i, 0] = forecast_mean[i] - 0.1 * abs(forecast_mean[i])
        
        if conf_int.iloc[i, 1] < forecast_mean[i]:
            conf_int.iloc[i, 1] = forecast_mean[i] + 0.1 * abs(forecast_mean[i])
    
    return forecast_mean, conf_int



def interpolate_to_minute(forecast_df: pd.DataFrame) -> pd.DataFrame:
    forecast_df = forecast_df.copy()
    forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'])
    
    forecast_df.set_index('timestamp', inplace=True)
    
    start_time = forecast_df.index.min()
    end_time = forecast_df.index.max()
    full_index = pd.date_range(start=start_time, end=end_time, freq='min')
    
    interpolated_df = pd.DataFrame(index=full_index)
    for col in ['forecast', 'lower_bound', 'upper_bound']:
        if col in forecast_df.columns:
            interpolated_df[col] = forecast_df[col].resample('min').interpolate('linear')

    for col in forecast_df.columns:
        if col not in ['forecast', 'lower_bound', 'upper_bound']:
            interpolated_df[col] = forecast_df[col].resample('min').ffill()
    
    interpolated_df.reset_index(inplace=True)
    interpolated_df.rename(columns={'index': 'timestamp'}, inplace=True)
    
    return interpolated_df

def train_sarima_and_forecast(df: pd.DataFrame, target_column: str, steps: int = 48, confidence_level: float = 0.95, interpolate: bool = True):
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
        conf_int = forecast.conf_int(alpha=1 - confidence_level)

        # forecast_mean, conf_int = apply_constraints(forecast_mean, conf_int, target_column)

        today = pd.Timestamp.now().normalize() 
        next_day = today + pd.Timedelta(days=1)  
        forecast_dates = pd.date_range(start=next_day, periods=48, freq='30T')

        forecast_df = pd.DataFrame({
            'timestamp': forecast_dates,
            'forecast': forecast_mean.values,
            'lower_bound': conf_int.iloc[:, 0].values,
            'upper_bound': conf_int.iloc[:, 1].values
        })
        forecast_df['metric'] = target_column

        logger.info(f"预测完成，生成{len(forecast_df)}个预测点（明天整天）")

        if interpolate:
            forecast_df = interpolate_to_minute(forecast_df)
        
        return forecast_df

    except Exception as e:
        logger.error(f"SARIMA模型训练或预测失败: {str(e)}")
        return pd.DataFrame()

