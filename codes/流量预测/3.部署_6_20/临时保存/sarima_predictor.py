import pandas as pd
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging
from models import interpolate_to_minute  


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_sarima(file_path: str, interpolate: bool = True) -> pd.DataFrame:
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
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05) 

        today = pd.Timestamp.now().normalize()  
        next_day = today + pd.Timedelta(days=1) 
        forecast_dates = pd.date_range(start=next_day, periods=48, freq='30T')

        forecast_df = pd.DataFrame({
            'timestamp': forecast_dates,
            'forecast': forecast_mean.values,
            'lower_bound': conf_int.iloc[:, 0].values,
            'upper_bound': conf_int.iloc[:, 1].values,
            'metric': '瞬时流量'
        })
        
        if interpolate:
            forecast_df = interpolate_to_minute(forecast_df)
        
        return forecast_df
    
    except Exception as e:
        logger.error(f"周日预测失败: {str(e)}")
        return pd.DataFrame()
