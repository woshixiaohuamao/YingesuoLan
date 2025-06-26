import numpy as np
import pandas as pd


def add_prediction_bounds(preds, lower=0, upper=350):

    preds = np.array(preds)

    # 生成与预测值相同形状的随机浮动
    random_offsets_lower = np.random.uniform(0, 20, size=preds.shape)
    random_offsets_upper = np.random.uniform(0, 20, size=preds.shape)

    # 下界 = 预测值 - 随机[0,20]
    lower_bounds = preds - random_offsets_lower
    # 上界 = 预测值 + 随机[0,20]
    upper_bounds = preds + random_offsets_upper

    # 限制上下界在指定范围内
    lower_bounds = np.clip(lower_bounds, lower, upper)
    upper_bounds = np.clip(upper_bounds, lower, upper)

    return lower_bounds, upper_bounds

def interpolate_to_minute(preds, freq='1T'):

    preds = np.array(preds)

    # 确保预测值长度为24（0点到23点）
    if len(preds) != 24:
        raise ValueError("输入的预测值长度必须为24，表示0点到23点。")

    # 构造时间戳，23点多加一个24点（次日0点）
    timestamps = pd.date_range(start='2025-01-01 00:00:00', periods=25, freq='1H')

    # 在23点后补充一个24点的值（暂时设为与23点相同，可根据需求更改）
    preds_extended = np.append(preds, preds[-1])

    # 构建DataFrame
    df = pd.DataFrame({'timestamp': timestamps, 'value': preds_extended})
    df.set_index('timestamp', inplace=True)

    # 重采样为1分钟频率并进行线性插值
    df_resampled = df.resample(freq).interpolate('linear')

    # 仅返回原始时间段内的数据 + 23点后的60分钟（也就是到次日0点）
    end_time = timestamps[-1]  # 次日0点
    start_time = timestamps[0] # 当日0点
    df_resampled = df_resampled[(df_resampled.index >= start_time) & 
                                (df_resampled.index <= end_time + pd.Timedelta(minutes=59))]

    return df_resampled.reset_index()

from datetime import timedelta

def convert_to_json_format(df, lower_bounds, upper_bounds, input_df, metric="瞬时流量"):
    """
    将预测结果、上下限转换为指定JSON格式列表（含动态时间、周日判定）。

    参数:
    - df: DataFrame，插值后结果，包含 'value' 列（不需要 'timestamp'）
    - lower_bounds: numpy数组，预测下限
    - upper_bounds: numpy数组，预测上限
    - input_df: 原始用于预测的DataFrame，包含 'timestamp'（用于确定预测起点）
    - metric: 指标名称（如“瞬时流量”）

    返回:
    - List[Dict] 格式，适合转为 JSON 输出
    """

    # 确定预测起点：原始数据最后一天 + 1天
    last_date = input_df.index.date.max()
    predict_start_date = last_date + timedelta(days=1)
    
    # 确定预测类型：周日还是工作日
    model_type = "sunday" if predict_start_date.weekday() == 6 else "workday"

    # 生成完整预测时间序列（和df行数对应）
    freq = pd.infer_freq(df['timestamp']) 
    if freq is None:  
        freq = '1T'
    predict_timestamps = pd.date_range(
        start=pd.Timestamp(predict_start_date),
        periods=len(df),
        freq=freq
    )
    
    # 构建JSON列表
    json_list = []
    for i in range(len(df)):
        item = {
            "timestamp": predict_timestamps[i].isoformat(),
            "forecast": float(df.loc[i, 'value']),
            "lower_bound": float(lower_bounds[i]),
            "upper_bound": float(upper_bounds[i]),
            "metric": metric,
            "model_type": model_type
        }
        json_list.append(item)
    
    return json_list


