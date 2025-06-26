import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings

# 忽略所有 UserWarning 类型的警告
warnings.filterwarnings("ignore", category=UserWarning)

# 1. 定义模型需要的特征
REQUIRED_FEATURES = [
    'hour', 'day_of_week', 'day_of_month', 'month', 'weekend',
    'lag_1', 'lag_2', 'lag_3', 'lag_12', 'lag_24', 'lag_48',
    'rolling_6h_mean', 'rolling_12h_std'
]


def prepare_data_for_prediction(file_path, timestamp_col='timestamp', value_col='瞬时流量'):
    """
    准备预测数据，确保没有NaN值，完全匹配模型输入格式
    
    参数:
    file_path: 数据文件路径 (支持.csv, .xlsx)
    timestamp_col: 时间戳列名 (默认为'timestamp')
    value_col: 流量值列名 (默认为'瞬时流量')
    
    返回:
    处理后的DataFrame，包含两列: 'timestamp' 和 '瞬时流量'
    """
    # 1. 读取数据
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("不支持的文件格式，请使用CSV或Excel文件")
    
    # 2. 检查必需列
    required_cols = [timestamp_col, value_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"数据中缺少必需的列: '{col}'")
    
    # 3. 提取所需列并重命名
    df = df[[timestamp_col, value_col]].copy()
    df.columns = ['timestamp', '瞬时流量']  # 统一列名
    
    # 4. 转换时间格式
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 5. 按时间排序
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 6. 设置时间索引并重采样为1小时频率
    df.set_index('timestamp', inplace=True)
    df_resampled = df.resample('1H').mean()
    
    # 7. 处理缺失值 - 更健壮的方法
    # 步骤1: 使用线性插值
    df_filled = df_resampled.interpolate(method='linear')
    
    # 步骤2: 向前填充剩余缺失值
    df_filled = df_filled.ffill()
    
    # 步骤3: 向后填充剩余缺失值
    df_filled = df_filled.bfill()
    
    # 步骤4: 如果还有缺失值，使用0填充
    df_filled = df_filled.fillna(0)
    
    # 8. 重置索引
    df_final = df_filled.reset_index()
    
    # 9. 确保有足够的数据
    if len(df_final) < 48:
        print(f"警告: 只有 {len(df_final)} 小时数据，但推荐至少48小时数据以获得最佳预测效果")
        # 如果数据不足，复制数据以满足最小要求
        if len(df_final) > 0:
            while len(df_final) < 48:
                last_row = df_final.iloc[-1].copy()
                last_row['timestamp'] = last_row['timestamp'] + pd.Timedelta(hours=1)
                df_final = pd.concat([df_final, pd.DataFrame([last_row])], ignore_index=True)
    
    return df_final

# 2. 加载训练好的模型
def load_model(model_path):
    """加载预训练模型"""
    return joblib.load(model_path)

# 3. 特征工程函数（必须与训练时完全一致）
def create_features(df, target='瞬时流量'):
    """创建模型需要的特征"""
    # 添加时间特征
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 创建滞后特征
    for lag in [1, 2, 3, 12, 24, 48]:
        df[f'lag_{lag}'] = df[target].shift(lag)
    
    # 创建滚动统计特征
    df['rolling_6h_mean'] = df[target].rolling(window=6, min_periods=1).mean()
    df['rolling_12h_std'] = df[target].rolling(window=12, min_periods=1).std()
    
    return df

# 4. 数据预处理函数
def preprocess_data(history_data, target='瞬时流量'):
    """
    预处理历史数据，准备用于预测
    
    参数:
    history_data - 包含历史数据的DataFrame，必须包含'timestamp'和'target'列
    target - 目标列名（默认为'瞬时流量'）
    
    返回:
    预处理后的DataFrame，包含所有需要的特征
    """
    # 确保时间戳为datetime类型
    history_data['timestamp'] = pd.to_datetime(history_data['timestamp'])
    
    # 按时间排序
    df = history_data.sort_values('timestamp').reset_index(drop=True)
    

    # 创建特征
    df = create_features(df, target)
    
    # 确保所有必需特征都存在
    for feature in REQUIRED_FEATURES:
        if feature not in df.columns:
            # 如果特征不存在，创建默认值
            df[feature] = 0
    
    return df

# 5. 预测函数
def predict_future(model, history_data, steps=24, target='瞬时流量'):
    """
    使用随机森林模型预测未来值
    """
    # 复制历史数据以避免修改原始数据
    df = history_data.copy()
    
    # 预处理数据
    df = preprocess_data(df, target)
    
    # 检查是否有足够的历史数据
    if len(df) < 48:
        raise ValueError(f"需要至少48小时的历史数据来创建所有特征，当前只有 {len(df)} 小时数据")
    
    # 初始化预测结果列表
    predictions = []
    
    # 逐步预测未来
    for i in range(steps):
        # 获取最新可用的数据点（用于预测下一步）
        latest_data = df.iloc[[-1]][REQUIRED_FEATURES]
        
        # 确保没有NaN值 - 最终防线
        if latest_data.isnull().any().any():
            print(f"警告: 预测前发现NaN值，位置: {df.index[-1]}")
            print("存在NaN的特征:")
            print(latest_data.columns[latest_data.isnull().any()].tolist())
            print("填充0处理...")
            latest_data = latest_data.fillna(0)
        
        # 预测下一步
        next_pred = model.predict(latest_data)[0]
        
        # 创建新的时间戳（增加1小时）
        last_timestamp = df['timestamp'].iloc[-1]
        next_timestamp = last_timestamp + timedelta(hours=1)
        
        # 记录预测结果
        predictions.append({
            'timestamp': next_timestamp,
            'prediction': next_pred
        })
        
        # 将预测值添加到历史数据中（用于下一步的滞后特征）
        new_row = {
            'timestamp': next_timestamp,
            target: next_pred
        }
        # 添加新行到DataFrame
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # 更新特征（只更新最后一行）
        df = update_features_for_last_row(df, target)
    
    return pd.DataFrame(predictions)

# 6. 更新最后一行特征
def update_features_for_last_row(df, target):
    """更新最后一行（最新预测点）的特征"""
    # 获取最后一行索引
    last_idx = len(df) - 1
    
    # 更新滞后特征
    for lag in [1, 2, 3, 12, 24, 48]:
        if last_idx - lag >= 0:
            df.loc[last_idx, f'lag_{lag}'] = df[target].iloc[last_idx - lag]
        else:
            # 如果历史数据不足，使用0填充
            df.loc[last_idx, f'lag_{lag}'] = 0
    
    # 更新时间特征
    df.loc[last_idx, 'hour'] = df.loc[last_idx, 'timestamp'].hour
    df.loc[last_idx, 'day_of_week'] = df.loc[last_idx, 'timestamp'].dayofweek
    df.loc[last_idx, 'day_of_month'] = df.loc[last_idx, 'timestamp'].day
    df.loc[last_idx, 'month'] = df.loc[last_idx, 'timestamp'].month
    df.loc[last_idx, 'weekend'] = 1 if df.loc[last_idx, 'day_of_week'] >= 5 else 0
    
    # 更新滚动特征
    try:
        # 重新计算滚动特征
        start_idx = max(0, last_idx - 11)  
        
        # 6小时滚动均值
        rolling_6h = df[target].iloc[start_idx:last_idx+1].rolling(window=6, min_periods=1).mean()
        df.loc[last_idx, 'rolling_6h_mean'] = rolling_6h.iloc[-1] if not rolling_6h.empty else 0
        
        # 12小时滚动标准差
        rolling_12h_std = df[target].iloc[start_idx:last_idx+1].rolling(window=12, min_periods=1).std()
        df.loc[last_idx, 'rolling_12h_std'] = rolling_12h_std.iloc[-1] if not rolling_12h_std.empty else 0
    except Exception as e:
        print(f"更新滚动特征时出错: {str(e)}")
        # 出错时使用默认值
        df.loc[last_idx, 'rolling_6h_mean'] = 0
        df.loc[last_idx, 'rolling_12h_std'] = 0
    
    # 确保最后一行没有NaN值
    for feature in REQUIRED_FEATURES:
        if pd.isna(df.loc[last_idx, feature]):
            print(f"警告: 特征 '{feature}' 在索引 {last_idx} 处为NaN，已用0填充")
            df.loc[last_idx, feature] = 0
    
    return df

# 7. 主函数
def main():
    # 加载模型
    MODEL_PATH = ''
    model = load_model(MODEL_PATH)
    # 准备数据
    history_data = prepare_data_for_prediction(
        file_path="",
        timestamp_col="timestamp",           
        value_col="瞬时流量"                
    )    

    if history_data.isnull().any().any():
        print("警告: 处理后的数据仍包含NaN值")
        print("NaN值统计:")
        print(history_data.isnull().sum())
        # 最终处理 - 填充0
        history_data = history_data.fillna(0)
        print("已用0填充所有剩余NaN值")
    else:
        print("数据预处理完成，无NaN值")

    # 预测未来24小时
    predictions = predict_future(model, history_data, steps=24)
    
    # 可视化预测结果
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.rcParams['font.sans-serif'] = ['SimSun']      
    plt.plot(history_data['timestamp'], history_data['瞬时流量'], 'b-', label='历史数据')
    plt.plot(predictions['timestamp'], predictions['prediction'], 'r--', label='预测值')
    plt.title('随机森林预测结果')
    plt.xlabel('时间')
    plt.ylabel('瞬时流量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('random_forest_prediction.png', dpi=300)
    #plt.show()
    
    # 保存预测结果到CSV
    predictions.to_csv('random_forest_predictions.csv', index=False)
    print("预测结果已保存到 random_forest_predictions.csv")

if __name__ == "__main__":
    main()