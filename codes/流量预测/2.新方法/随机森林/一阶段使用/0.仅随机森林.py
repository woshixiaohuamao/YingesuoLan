# 在你的主代码中
from change_point_detector import ChangePointDetector
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为"宋体"
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
mpl.rcParams['font.family'] = 'SimSun'

# 1. 加载并预处理数据
def load_and_preprocess(file_path):
    """加载数据并进行预处理"""
    # 读取数据
    df = pd.read_excel(file_path, usecols=['timestamp', '瞬时流量'])
    
    # 确保日期时间格式正确
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 重采样为1小时频率
    df.set_index('timestamp', inplace=True)
    df_resampled = df.resample('1H').mean()
    df_resampled = df_resampled.sort_index()  # 确保按时间排序
    
    # 补齐缺失值 - 使用前后两小时的均值填充
    df_filled = df_resampled.copy()
    for col in df_filled.columns:
        # 使用前后两小时的均值填充
        df_filled[col] = df_filled[col].fillna(
            df_filled[col].rolling(window=4, min_periods=1, center=True).mean()
        )
        # 如果还有缺失值，使用整体均值填充
        df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
    
    # 重置索引以便添加时间特征
    df = df_filled.reset_index()
    
    # 添加时间特征
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    print(f"数据集加载完成，包含 {len(df)} 条记录，时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"特征数量: {len(df.columns)}")
    
    return df

# 2. 准备建模数据
def prepare_data(df, target='瞬时流量'):
    """准备特征和目标变量"""
    # 创建滞后特征
    for lag in [1, 2, 3, 12, 24, 48]:
        df[f'lag_{lag}'] = df[target].shift(lag)
    
    # 创建滚动统计特征
    df['rolling_6h_mean'] = df[target].rolling(window=6, min_periods=1).mean()
    df['rolling_12h_std'] = df[target].rolling(window=12, min_periods=1).std()
    
    # 选择特征
    features = [
        'hour', 'day_of_week', 'day_of_month', 'month', 'weekend',
        'lag_1', 'lag_2', 'lag_3', 'lag_12', 'lag_24', 'lag_48',
        'rolling_6h_mean', 'rolling_12h_std'
    ]
    
    # 确保特征存在
    available_features = [f for f in features if f in df.columns]
    print(f"使用的特征: {available_features}")
    
    # 删除因滞后特征产生的缺失值
    df_clean = df.dropna(subset=available_features)
    
    X = df_clean[available_features]
    y = df_clean[target]
    timestamps = df_clean['timestamp']  # 获取时间戳序列
    
    return X, y, available_features, df_clean, timestamps  # 返回时间戳序列


# 加载和预处理数据
data_file_path = ""
# 1. 加载并预处理数据
df = load_and_preprocess(data_file_path)

# 2. 准备数据 - 添加第5个变量接收时间戳
X, y, features, df_clean, timestamps = prepare_data(df)  

# 3. 创建检测器实例
detector = ChangePointDetector()

# 4. 检测突变点并应用处理策略
results = detector.detect_and_handle(X, y, confidence_level=0.95)

# 5. 获取结果
anomaly_mask = results['detection']['anomaly_mask']
anomaly_indices = results['detection']['anomaly_indices']

# 使用策略一处理后的模型
remove_model = results['strategy_remove']['model']

# 使用策略二处理后的模型
add_model = results['strategy_add_feature']['model']
X_augmented = results['strategy_add_feature']['X_augmented']

plt.figure(figsize=(12, 6))
plt.plot(df_clean['timestamp'], y, 'b-', label='实际值')
plt.plot(df_clean['timestamp'], results['detection']['y_pred'], 'g-', label='预测值')
plt.fill_between(df_clean['timestamp'], 
                results['detection']['lower_bound'], 
                results['detection']['upper_bound'], 
                color='gray', alpha=0.3, label='置信区间')
plt.scatter(df_clean.loc[anomaly_mask, 'timestamp'], 
           y[anomaly_mask], 
           color='red', s=50, label='突变点')
plt.title('突变点检测结果')
plt.xlabel('时间')
plt.ylabel('瞬时流量')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('change_points_detection.png', dpi=300)
plt.show()