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
plt.rcParams['font.sans-serif'] = ['SimSun']  
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

# 3. 时间序列交叉验证
def time_series_cv(X, y, timestamps):
    """执行时间序列交叉验证"""
    # 设置5折交叉验证，每个折预测最后12个点
    tscv = TimeSeriesSplit(n_splits=5, test_size=12)
    
    results = []
    fold = 0
    
    # 创建大图用于所有折叠的结果
    fig, axes = plt.subplots(5, 1, figsize=(15, 25))
    fig.suptitle('各折叠真实值与预测值对比', fontsize=16)
    
    for train_index, test_index in tscv.split(X):
        fold += 1
        print(f"\n=== 折叠 {fold} ===")
        
        # 划分训练集和测试集
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        test_timestamps = timestamps.iloc[test_index]  # 获取测试集对应的时间戳
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 评估
        mse = mean_squared_error(y_test, y_pred)
        print(f"测试集MSE: {mse:.4f}")
        
        # 记录结果
        results.append({
            'fold': fold,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'mse': mse,
            'y_test': y_test.values,
            'y_pred': y_pred
        })
        
        # 绘制当前折叠的时序图 - 使用时间戳作为x轴
        ax = axes[fold-1]
        ax.plot(test_timestamps, y_test.values, 'b-', label='真实值')
        ax.plot(test_timestamps, y_pred, 'r--', label='预测值')
        ax.set_title(f'折叠 {fold} - MSE: {mse:.4f}')
        ax.set_xlabel('时间')
        ax.set_ylabel('瞬时流量')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置日期格式并旋转标签
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%m-%d %H:%M'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    plt.savefig('cross_validation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

# 4. 主函数
def main(data_path):
    """主流程函数"""
    # 加载和预处理数据
    df = load_and_preprocess(data_path)
    
    # 准备特征和目标变量
    X, y, features, df_clean, timestamps = prepare_data(df)  # 获取时间戳序列
    
    # 执行时间序列交叉验证
    print("\n=== 时间序列交叉验证 ===")
    cv_results = time_series_cv(X, y, timestamps)  
    
    # 汇总评估结果
    mse_scores = [result['mse'] for result in cv_results]
    avg_mse = np.mean(mse_scores)
    print(f"\n平均MSE: {avg_mse:.4f}")
    
    # 返回结果
    return {
        'df': df,
        'cv_results': cv_results,
        'avg_mse': avg_mse
    }

# 执行主函数
if __name__ == "__main__":
    data_file_path = ""
    
    # 运行主流程
    results = main(data_file_path)
    
    print("\n=== 分析完成 ===")