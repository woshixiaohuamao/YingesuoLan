import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')
# 在文件顶部添加导入
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为"宋体"
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
mpl.rcParams['font.family'] = 'SimSun'

# 创建结果目录
RESULTS_DIR = "results_rf"
os.makedirs(RESULTS_DIR, exist_ok=True)

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
    
    return X, y, available_features, df_clean, timestamps

# 突变点检测器类
class ChangePointDetector:
    """突变点检测器，提供两种处理策略"""
    
    def __init__(self, model=None):
        """
        初始化检测器
        
        参数：
        model: 自定义模型（需实现fit和predict方法），默认为RandomForestRegressor
        """
        self.model = model or RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.anomaly_mask = None
        self.anomaly_indices = None
        self.residuals = None
    
    def detect_change_points(self, X, y, confidence_level=0.95):
        """检测突变点"""
        # 检查输入数据是否有NaN
        if X.isnull().any().any():
            print("警告: 输入特征矩阵包含NaN值，正在填充0...")
            X = X.fillna(0)
        
        if y.isnull().any():
            print("警告: 目标变量包含NaN值，正在填充0...")
            y = y.fillna(0)
        
        # 训练初始模型
        self.model.fit(X, y)
        
        # 获取预测值和残差
        y_pred = self.model.predict(X)
        self.residuals = y - y_pred
        
        # 计算置信区间
        std_residual = np.std(self.residuals)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * std_residual
        
        # 创建置信区间
        lower_bound = y_pred - margin_of_error
        upper_bound = y_pred + margin_of_error
        
        # 识别突变点
        self.anomaly_mask = (y < lower_bound) | (y > upper_bound)
        self.anomaly_indices = np.where(self.anomaly_mask)[0]
        
        print(f"检测到 {len(self.anomaly_indices)} 个突变点 (占总数据 {len(self.anomaly_indices)/len(y):.2%})")
        
        return {
            'model': self.model,
            'y_pred': y_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'anomaly_mask': self.anomaly_mask,
            'anomaly_indices': self.anomaly_indices,
            'residuals': self.residuals
        }
    
    def strategy_remove_anomalies(self, X, y):
        """剔除突变点后重新训练"""
        if self.anomaly_mask is None:
            raise ValueError("请先运行 detect_change_points() 方法")
        
        # 确保没有NaN值
        if X.isnull().any().any():
            print("警告: 特征矩阵包含NaN值，正在填充0...")
            X = X.fillna(0)
        
        if y.isnull().any():
            print("警告: 目标变量包含NaN值，正在填充0...")
            y = y.fillna(0)
        
        X_clean = X[~self.anomaly_mask]
        y_clean = y[~self.anomaly_mask]
        
        print(f"剔除策略: 使用 {len(X_clean)} 个样本 (原始样本 {len(X)})")
        
        model = self.model.__class__(**self.model.get_params())
        model.fit(X_clean, y_clean)
        y_pred = model.predict(X)
        
        return model, y_pred
    
    def strategy_add_parameters(self, X, y):
        """将突变点作为额外特征加入模型"""
        if self.anomaly_indices is None:
            raise ValueError("请先运行 detect_change_points() 方法")
        
        # 确保没有NaN值
        if X.isnull().any().any():
            print("警告: 特征矩阵包含NaN值，正在填充0...")
            X = X.fillna(0)
        
        if y.isnull().any():
            print("警告: 目标变量包含NaN值，正在填充0...")
            y = y.fillna(0)
        
        # 创建突变点特征
        X_augmented = X.copy()
        
        # 添加突变点指示器 - 更安全的方法
        # 创建一个全零矩阵，然后设置突变点位置为1
        anomaly_matrix = np.zeros((len(X), len(self.anomaly_indices)))
        
        for i, idx in enumerate(self.anomaly_indices):
            # 确保索引在有效范围内
            if idx < len(anomaly_matrix):
                anomaly_matrix[idx, i] = 1
        
        # 将矩阵转换为DataFrame
        anomaly_df = pd.DataFrame(
            anomaly_matrix,
            columns=[f"anomaly_{i}" for i in range(len(self.anomaly_indices))],
            index=X.index
        )
        
        # 合并到原始特征矩阵
        X_augmented = pd.concat([X_augmented, anomaly_df], axis=1)
        
        # 确保没有NaN值
        if X_augmented.isnull().any().any():
            print("警告: 特征矩阵包含NaN值，正在填充0...")
            X_augmented = X_augmented.fillna(0)
        
        print(f"参数增强策略: 添加 {len(self.anomaly_indices)} 个突变点特征")
        
        model = self.model.__class__(**self.model.get_params())
        model.fit(X_augmented, y)
        y_pred = model.predict(X_augmented)
        
        return model, y_pred, X_augmented
    
    def detect_and_handle(self, X, y, confidence_level=0.95):
        """完整流程：检测突变点并应用两种处理策略"""
        # 检测突变点
        detection_results = self.detect_change_points(X, y, confidence_level)
        
        # 应用策略一
        remove_model, remove_pred = self.strategy_remove_anomalies(X, y)
        
        # 应用策略二
        add_model, add_pred, X_augmented = self.strategy_add_parameters(X, y)
        
        return {
            'detection': detection_results,
            'strategy_remove': {
                'model': remove_model,
                'y_pred': remove_pred
            },
            'strategy_add_feature': {
                'model': add_model,
                'y_pred': add_pred,
                'X_augmented': X_augmented
            }
        }

# 评估模型并保存结果
def evaluate_and_save_model(model, model_name, X, y_true, y_pred, timestamps, df_clean, results_dir=RESULTS_DIR):
    """评估模型并保存结果"""
    # 创建模型目录
    model_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # 计算评估指标
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # 保存评估指标
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    with open(os.path.join(model_dir, f'metrics_{model_name}.txt'), 'w') as f:
        f.write(f"{model_name} 模型评估指标:\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"R2: {r2:.4f}\n")
    
    # 保存模型
    joblib.dump(model, os.path.join(model_dir, f'model_{model_name}.pkl'))
    
    # 可视化预测结果
    plt.figure(figsize=(15, 8))
    plt.plot(timestamps, y_true, 'b-', label='实际值', alpha=0.7)
    plt.plot(timestamps, y_pred, 'r--', label='预测值', linewidth=2)
    plt.title(f'{model_name}模型预测结果')
    plt.xlabel('时间')
    plt.ylabel('瞬时流量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f'prediction_{model_name}.png'), dpi=300)
    plt.close()
    
    # 可视化残差
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title(f'{model_name}模型残差图')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f'residuals_{model_name}.png'), dpi=300)
    plt.close()
    
    # 可视化特征重要性（如果是随机森林）
    if hasattr(model, 'feature_importances_'):
        try:
            
            if hasattr(model, 'feature_names_in_'):
               
                feature_names = model.feature_names_in_
            elif hasattr(X, 'columns'):
                
                feature_names = X.columns
            else:
             
                feature_names = [f'feature_{i}' for i in range(model.n_features_in_)]
            
            # 确保特征数量匹配
            if len(feature_names) == len(model.feature_importances_):
                feature_importances = pd.Series(model.feature_importances_, index=feature_names)
                sorted_importances = feature_importances.sort_values(ascending=False)
                
                plt.figure(figsize=(12, 8))
                sns.barplot(x=sorted_importances.values, y=sorted_importances.index, palette="viridis")
                plt.title(f'{model_name}模型特征重要性')
                plt.xlabel('重要性')
                plt.ylabel('特征')
                plt.tight_layout()
                plt.savefig(os.path.join(model_dir, f'feature_importance_{model_name}.png'), dpi=300)
                plt.close()
            else:
                print(f"警告: 特征数量不匹配 ({len(feature_names)} vs {len(model.feature_importances_)})，跳过特征重要性可视化")
        except Exception as e:
            print(f"生成特征重要性图时出错: {str(e)}")
    
    return metrics

# 主函数
def main(data_path):
    """主流程函数"""
    # 加载和预处理数据
    df = load_and_preprocess(data_path)
    
    # 准备特征和目标变量
    X, y, features, df_clean, timestamps = prepare_data(df)
    
    # 创建检测器实例
    detector = ChangePointDetector()
    
    # 检测突变点并应用处理策略
    print("\n=== 执行突变点检测和处理策略 ===")
    results = detector.detect_and_handle(X, y, confidence_level=0.95)
    
    # 获取三种模型的结果
    initial_model = results['detection']['model']
    initial_pred = results['detection']['y_pred']
    
    remove_model = results['strategy_remove']['model']
    remove_pred = results['strategy_remove']['y_pred']
    
    add_model = results['strategy_add_feature']['model']
    add_pred = results['strategy_add_feature']['y_pred']
    
    # 获取突变点信息
    anomaly_mask = results['detection']['anomaly_mask']
    anomaly_indices = results['detection']['anomaly_indices']
    
    # 1. 保存并评估初始模型
    print("\n=== 评估初始模型 ===")
    initial_metrics = evaluate_and_save_model(
        initial_model, "initial_model", X, y, initial_pred, timestamps, df_clean
    )
    
    # 2. 保存并评估策略一模型（剔除突变点）
    print("\n=== 评估策略一模型（剔除突变点）===")
    remove_metrics = evaluate_and_save_model(
        remove_model, "strategy_remove", X, y, remove_pred, timestamps, df_clean
    )
    
    # 3. 保存并评估策略二模型（添加特征）
    print("\n=== 评估策略二模型（添加特征）===")
    # 对于策略二，使用增强后的特征矩阵 X_augmented
    add_metrics = evaluate_and_save_model(
        add_model, "strategy_add_feature", 
        results['strategy_add_feature']['X_augmented'],  # 使用增强后的特征矩阵
        y, add_pred, timestamps, df_clean
    )
    
    # 比较三种模型的性能
    metrics_df = pd.DataFrame({
        '初始模型': initial_metrics,
        '策略一（剔除）': remove_metrics,
        '策略二（添加）': add_metrics
    }).T
    
    metrics_df.to_csv(os.path.join(RESULTS_DIR, 'model_metrics_comparison.csv'))
    
    # 可视化三种模型的MSE比较
    plt.figure(figsize=(12, 8))
    bars = plt.bar(metrics_df.index, metrics_df['MSE'], color=['blue', 'green', 'red'])
    plt.title('三种模型MSE比较')
    plt.ylabel('均方误差(MSE)')
    plt.grid(axis='y', alpha=0.3)
    
    # 在柱子上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison_mse.png'), dpi=300)
    plt.close()
    
    # 可视化突变点检测结果
    plt.figure(figsize=(15, 8))
    plt.plot(df_clean['timestamp'], y, 'b-', label='实际值', alpha=0.7)
    plt.plot(df_clean['timestamp'], initial_pred, 'g-', label='初始模型预测值', alpha=0.7)
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
    plt.savefig(os.path.join(RESULTS_DIR, 'change_points_detection.png'), dpi=300)
    plt.close()
    
    # 可视化三种模型的预测结果对比
    plt.figure(figsize=(18, 10))
    plt.plot(df_clean['timestamp'], y, 'b-', label='实际值', linewidth=1.5)
    plt.plot(df_clean['timestamp'], initial_pred, 'g-', label='初始模型', alpha=0.7)
    plt.plot(df_clean['timestamp'], remove_pred, 'r--', label='策略一（剔除）', linewidth=1.5)
    plt.plot(df_clean['timestamp'], add_pred, 'm:', label='策略二（添加）', linewidth=1.5)
    
    # 标记突变点
    plt.scatter(df_clean.loc[anomaly_mask, 'timestamp'], 
               y[anomaly_mask], 
               color='black', s=80, marker='x', label='突变点')
    
    plt.title('三种模型预测结果对比')
    plt.xlabel('时间')
    plt.ylabel('瞬时流量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'all_models_prediction_comparison.png'), dpi=300)
    plt.close()
    
    print("\n=== 所有结果已保存到文件夹: results_rf ===")
    
    return {
        'metrics': metrics_df,
        'results': results
    }

# 执行主函数
if __name__ == "__main__":
    # 替换为你的数据文件路径
    data_file_path = ""
    
    # 运行主流程
    results = main(data_file_path)
    
    # 打印模型性能比较
    print("\n=== 模型性能比较 ===")
    print(results['metrics'])
    
    print("\n=== 分析完成 ===")