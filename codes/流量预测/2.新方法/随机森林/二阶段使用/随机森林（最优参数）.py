import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.multioutput import MultiOutputRegressor
import os

# 1. 数据加载与预处理 
def load_data(file_path):
    df = pd.read_excel(file_path)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 提取时间特征
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    
    # 创建预测目标：未来48个点（一天的数据）
    target = '瞬时流量'
    n_forecast = 48  # 预测48个点（一天）
    
    # 创建未来48步的目标变量
    for i in range(1, n_forecast+1):
        df[f'target_{i}'] = df[target].shift(-i)
    
    # 删除最后48行（没有未来的目标值）
    df = df.dropna(subset=[f'target_{n_forecast}'])
    
    # 设置目标变量
    target_columns = [f'target_{i}' for i in range(1, n_forecast+1)]
    y = df[target_columns]
    X = df.drop(columns=[target, 'timestamp'] + target_columns)
    
    return X, y, df['timestamp'], n_forecast

# 3. 评估与可视化
def evaluate_and_visualize(model, X_test, y_test, timestamps, n_forecast):
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算总体MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f"测试集总体MSE: {mse:.4f}")
    print(f"测试集总体RMSE: {np.sqrt(mse):.4f}")
    
    # 计算每个预测步长的MSE
    step_mses = []
    for i in range(n_forecast):
        step_mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
        step_mses.append(step_mse)
        print(f"步长 {i+1} ({(i+1)*30}分钟) MSE: {step_mse:.4f}, RMSE: {np.sqrt(step_mse):.4f}")
    
    # 创建更大的画布
    plt.figure(figsize=(18, 12))
    plt.rcParams['font.sans-serif'] = ['SimSun']       # 设置字体为“宋体”，确保绘制图像不出错
    # 子图1：测试集实际值与预测值对比
    plt.subplot(2, 1, 1)
    test_dates = timestamps.iloc[X_test.index]
    
    # 准备时间轴（每30分钟一个点）
    time_points = [test_dates.iloc[0] + pd.Timedelta(minutes=30*i) for i in range(n_forecast)]
    
    # 绘制实际值
    plt.plot(time_points, y_test.values[0], 'b-o', label='实际值', alpha=0.9, linewidth=2)
    
    # 绘制预测值
    plt.plot(time_points, y_pred[0], 'r--s', label='预测值', alpha=0.9, linewidth=2)
    
    plt.xlabel('时间')
    plt.ylabel('流量值')
    plt.title('周日全天预测对比 (实际值 vs 预测值)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 子图2：预测误差随步长变化
    plt.subplot(2, 1, 2)
    steps = np.arange(1, n_forecast+1)
    plt.plot(steps, np.sqrt(step_mses), 'g-o', linewidth=2)
    plt.xlabel('预测步长 (每步30分钟)')
    plt.ylabel('RMSE')
    plt.title('预测误差随步长变化')
    plt.grid(True)
    plt.xticks(np.arange(0, n_forecast+1, 6))
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('sunday_prediction_results.png', dpi=300)
    plt.show()
    

# 主函数
def main(file_path):
    # 加载数据
    X, y, timestamps, n_forecast = load_data(file_path)
    
    # 划分数据集 - 使用最后48个点作为测试集（一天）
    # 其余所有数据作为训练集
    X_train = X.iloc[:-n_forecast]
    y_train = y.iloc[:-n_forecast]
    
    X_test = X.iloc[-n_forecast:]
    y_test = y.iloc[-n_forecast:]
    
    print(f"训练集大小: {X_train.shape[0]} 个样本")
    print(f"测试集大小: {X_test.shape[0]} 个样本 (一天)")
    
    # 使用指定的超参数创建模型
    print("\n使用指定超参数构建模型...")
    base_model = RandomForestRegressor(
        n_estimators=200,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features=0.8,
        max_depth=10,
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    )
    
    model = MultiOutputRegressor(base_model)
    
    # 训练模型
    print("\n训练模型...")
    model.fit(X_train, y_train)
    
    # 评估与可视化
    evaluate_and_visualize(model, X_test, y_test, timestamps, n_forecast)
    
    return model

# 使用示例
if __name__ == "__main__":
    file_path = "" 
    trained_model = main(file_path)
    
    # 打印模型信息
    print("\n模型训练完成！")
    print(f"模型配置: {trained_model.estimator}")