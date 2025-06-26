import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import os
from datetime import timedelta
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# ========== 配置参数 ==========
CONFIG = {
    'data_path': r'',          # 数据文件路径
    'timestamp_col': 'timestamp',        # 时间戳列名
    'target_col': '瞬时流量',             # 预测目标列名
    'resample_freq': '30T',              # 重采样频率（30分钟）
    'sequence_length': 48,               # 输入序列长度（24小时）
    'prediction_length': 48,             # 预测长度（24小时）
    'n_splits': 5,                       # 时间序列交叉验证折数
    'lstm_params': {
        'input_size': 1,                 # 初始值，将在数据加载后更新
        'hidden_size': 128,               # LSTM隐藏层大小
        'num_layers': 2,                 # LSTM层数
        'output_size': 48                # 输出序列长度（预测48个点）
    },
    'training': {
        'batch_size': 32,
        'epochs': 100,
        'patience': 15,
        'lr': 0.001
    },
    'output_dir': 'results'              # 输出目录
}

# 创建输出目录
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ========== 1. 数据加载与预处理 ==========
def load_and_preprocess_data(config):
    """加载数据并进行预处理"""
    # 读取Excel文件
    df = pd.read_excel(config['data_path'], parse_dates=[config['timestamp_col']])
    df.set_index(config['timestamp_col'], inplace=True)
    
    # 重采样到30分钟频率
    resampled = df[config['target_col']].resample(config['resample_freq']).mean()
    
    # 处理缺失值（使用前向填充）
    resampled = resampled.ffill()
    
    # 创建时间特征
    data = pd.DataFrame({config['target_col']: resampled})
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['day_of_month'] = data.index.day
    data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
    
    # 更新输入特征维度
    config['lstm_params']['input_size'] = data.shape[1]
    print(f"输入特征维度更新为: {data.shape[1]}")
    
    return data

data = load_and_preprocess_data(CONFIG)
print(f"重采样后数据形状: {data.shape}")
print(f"数据时间范围: {data.index.min()} 至 {data.index.max()}")
print(f"特征列表: {data.columns.tolist()}")

# ========== 2. 序列数据生成 ==========
def create_sequences(data, seq_len, pred_len, target_col):
    """创建输入序列和目标序列"""
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        # 输入序列：seq_len个时间步的所有特征
        X_seq = data.iloc[i:i+seq_len].values
        
        # 目标序列：接下来的pred_len个时间步的目标值
        y_seq = data[target_col].iloc[i+seq_len:i+seq_len+pred_len].values
        
        X.append(X_seq)
        y.append(y_seq)
    
    return np.array(X), np.array(y)

# 创建序列数据集
X, y = create_sequences(
    data, 
    seq_len=CONFIG['sequence_length'], 
    pred_len=CONFIG['prediction_length'],
    target_col=CONFIG['target_col']
)

print(f"序列数据集形状: X={X.shape}, y={y.shape}")

# ========== 3. 定义LSTM模型（多步预测） ==========
class MultiStepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # LSTM处理
        out, _ = self.lstm(x)
        
        # 只取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 全连接层生成预测
        return self.fc(out)

# ========== 4. 模型训练函数 ==========
def train_model(model, train_loader, val_loader, config):
    """训练模型并实现早停机制"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"使用设备: {device}")
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    best_model = None
    counter = 0
    history = {'train': [], 'val': []}
    
    for epoch in range(config['training']['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        # 计算平均损失
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{config['training']['epochs']} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # 早停机制
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= config['training']['patience']:
                print(f"早停在第 {epoch+1} 轮")
                break
    
    # 加载最佳模型
    model.load_state_dict(best_model)
    return model, history, best_loss

# ========== 5. 时间序列交叉验证 ==========
def time_series_cross_validation(X, y, data, config):
    """执行时间序列交叉验证"""
    tscv = TimeSeriesSplit(n_splits=config['n_splits'])
    
    # 存储结果
    results = {
        'fold': [],
        'mse': [],
        'predictions': [],
        'actuals': [],
        'timestamps': []
    }
    
    fold = 1
    for train_index, test_index in tscv.split(X):
        print(f"\n{'='*50}")
        print(f"正在处理第 {fold}/{config['n_splits']} 折")
        print(f"训练集大小: {len(train_index)}, 测试集大小: {len(test_index)}")
        print(f"训练时间范围: {data.index[train_index[0]]} 至 {data.index[train_index[-1]]}")
        print(f"测试时间范围: {data.index[test_index[0]]} 至 {data.index[test_index[-1]]}")
        print(f"{'='*50}")
        
        # 划分训练集和测试集
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # 归一化处理 - 为每个特征单独归一化
        scalers = []
        X_train_norm = np.zeros_like(X_train)
        X_test_norm = np.zeros_like(X_test)
        
        # 对每个特征单独归一化
        for feature_idx in range(X_train.shape[2]):
            feature_scaler = MinMaxScaler()
            
            # 训练集特征归一化
            train_feature = X_train[:, :, feature_idx].reshape(-1, 1)
            feature_scaler.fit(train_feature)
            
            # 应用归一化到训练集
            train_feature_norm = feature_scaler.transform(train_feature)
            X_train_norm[:, :, feature_idx] = train_feature_norm.reshape(X_train.shape[0], X_train.shape[1])
            
            # 应用归一化到测试集
            test_feature = X_test[:, :, feature_idx].reshape(-1, 1)
            test_feature_norm = feature_scaler.transform(test_feature)
            X_test_norm[:, :, feature_idx] = test_feature_norm.reshape(X_test.shape[0], X_test.shape[1])
            
            scalers.append(feature_scaler)
        
        # 转换为PyTorch张量
        X_train_t = torch.tensor(X_train_norm, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test_norm, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=True
        )
        
        test_dataset = TensorDataset(X_test_t, y_test_t)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config['training']['batch_size']
        )
        
        # 初始化模型
        model = MultiStepLSTM(
            input_size=config['lstm_params']['input_size'],
            hidden_size=config['lstm_params']['hidden_size'],
            num_layers=config['lstm_params']['num_layers'],
            output_size=config['lstm_params']['output_size']
        )
        
        # 训练模型
        trained_model, history, val_loss = train_model(
            model, 
            train_loader, 
            test_loader, 
            config
        )
        
        # 在测试集上进行预测
        trained_model.eval()
        device = next(trained_model.parameters()).device
        with torch.no_grad():
            predictions = trained_model(X_test_t.to(device)).cpu().numpy()
        
        # 反归一化预测值 
        # 目标值在特征中的位置
        target_idx = data.columns.get_loc(config['target_col'])
        target_scaler = scalers[target_idx]
        
        # 反归一化预测值
        predictions_denorm = target_scaler.inverse_transform(predictions)
        
        # 反归一化实际值
        actuals_denorm = target_scaler.inverse_transform(y_test)
        
        # 计算MSE
        mse = mean_squared_error(actuals_denorm.flatten(), predictions_denorm.flatten())
        
        # 获取时间戳
        start_idx = test_index[0] + config['sequence_length']
        end_idx = start_idx + config['prediction_length']
        timestamps = data.index[start_idx:end_idx]
        
        # 保存结果
        results['fold'].append(fold)
        results['mse'].append(mse)
        results['predictions'].append(predictions_denorm)
        results['actuals'].append(actuals_denorm)
        results['timestamps'].append(timestamps)
        
        # 绘制并保存本折结果
        plot_fold_results(fold, timestamps, actuals_denorm, predictions_denorm, mse, config)
        
        fold += 1
    
    return results

def plot_fold_results(fold, timestamps, actuals, predictions, mse, config):
    """绘制并保存单折结果"""
    plt.figure(figsize=(14, 7))
    
    # 绘制预测序列
    for i in range(actuals.shape[0]):
        # 实际值
        plt.plot(
            timestamps, 
            actuals[i], 
            color='blue', 
            alpha=0.3 if i > 0 else 0.7,
            label='实际值' if i == 0 else None
        )
        
        # 预测值
        plt.plot(
            timestamps, 
            predictions[i], 
            color='red', 
            alpha=0.3 if i > 0 else 0.7,
            linestyle='--',
            label='预测值' if i == 0 else None
        )
    
    # 添加平均线
    mean_actual = np.mean(actuals, axis=0)
    mean_pred = np.mean(predictions, axis=0)
    
    plt.plot(timestamps, mean_actual, color='darkblue', linewidth=3, label='平均实际值')
    plt.plot(timestamps, mean_pred, color='darkred', linewidth=3, linestyle='--', label='平均预测值')
    
    # 设置标题和标签
    plt.title(f'第 {fold} 折预测结果 (MSE: {mse:.2f})', fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel(config['target_col'], fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{config['output_dir']}/fold_{fold}_prediction.png", dpi=300)
    plt.close()
    
    print(f"第 {fold} 折结果图已保存")

# ========== 6. 执行交叉验证 ==========
results = time_series_cross_validation(X, y, data, CONFIG)

# ========== 7. 最终结果分析 ==========
def analyze_final_results(results, config):
    """分析并展示最终结果"""
    # 计算平均MSE
    avg_mse = np.mean(results['mse'])
    
    # 创建汇总图
    plt.figure(figsize=(14, 8))
    
    for i, fold in enumerate(results['fold']):
        # 绘制每个fold的平均预测和实际值
        mean_actual = np.mean(results['actuals'][i], axis=0)
        mean_pred = np.mean(results['predictions'][i], axis=0)
        
        plt.plot(
            results['timestamps'][i], 
            mean_actual, 
            color=plt.cm.tab10(i),
            label=f'Fold {fold} 实际值'
        )
        
        plt.plot(
            results['timestamps'][i], 
            mean_pred, 
            color=plt.cm.tab10(i),
            linestyle='--',
            label=f'Fold {fold} 预测值'
        )
    
    plt.title(f'时间序列交叉验证结果 (平均MSE: {avg_mse:.2f})', fontsize=16)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel(config['target_col'], fontsize=12)
    plt.legend(ncol=2, fontsize=10)
    plt.grid(True)
    plt.xticks(rotation=45)
    
    # 保存汇总图
    plt.tight_layout()
    plt.savefig(f"{config['output_dir']}/all_folds_summary.png", dpi=300)
    plt.close()
    
    # 创建MSE对比图
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(results['mse'])), results['mse'], color=plt.cm.viridis(np.linspace(0, 1, len(results['mse']))))
    plt.axhline(y=avg_mse, color='r', linestyle='--', label=f'平均MSE: {avg_mse:.2f}')
    plt.title('各折验证MSE分数', fontsize=14)
    plt.xlabel('折数', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.legend()
    plt.grid(True, axis='y')
    
    # 保存MSE图
    plt.tight_layout()
    plt.savefig(f"{config['output_dir']}/mse_scores.png", dpi=300)
    plt.close()
    
    # 保存结果到CSV
    results_df = pd.DataFrame({
        'Fold': results['fold'],
        'MSE': results['mse']
    })
    results_df.to_csv(f"{config['output_dir']}/results_summary.csv", index=False)
    
    print(f"\n{'='*50}")
    print(f"交叉验证完成，平均MSE: {avg_mse:.4f}")
    print(f"结果已保存至目录: {config['output_dir']}")
    print(f"{'='*50}")
    
    # 打印各折MSE
    print("\n各折MSE分数:")
    for fold, mse in zip(results['fold'], results['mse']):
        print(f"折 {fold}: MSE = {mse:.4f}")

# 分析最终结果
analyze_final_results(results, CONFIG)