import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

# 1: 加载数据集
def load_data(file_path):
    df = pd.read_excel(file_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    
    # 确保时间序列连续（30分钟频率）
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='30T')
    df = df.reindex(full_range)
    
    # 填充缺失值 - 使用前向填充，然后后向填充
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # 添加额外的时间特征
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    
    # 周期性编码
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

# 2: 构建数据集，用于Transformer训练（滑动窗口）
class MultiStepTimeSeriesDataset(Dataset):
    def __init__(self, series, features, window_size, forecast_horizon=48):
        self.X = []
        self.y = []
        self.features = []
        
        for i in range(len(series) - window_size - forecast_horizon):
            # 历史实际值序列
            seq = series[i:i+window_size]
            
            # 预测时刻的特征（使用窗口结束时刻的特征）
            feat = features.iloc[i+window_size].values
            
            # 预测目标（未来48个时间点）
            target = series[i+window_size:i+window_size+forecast_horizon]
            
            self.X.append(seq)
            self.features.append(feat)
            self.y.append(target)
            
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.features[idx], self.y[idx]

# 3: 定义Transformer模型（多步预测）
class MultiStepTransformerModel(nn.Module):
    def __init__(self, input_size=1, feature_size=0, d_model=128, nhead=8, num_layers=3, forecast_horizon=48, dropout=0.1):
        super(MultiStepTransformerModel, self).__init__()
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        
        # 时间序列编码层
        self.input_linear = nn.Linear(input_size, d_model)
        
        # 特征编码层
        self.feature_linear = nn.Linear(feature_size, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 解码器（预测未来48个时间点）
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, forecast_horizon)
        )

    def forward(self, src, features):
        # 编码时间序列
        src = self.input_linear(src)  # (batch_size, seq_len, d_model)
        
        # 编码特征并扩展为与时间序列相同的维度
        feat_encoded = self.feature_linear(features).unsqueeze(1)  # (batch_size, 1, d_model)
        feat_encoded = feat_encoded.expand(-1, src.size(1), -1)  # (batch_size, seq_len, d_model)
        
        # 合并时间序列和特征信息
        combined = src + feat_encoded
        
        # Transformer处理
        encoded = self.transformer_encoder(combined)  # (batch_size, seq_len, d_model)
        
        # 使用最后时间步的输出进行预测
        last_output = encoded[:, -1, :]  # (batch_size, d_model)
        
        # 预测未来48个时间点
        output = self.output_layer(last_output)  # (batch_size, forecast_horizon)
        
        return output

# 4: 训练Transformer模型（多步预测）
def train_transformer_model(data, feature_columns, window_size=336, forecast_horizon=48, 
                           epochs=100, lr=1e-3, patience=15, batch_size=64):
    # 准备数据
    series = data['Actual'].values
    features = data[feature_columns]
    
    # 创建数据集
    dataset = MultiStepTimeSeriesDataset(series, features, window_size, forecast_horizon)
    
    # 按时间顺序划分数据集
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.15)
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_size = len(feature_columns)
    
    model = MultiStepTransformerModel(
        feature_size=feature_size,
        d_model=128,
        nhead=8,
        num_layers=3,
        forecast_horizon=forecast_horizon
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    # EarlyStopping初始化
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        if early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        for X_seq, X_feat, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            X_seq = X_seq.unsqueeze(-1).to(device) 
            X_feat = X_feat.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_seq, X_feat)
            loss = criterion(output, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            epoch_train_loss += loss.item()
        
        train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for X_seq, X_feat, y_batch in val_loader:
                X_seq = X_seq.unsqueeze(-1).to(device)
                X_feat = X_feat.to(device)
                y_batch = y_batch.to(device)
                
                output = model(X_seq, X_feat)
                loss = criterion(output, y_batch)
                epoch_val_loss += loss.item()
        
        val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss) 
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # EarlyStopping检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_transformer_model.pth')
            print(f"Validation loss improved. Saving model...")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve. Patience: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                early_stop = True

    # 加载最佳模型
    model.load_state_dict(torch.load('best_transformer_model.pth'))
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png', bbox_inches='tight')
    plt.show()
    
    return model, test_loader, train_loader, val_loader

# 5: 提取Transformer中间表示作为XGBoost特征
def extract_transformer_features(model, loader, device):
    model.eval()
    transformer_features = []
    targets = []
    original_features = []
    
    with torch.no_grad():
        for X_seq, X_feat, y_batch in loader:
            X_seq = X_seq.unsqueeze(-1).to(device)
            X_feat = X_feat.to(device)
            
            # 获取Transformer的编码
            src = model.input_linear(X_seq)
            feat_encoded = model.feature_linear(X_feat).unsqueeze(1)
            feat_encoded = feat_encoded.expand(-1, src.size(1), -1)
            combined = src + feat_encoded
            encoded = model.transformer_encoder(combined)
            
            # 使用最后时间步的输出作为特征
            features = encoded[:, -1, :].cpu().numpy()
            
            transformer_features.append(features)
            targets.append(y_batch.numpy())
            original_features.append(X_feat.cpu().numpy())
    
    transformer_features = np.vstack(transformer_features)
    targets = np.vstack(targets)
    original_features = np.vstack(original_features)
    
    return transformer_features, original_features, targets

# 6: 多步预测评估
def evaluate_multi_step_predictions(y_true, y_pred, horizon=48):
    # 计算每个时间步的误差
    step_errors = []
    for i in range(horizon):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        step_errors.append((mse, mae))
    
    # 计算整体误差
    total_mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    total_mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    
    print(f"\nOverall Performance:")
    print(f"MSE: {total_mse:.4f}")
    print(f"MAE: {total_mae:.4f}")
    
    print("\nPerformance by Forecast Horizon:")
    for i, (mse, mae) in enumerate(step_errors):
        print(f"Step {i+1}: MSE={mse:.4f}, MAE={mae:.4f}")
    
    # 可视化第一个预测点的性能
    plt.figure(figsize=(14, 6))
    plt.plot(y_true[0], label='Actual', marker='o', color='dodgerblue')
    plt.plot(y_pred[0], label='Predicted', marker='x', color='orange')
    plt.title(f"Actual vs Predicted Values (First Sample)")
    plt.xlabel("Time Step (30-min intervals)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig('first_sample_prediction.png', bbox_inches='tight')
    plt.show()
    
    # 可视化所有样本的平均预测性能
    mean_actual = y_true.mean(axis=0)
    mean_pred = y_pred.mean(axis=0)
    
    plt.figure(figsize=(14, 6))
    plt.plot(mean_actual, label='Average Actual', marker='o', color='dodgerblue')
    plt.plot(mean_pred, label='Average Predicted', marker='x', color='orange')
    plt.title(f"Average Actual vs Predicted Values (All Samples)")
    plt.xlabel("Time Step (30-min intervals)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig('average_prediction.png', bbox_inches='tight')
    plt.show()
    
    # 误差随时间步的变化
    mse_by_step = [mse for mse, _ in step_errors]
    mae_by_step = [mae for _, mae in step_errors]
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(mse_by_step, marker='o', color='purple')
    plt.title("MSE by Forecast Horizon")
    plt.xlabel("Time Step")
    plt.ylabel("MSE")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(mae_by_step, marker='o', color='green')
    plt.title("MAE by Forecast Horizon")
    plt.xlabel("Time Step")
    plt.ylabel("MAE")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('error_by_horizon.png', bbox_inches='tight')
    plt.show()
    
    return total_mse, total_mae

# 主流程
def main():
    # 1. 加载数据
    file_path = ""
    data = load_data(file_path)
    
    # 2. 定义特征列（根据您的数据集）
    feature_columns = [
        'predict', 'hour', 'weekday', 'day', 'residual', 'day_part',
        'residual_lag1', 'SARIMA_lag1', 'residual_lag2', 'SARIMA_lag2',
        'residual_lag3', 'SARIMA_lag3', 'residual_lag24', 'SARIMA_lag24',
        'residual_lag48', 'SARIMA_lag48', 'rolling_8h_mean',
        'sin_hour', 'cos_hour', 'sin_day', 'cos_day'
    ]
    
    # 3. 训练Transformer模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window_size = 7 * 48  # 7天历史数据 (7*48=336个30分钟间隔)
    forecast_horizon = 48  # 预测未来48步（一天）
    
    print("Starting Transformer training...")
    transformer_model, test_loader, train_loader, val_loader = train_transformer_model(
        data, 
        feature_columns,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        epochs=100,
        lr=1e-3,
        patience=15,
        batch_size=64
    )
    
    # 4. 提取特征
    print("\nExtracting Transformer features...")
    X_train_trans, X_train_feat, y_train = extract_transformer_features(transformer_model, train_loader, device)
    X_val_trans, X_val_feat, y_val = extract_transformer_features(transformer_model, val_loader, device)
    X_test_trans, X_test_feat, y_test = extract_transformer_features(transformer_model, test_loader, device)
    
    # 合并Transformer特征和原始特征
    X_train = np.hstack([X_train_trans, X_train_feat])
    X_val = np.hstack([X_val_trans, X_val_feat])
    X_test = np.hstack([X_test_trans, X_test_feat])
    
    # 5. 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. 训练XGBoost模型（为每个时间步训练一个模型）
    print("\nTraining XGBoost models for each time step...")
    models = []
    predictions = np.zeros_like(y_test)
    # 对于预测的48个点，每个点单独训练一个模型
    for step in tqdm(range(forecast_horizon), desc="Training XGBoost models"):
        # 当前时间步的目标
        y_train_step = y_train[:, step]
        y_val_step = y_val[:, step]
        
        # 超参数调优
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        
        grid_search = GridSearchCV(
            estimator=xgb_model, 
            param_grid=param_grid, 
            cv=TimeSeriesSplit(n_splits=3),
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_scaled, y_train_step)
        best_model = grid_search.best_estimator_
        
        # 在验证集上微调
        best_model.fit(
            X_train_scaled, y_train_step,
            eval_set=[(X_val_scaled, y_val_step)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        models.append(best_model)
        predictions[:, step] = best_model.predict(X_test_scaled)
    
    # 7. 评估模型性能
    print("\nEvaluating model performance...")
    evaluate_multi_step_predictions(y_test, predictions, horizon=forecast_horizon)
    
    # 8. 保存模型
    import joblib
    joblib.dump(models, 'xgboost_models.pkl')
    torch.save(transformer_model.state_dict(), 'transformer_model.pth')
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    print("Models saved successfully.")

if __name__ == "__main__":
    main()