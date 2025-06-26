import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from joblib import dump, load
import os


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, n_steps_out):
        self.seq_length = seq_length
        self.n_steps_out = n_steps_out
        self.X = []
        self.y = []
        
        for i in range(len(data) - seq_length - n_steps_out + 1):
            self.X.append(data[i:i+seq_length])
            self.y.append(data[i+seq_length:i+seq_length+n_steps_out])
        
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def test_model():
    FILE_PATH = './模型评估/模型评估数据集.xlsx'
    TARGET_COLUMN = '瞬时流量'
    SEQ_LENGTH = 144
    N_STEPS_OUT = 60
    MODEL_PATH = 'lstm_144_60.pth'
    SCALER_PATH = 'scaler_model.joblib'
    BATCH_SIZE = 16
    

    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件 '{MODEL_PATH}' 不存在")
        return
    if not os.path.exists(SCALER_PATH):
        print(f"错误: 归一化文件 '{SCALER_PATH}' 不存在")
        return
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    

    df = pd.read_excel(FILE_PATH)
    data = df[TARGET_COLUMN].values.reshape(-1, 1)
    

    scaler = load(SCALER_PATH)
    scaled_data = scaler.transform(data)  
    

    test_dataset = TimeSeriesDataset(scaled_data, SEQ_LENGTH, N_STEPS_OUT)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    

    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    print("模型加载成功")
    

    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            pred = model(batch_X)
            
            all_predictions.append(pred.cpu().numpy())
            all_targets.append(batch_y.numpy())
    

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    

    predictions_inv = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    targets_inv = scaler.inverse_transform(targets.reshape(-1, 1)).reshape(targets.shape)
    

    r2_scores = []
    mse_scores = []
    mae_scores = []
    
    for step in range(N_STEPS_OUT):
        step_pred = predictions_inv[:, step]
        step_target = targets_inv[:, step]
        
        r2 = r2_score(step_target, step_pred)
        mse = mean_squared_error(step_target, step_pred)
        mae = mean_absolute_error(step_target, step_pred)
        
        r2_scores.append(r2)
        mse_scores.append(mse)
        mae_scores.append(mae)
    

    print("\n模型整体性能:")
    print(f"平均 R²: {np.mean(r2_scores):.4f}")
    print(f"平均 MSE: {np.mean(mse_scores):.4f}")
    print(f"平均 MAE: {np.mean(mae_scores):.4f}")
    
 
    print("\n每个预测时间步的性能:")
    print(f"{'时间步':<8}{'R²':<10}{'MSE':<15}{'MAE':<10}")
    for step in range(N_STEPS_OUT):
        print(f"Step {step+1:<3} | {r2_scores[step]:<8.4f} | {mse_scores[step]:<12.4f} | {mae_scores[step]:<8.4f}")
    

    plt.rcParams['font.sans-serif'] = ['SimSun']       
    plt.figure(figsize=(14, 10))
    

    plt.subplot(2, 1, 1)
    plt.plot(range(1, N_STEPS_OUT+1), r2_scores, 'bo-', label='R²')
    plt.plot(range(1, N_STEPS_OUT+1), np.array(mse_scores)/max(mse_scores), 'ro-', label='归一化MSE')
    plt.plot(range(1, N_STEPS_OUT+1), np.array(mae_scores)/max(mae_scores), 'go-', label='归一化MAE')
    plt.title('各预测步长性能指标')
    plt.xlabel('预测步长')
    plt.ylabel('指标值')
    plt.legend()
    plt.grid(True)
    

    plt.subplot(2, 1, 2)
    for i in range(3):
        plt.plot(range(1, N_STEPS_OUT+1), targets_inv[i], 'o-', label=f'样本{i+1}真实值')
        plt.plot(range(1, N_STEPS_OUT+1), predictions_inv[i], 'x--', label=f'样本{i+1}预测值')
    
    plt.title('前3个样本的预测结果')
    plt.xlabel('预测步长')
    plt.ylabel('值')
    plt.legend()
    plt.grid(True)    
    plt.tight_layout()
    plt.savefig('./模型评估/model_performance.png')
    plt.show()
    

    result_df = pd.DataFrame({
        '时间步': range(1, N_STEPS_OUT+1),
        'R2': r2_scores,
        'MSE': mse_scores,
        'MAE': mae_scores
    })
    result_df.to_excel('./模型评估/model_performance_metrics.xlsx', index=False)
    print("\n性能指标已保存到 'model_performance_metrics.xlsx'")

if __name__ == "__main__":
    test_model()