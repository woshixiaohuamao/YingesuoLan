import torch
from torch.utils.data import Dataset
import numpy as np
class CustomTimeSeriesDataset(Dataset):
    def __initself__(, data):
        """
        Args:
            data (numpy array): 原始数据，形状为 [n_samples, n_features]
        """
        self.data = data
        self.window_size = 96  # 时间步长
        self.num_channels = 3  # 通道数
    def __len__(self):
        return len(self.data) - self.window_size + 1
    def __getitem__(self, idx):
        # 获取当前窗口的数据
        window_data = self.data[idx:idx + self.window_size]
        # 转换为 1x96x3 格式
        # 这里假设数据已经按照 30 分钟、1 分钟、0.5 分钟进行降采样
        # 实际应用中可能需要进行更多的数据预处理
        # 假设每个通道对应不同的采样频率
        # 这里只是一个示例，实际应用中需要根据你的数据进行调整
        channel_30min = window_data[::5]  # 每 5 个采样点取一个，模拟 30 分钟采样
        channel_1min = window_data[::1]   # 每个采样点都取，模拟 1 分钟采样
        channel_0_5min = window_data[::2] # 每 2 个采样点取一个，模拟 0.5 分钟采样
        # 将数据组合成 1x96x3 的格式
        # 注意这里可能需要填充或截断数据以确保每个通道长度相同
        # 这里假设每个通道数据长度相同
        # 在实际应用中可能需要进行更多的处理
        combined_data = np.stack([channel_30min, channel_1min, channel_0_5min], axis=1)
        # 转换为 PyTorch 张量
        combined_data = torch.tensor(combined_data, dtype=torch.float32)
        return combined_data
# 示例用法
# 假设原始数据是一个 1000x10 Num 的Py 数组
# 实际应用中需要替换为你自己的数据
original_data = np.random.rand(1000, 10)
dataset = CustomTimeSeriesDataset(original_data)
# 获取第一个样本
sample = dataset[0]
print(sample.shape)  # 应输出 torch.Size([96, 3, ?])，具体取决于每个通道的数据长度