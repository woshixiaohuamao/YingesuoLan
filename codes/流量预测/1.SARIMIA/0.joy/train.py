"""
SARIMA气体产量预测 - 训练模块
训练SARIMA模型并保存
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import warnings
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import time


# 配置参数
DATA_FILE_PATH = r''
DATE_COLUMN = 'timestamp'
TARGET_COLUMN = '瞬时流量'
SAMPLING_INTERVAL = 30  # 重采样间隔
OUTPUT_DIR = 'sarima_results'
MODEL_FILE = 'sarima_model_6_13.pkl'

# SARIMA参数 (p,d,q)(P,D,Q,s)
SARIMA_ORDER = (0, 2, 1)
SEASONAL_ORDER = (1, 1, 0, 48)  # 季节性周期为一天(144个10分钟)


def setup_output_directory(output_dir):
    """创建输出目录"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    return output_dir


def load_and_process_data(file_path):
    """加载并预处理数据"""
    try:
        # 读取原始数据
        df = pd.read_excel(file_path)
        print("成功加载数据文件!")
        
        # 检查并转换日期列
        if DATE_COLUMN not in df.columns or TARGET_COLUMN not in df.columns:
            raise ValueError(f"数据缺少必要的'{DATE_COLUMN}'或'{TARGET_COLUMN}'列")
        
        # 转换日期并设置为索引
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        df.set_index(DATE_COLUMN, inplace=True)
        
        # 重采样为指定间隔
        data = df[TARGET_COLUMN].resample(f'{SAMPLING_INTERVAL}T').mean()
        
        # 填充可能的缺失值
        data = data.ffill()
        
        # 计算每天有多少个数据点
        points_per_day = 24 * 60 // SAMPLING_INTERVAL
        
        print(f"成功处理数据! 数据范围: {data.index.min()} 到 {data.index.max()}")
        print(f"总计 {len(data)} 个{SAMPLING_INTERVAL}分钟间隔的数据点")
        print(f"每天有 {points_per_day} 个数据点")
        
        return data, points_per_day
        
    except Exception as e:
        print(f"数据加载或处理失败: {e}")
        import traceback
        traceback.print_exc()
        raise

def save_data_to_csv(data, filename, output_dir):
    """保存数据到CSV文件"""
    # 确保数据是DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    # 保存到xlsx文件
    filepath = os.path.join(output_dir, filename)
    data.to_excel(filepath)
    print(f"数据已保存至: {filepath}")
    return filepath


def split_data(data, points_per_day):
    """将数据分为训练集和测试集（最后一天）"""
    # 从数据集中分离出最后一天作为测试集
    test_size = points_per_day  # 1天的数据点数量
    test_data = data[-test_size:]
    train_data = data[:-test_size]
    
    print(f"训练集大小: {len(train_data)} 个点 (约 {len(train_data)/points_per_day:.1f} 天)")
    print(f"测试集大小: {len(test_data)} 个点 (1天)")
    
    return train_data, test_data


def fit_sarima_model(train_data, points_per_day):
    """拟合SARIMA模型"""
    print(f"使用SARIMA参数: {SARIMA_ORDER}{SEASONAL_ORDER}")
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            # 构建并拟合模型
            model = SARIMAX(
                train_data, 
                order=SARIMA_ORDER,
                seasonal_order=SEASONAL_ORDER,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            fitted_model = model.fit(disp=False)
            print("模型拟合成功")
            return fitted_model
            
        except Exception as e:
            print(f"模型拟合失败: {e}")
            import traceback
            traceback.print_exc()
            return None


def save_model(model, output_dir, filename=MODEL_FILE):
    """保存模型到文件"""
    try:
        model_path = os.path.join(output_dir, filename)
        joblib.dump(model, model_path)
        print(f"模型已保存至: {model_path}")
        return model_path
    except Exception as e:
        print(f"保存模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    start_time = time.time()
    
    # 设置输出目录
    output_dir = setup_output_directory(OUTPUT_DIR)
    
    # 加载并处理数据
    data, points_per_day = load_and_process_data(DATA_FILE_PATH)
    
    # 分割数据为训练集和测试集
    train_data, test_data = split_data(data, points_per_day)
    
    # 保存训练集和测试集数据
    save_data_to_csv(train_data, 'train_data.xlsx', output_dir)
    save_data_to_csv(test_data, 'test_data.xlsx', output_dir)
    
    # 保存点数信息到文件，方便测试脚本读取
    with open(os.path.join(output_dir, 'model_info.txt'), 'w') as f:
        f.write(f"POINTS_PER_DAY={points_per_day}\n")
        f.write(f"SARIMA_ORDER={SARIMA_ORDER}\n")
        f.write(f"SEASONAL_ORDER={SEASONAL_ORDER}\n")
        f.write(f"SAMPLING_INTERVAL={SAMPLING_INTERVAL}\n")
    
    # 拟合SARIMA模型
    model = fit_sarima_model(train_data, points_per_day)
    
    if model is not None:
        # 保存模型
        save_model(model, output_dir)
    else:
        print("模型拟合失败，无法保存模型")
    
    # 打印运行时间
    end_time = time.time()
    print(f"\n程序运行总时间: {(end_time - start_time):.2f} 秒")
    print(f"结果已保存到目录: {output_dir}")


if __name__ == "__main__":
    main() 