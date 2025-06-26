import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import os

# 配置路径 
MODEL_PATH = ''  # 训练好的模型路径
TEST_DATA_PATH = ''  # 测试数据路径
OUTPUT_DIR = 'evaluation_results'  # 输出目录
PRED_LEN = 48
TARGET_COL = '瞬时流量'  # 目标列名

def load_model(model_path):
    """加载训练好的模型"""
    try:
        model = joblib.load(model_path)
        print(f"成功加载模型: {model_path}")
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_model_and_data(model_path, test_data_path):
    """加载训练好的模型和测试数据"""
    print(f"加载模型: {model_path}")
    model = load_model(model_path)
    
    print(f"加载测试数据: {test_data_path}")
    test = pd.read_excel(test_data_path, parse_dates=['timestamp'], index_col='timestamp')
    
    return model, test

def rolling_forecast(model, test, steps_per_forecast=12):
    """
    执行滚动预测以处理长测试集
    :param model: 已训练的SARIMAX模型
    :param test: 测试数据集
    :param steps_per_forecast: 每次预测的步数
    :return: 预测值序列和置信区间
    """
    print(f"开始滚动预测 (测试集长度={len(test)}, 每批次={steps_per_forecast}步)")
    
    # 初始化存储结构
    all_preds = []
    all_conf_int = []
    
    # 复制模型以便更新状态
    updated_model = model
    
    # 分批次预测
    for start in range(0, len(test), steps_per_forecast):
        end = min(start + steps_per_forecast, len(test))
        steps = end - start
        
        # 进行预测
        forecast = updated_model.get_forecast(steps=steps)
        preds = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        # 存储结果
        all_preds.append(preds)
        all_conf_int.append(conf_int)
        
        # 使用真实值更新模型状态（更准确）
        if end < len(test):
            # 获取下一批需要更新的真实值
            update_data = test.iloc[start:end]
            
            # 更新模型
            updated_model = updated_model.append(update_data, refit=False)
            
            print(f"已预测 {end}/{len(test)} 步，更新模型状态...")
    
    # 合并结果
    final_preds = pd.concat(all_preds)
    final_conf_int = pd.concat(all_conf_int)
    
    return final_preds, final_conf_int

def calculate_metrics(test, predictions):
    """计算评估指标"""
    # 确保对齐
    aligned_test = test.reindex(predictions.index)
    
    # 计算指标
    mse = mean_squared_error(aligned_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(aligned_test, predictions)
    mape = np.mean(np.abs((aligned_test.values - predictions.values) / aligned_test.values)) * 100
    
    metrics = {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'MAPE': float(mape),
        'Test_size': len(aligned_test),
        'Predictions_size': len(predictions)
    }
    
    print("\n评估指标结果:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    
    return metrics

def save_results(metrics, predictions, conf_int, test, output_dir='results'):
    """保存所有结果"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存评估指标
    metrics_path = os.path.join(output_dir, 'sarima_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n评估指标已保存至: {metrics_path}")
    
    # 2. 保存预测值
    predictions_df = pd.DataFrame({
        'date': predictions.index,
        'actual_value': test[TARGET_COL].values,
        'prediction': predictions.values,
        'lower_bound': conf_int.iloc[:, 0].values,
        'upper_bound': conf_int.iloc[:, 1].values
    })
    predictions_path = os.path.join(output_dir, 'sarima_predictions.xlsx')
    predictions_df.to_excel(predictions_path, index=False)
    print(f"预测结果已保存至: {predictions_path}")
    
    # 3. 保存可视化图表
    plt.figure(figsize=(14, 8))
    # 设置中文字体和解决负号显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
        
    # 绘制测试集和预测值
    plt.plot(test.index, test[TARGET_COL], 'b-', label='真实值', alpha=0.7)
    plt.plot(predictions.index, predictions, 'r--', label='预测值', linewidth=1.5)
    
    # 绘制置信区间
    plt.fill_between(predictions.index, 
                    conf_int.iloc[:, 0],
                    conf_int.iloc[:, 1], 
                    color='gray', alpha=0.2, label='95% 置信区间')
    
    plt.title('SARIMAX模型预测结果 vs 真实值', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('数值', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 优化布局
    plt.tight_layout()
    
    # 保存图像
    plot_path = os.path.join(output_dir, 'sarima_forecast_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存至: {plot_path}")
    
    # 显示图表
    plt.show()
    
    # 4. 保存残差图
    residuals = test[TARGET_COL] - predictions
    plt.figure(figsize=(12, 6))
    plt.plot(residuals)
    plt.title('预测残差', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('残差值', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    residuals_path = os.path.join(output_dir, 'sarima_residuals_plot.png')
    plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
    print(f"残差图已保存至: {residuals_path}")
    plt.show()
    
    return {
        'metrics': metrics_path,
        'predictions': predictions_path,
        'forecast_plot': plot_path,
        'residuals_plot': residuals_path
    }

def main():

    # 1. 加载模型和测试数据
    model, test = load_model_and_data(MODEL_PATH, TEST_DATA_PATH)
    
    # 2. 执行滚动预测
    predictions, conf_int = rolling_forecast(model, test, steps_per_forecast=PRED_LEN)
    
    # 3. 计算评估指标
    metrics = calculate_metrics(test[TARGET_COL], predictions)
    
    # 4. 保存所有结果
    results = save_results(metrics, predictions, conf_int, test, OUTPUT_DIR)
    
    print("\n评估完成! 所有结果已保存至目录:", OUTPUT_DIR)

if __name__ == "__main__":
    
    main()