import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# 设置图形风格
sns.set_style("whitegrid")


def setup_font_for_chinese():
    GENERATE_PLOTS = True
    """设置中文字体"""
    if not GENERATE_PLOTS:
        return
        
    try:
        from matplotlib.font_manager import FontManager
        
        # 获取所有可用字体
        fm = FontManager()
        mat_fonts = set([f.name for f in fm.ttflist])
        
        # 优先使用这些中文字体
        chinese_fonts = ['Heiti TC', 'Songti SC', 'STHeiti', 'STSong', 'AR PL UMing CN', 
                        'Hiragino Sans GB', 'Apple LiGothic Medium', 'SimHei', 'WenQuanYi Zen Hei']
        
        # 检查哪些字体可用
        available_fonts = [f for f in chinese_fonts if f in mat_fonts]
        
        if available_fonts:
            chinese_font = available_fonts[0]
            print(f"使用中文字体: {chinese_font}")
            plt.rcParams['font.family'] = chinese_font
        else:
            print("未找到中文字体，使用系统默认字体")
            plt.rcParams['font.family'] = ['sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
        # 确保可以显示中文标签
        try:
            plt.rcParams['font.sans-serif'] = ['Songti SC', 'Heiti TC', 'STHeiti', 'Arial Unicode MS'] + plt.rcParams['font.sans-serif']
        except:
            pass
            
    except Exception as e:
        print(f"设置中文字体时出错: {e}")
        print("继续使用默认字体")

# 加载数据
def load_data(file_path):
    """加载并预处理数据"""
    df = pd.read_excel(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    data = df['瞬时流量'].resample('30T').mean().ffill()
    return data

# 加载模型
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

# 生成预测
def generate_predictions(model, data, train_size):
    """
    生成样本内拟合值和样本外预测值
    """
    # 样本内拟合值（训练集）
    train_data = data[:train_size]
    train_fit = model.get_prediction(start=0, end=train_size-1, dynamic=False)
    train_pred = train_fit.predicted_mean
    
    # 为训练集预测值分配正确的时间索引
    train_pred = pd.Series(train_pred.values, index=train_data.index)
    
    # 样本外预测值（测试集）
    test_data = data[train_size:]
    
    # 计算预测开始位置（训练集结束的下一个时间点）
    forecast_start = train_data.index[-1] + pd.Timedelta(minutes=30)
    

    test_index = pd.date_range(
        start=forecast_start,
        periods=len(test_data),
        freq='30T'
    )
    
    test_forecast = model.get_forecast(steps=len(test_data))
    test_pred = test_forecast.predicted_mean
    

    test_pred = pd.Series(test_pred.values, index=test_index)
    full_pred = pd.concat([train_pred, test_pred])
    
    return train_pred, test_pred, full_pred

# 计算评估指标
def calculate_metrics(true, pred, set_name):
    """计算各种评估指标"""
    mask = true != 0  
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, pred)
    mape = mean_absolute_percentage_error(true, pred) * 100
 
    return {
        '数据集': set_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE(%)': mape,
        '数据点数量': len(true)
    }

# 保存结果
def save_results(data, full_pred, metrics_df, output_dir):
    """保存预测结果和评估指标"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'timestamp': full_pred.index,  
        'true_value': data.reindex(full_pred.index).values,  
        'predicted_value': full_pred.values
    })
    
    # 添加预测类型列
    test_metrics = [m for m in metrics_df if m['数据集'] == '测试集'][0]
    test_data_points = test_metrics['数据点数量']
    train_size = len(data) - test_data_points
    results_df['type'] = '训练集'
    results_df.loc[train_size:, 'type'] = '测试集'
    
    # 保存结果
    results_path = os.path.join(output_dir, 'sarima_predictions_results.xlsx')
    results_df.to_excel(results_path, index=False)
    
    # 保存评估指标
    metrics_path = os.path.join(output_dir, 'sarima_evaluation_metrics.xlsx')
    pd.DataFrame(metrics_df).to_excel(metrics_path, index=False)
    
    print(f"预测结果已保存至: {results_path}")
    print(f"评估指标已保存至: {metrics_path}")
    
    return results_df

# 绘制结果图表
def plot_results(data, full_pred, train_pred, test_pred, metrics_df, output_dir):
    """绘制并保存结果图表"""
    print("\n时间索引调试信息:")
    print(f"原始数据开始时间: {data.index[0]}, 结束时间: {data.index[-1]}")
    print(f"完整预测开始时间: {full_pred.index[0]}, 结束时间: {full_pred.index[-1]}")
    print(f"训练预测开始时间: {train_pred.index[0]}, 结束时间: {train_pred.index[-1]}")
    print(f"测试预测开始时间: {test_pred.index[0]}, 结束时间: {test_pred.index[-1]}")

    train_size = len(train_pred)
    test_size = len(test_pred)
    
    # 整体视图
    plt.figure(figsize=(16, 10))
    
    # 完整数据视图
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data, 'b-', label='真实值', alpha=0.7, linewidth=1.5)
    plt.plot(full_pred.index, full_pred, 'r--', label='预测值', alpha=0.9, linewidth=1.2)
    plt.axvline(x=data.index[train_size], color='k', linestyle='--', 
                label=f'训练/测试分界 ({data.index[train_size].strftime("%Y-%m-%d")})')
    
    # 添加填充区域显示训练集和测试集
    plt.axvspan(data.index[0], data.index[train_size-1], 
                alpha=0.1, color='green', label='训练集')
    plt.axvspan(data.index[train_size], data.index[-1], 
                alpha=0.1, color='orange', label='测试集')
    
    plt.title('SARIMA 模型预测结果 - 完整视图', fontsize=16)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('瞬时流量', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 测试集详细视图
    plt.subplot(2, 1, 2)
    plt.plot(data.index[train_size:], data[train_size:], 'bo-', 
             label='真实值', alpha=0.7, markersize=4, linewidth=1.5)
    plt.plot(test_pred.index, test_pred, 'rs--', 
             label='预测值', alpha=0.9, markersize=4, linewidth=1.2)
    
    # 添加误差线
    for i in range(len(test_pred)):
        plt.plot([data.index[train_size+i], data.index[train_size+i]], 
                 [data[train_size+i], test_pred[i]], 'gray', alpha=0.4)
    
    # 添加评估指标文本
    test_metrics = [m for m in metrics_df if m['数据集'] == '测试集'][0]
    metrics_text = (f"测试集评估指标:\n"
                   f"MSE = {test_metrics['MSE']:.4f}\n"
                   f"RMSE = {test_metrics['RMSE']:.4f}\n"
                   f"MAE = {test_metrics['MAE']:.4f}\n"
                   f"MAPE = {test_metrics['MAPE(%)']:.2f}%")
    
    plt.text(0.02, 0.95, metrics_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.2))
    
    plt.title('SARIMA 模型预测结果 - 测试集详细视图', fontsize=16)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('瞬时流量', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(output_dir, 'sarima_forecast_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"预测结果图表已保存至: {plot_path}")
    
    # 单独保存测试集图表
    plt.figure(figsize=(14, 7))
    plt.plot(data.index[train_size:], data[train_size:], 'bo-', 
             label='真实值', alpha=0.7, markersize=5, linewidth=1.5)
    plt.plot(test_pred.index, test_pred, 'rs--', 
             label='预测值', alpha=0.9, markersize=5, linewidth=1.2)
    
    # 添加误差线
    for i in range(len(test_pred)):
        plt.plot([data.index[train_size+i], data.index[train_size+i]], 
                 [data[train_size+i], test_pred[i]], 'gray', alpha=0.4)
    
    plt.title('SARIMA 模型测试集预测结果', fontsize=16)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('瞬时流量', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加评估指标文本
    plt.text(0.02, 0.95, metrics_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.2))
    
    test_plot_path = os.path.join(output_dir, 'sarima_test_set_forecast.png')
    plt.savefig(test_plot_path, dpi=300, bbox_inches='tight')
    print(f"测试集详细图表已保存至: {test_plot_path}")

# 主函数
def main():
    # 配置参数
    DATA_PATH = ''
    MODEL_PATH = ''
    OUTPUT_DIR = 'sarima_test_results'

    # 1. 加载数据
    print("加载数据...")
    data = load_data(DATA_PATH)
    
    # 确定训练集大小（假设测试集大小为1天）
    points_per_day = 24 * 60 // 30  # 30分钟间隔
    train_size = len(data) - points_per_day
    
    setup_font_for_chinese()
    # 2. 加载模型
    print("加载模型...")
    model = load_model(MODEL_PATH)
    
    # 3. 生成预测
    print("生成预测...")
    train_pred, test_pred, full_pred = generate_predictions(model, data, train_size)
    
    # 4. 计算评估指标
    print("计算评估指标...")
    train_metrics = calculate_metrics(data[:train_size], train_pred, '训练集')
    test_metrics = calculate_metrics(data[train_size:], test_pred, '测试集')
    metrics_df = [train_metrics, test_metrics]
    
    # 打印评估结果
    print("\n评估结果:")
    print(f"训练集 MSE: {train_metrics['MSE']:.4f}, MAPE: {train_metrics['MAPE(%)']:.2f}%")
    print(f"测试集 MSE: {test_metrics['MSE']:.4f}, MAPE: {test_metrics['MAPE(%)']:.2f}%")
    
    # 5. 保存结果
    print("\n保存结果...")
    results_df = save_results(data, full_pred, metrics_df, OUTPUT_DIR)
    
    # 6. 绘制图表
    print("绘制图表...")
    plot_results(data, full_pred, train_pred, test_pred, metrics_df, OUTPUT_DIR)
    
    print("\n测试完成! 所有结果已保存至", OUTPUT_DIR)

if __name__ == "__main__":
    main()