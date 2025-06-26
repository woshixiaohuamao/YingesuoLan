"""
SARIMA气体产量预测 - 预测模块
使用训练好的SARIMA模型进行预测，支持输入最近3天数据预测未来1天
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import warnings
import joblib
import argparse
from datetime import datetime, timedelta
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
import traceback


# 默认配置参数
DEFAULT_OUTPUT_DIR = 'sarima_results'
DEFAULT_MODEL_FILE = 'sarima_model.pkl'
DEFAULT_INPUT_DAYS = 3  # 默认使用最近3天的数据
GENERATE_PLOTS = True
CONFIDENCE_INTERVAL = 0.95  # 95%置信区间
ALPHA = 1 - CONFIDENCE_INTERVAL  # 0.05


def setup_output_directory(output_dir):
    """创建输出目录"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    return output_dir


def setup_font_for_chinese():
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


def load_model_info(output_dir):
    """加载模型信息"""
    try:
        info = {}
        with open(os.path.join(output_dir, 'model_info.txt'), 'r') as f:
            for line in f:
                key, value = line.strip().split('=')
                if key == 'POINTS_PER_DAY':
                    info[key] = int(value)
                elif key == 'SAMPLING_INTERVAL':
                    info[key] = int(value)
                else:
                    info[key] = value
        return info
    except Exception as e:
        print(f"加载模型信息失败: {e}")
        # 设置默认值
        return {
            'POINTS_PER_DAY': 48,  # 默认每天48个点（30分钟间隔）
            'SAMPLING_INTERVAL': 30
        }


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


def load_input_data(input_file, date_column, target_column, sampling_interval):
    """加载输入数据"""
    try:
        # 读取原始数据
        df = pd.read_excel(input_file)
        print("成功加载输入数据文件!")
        
        # 检查并转换日期列
        if date_column not in df.columns or target_column not in df.columns:
            raise ValueError(f"数据缺少必要的'{date_column}'或'{target_column}'列")
        
        # 转换日期并设置为索引
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
        
        # 重采样为指定间隔
        data = df[target_column].resample(f'{sampling_interval}T').mean()
        
        # 填充可能的缺失值
        data = data.ffill()
        
        print(f"成功处理输入数据! 数据范围: {data.index.min()} 到 {data.index.max()}")
        print(f"总计 {len(data)} 个数据点")
        
        return data
        
    except Exception as e:
        print(f"数据加载或处理失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def get_recent_data(data, input_days, points_per_day):
    """获取最近几天的数据
    - input_days: 输入的几天的数据
    - points_per_day: 每天的数据点数
    """
    # 计算需要的数据点数
    n_points = input_days * points_per_day 
    
    # 如果数据不足，发出警告
    if len(data) < n_points:
        print(f"警告: 数据点数不足{n_points}个 (只有{len(data)}个)，将使用所有可用数据")
        return data
    
    # 返回最近的数据点
    return data[-n_points:] # 只截取最近的历史数据


def save_data_to_csv(data, filename, output_dir):
    """保存数据到CSV文件"""
    # 确保数据是DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    # 保存到CSV
    filepath = os.path.join(output_dir, filename)
    data.to_excel(filepath)
    print(f"数据已保存至: {filepath}")
    return filepath

def evaluate_and_save_metrics(y_true_values, y_pred_values, output_dir, timestamp):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    try:
        from sklearn.metrics import mean_absolute_percentage_error
    except ImportError:
        def mean_absolute_percentage_error(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.any(y_true) else np.inf

    mse = mean_squared_error(y_true_values, y_pred_values)
    mae = mean_absolute_error(y_true_values, y_pred_values)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true_values, y_pred_values)

    metrics_file = os.path.join(output_dir, f'metrics_{timestamp}.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"MAPE: {mape:.6f}%\n")
    print(f"评估指标已保存到 {metrics_file}")

def predict_with_95ci(model, input_data, points_per_day, output_dir, prediction_date=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\n预测未来一天数据（含95%置信区间）...")
    print(f"使用置信水平: {CONFIDENCE_INTERVAL} (alpha={ALPHA})")

    try:
        # 创建预测的日期范围
        if prediction_date is None:
            last_date = input_data.index[-1]
            start_date = last_date + timedelta(minutes=30)
        else:
            start_date = pd.to_datetime(prediction_date)

        future_dates = pd.date_range(start=start_date, periods=points_per_day, freq='30T')
        print(f"预测日期范围: {future_dates[0]} 到 {future_dates[-1]}")

        # 使用模型进行预测
        forecast_results = model.get_forecast(steps=points_per_day)
        forecast_mean = forecast_results.predicted_mean
        forecast_ci = forecast_results.conf_int(alpha=ALPHA)
        forecast_mean.index = future_dates
        forecast_ci.index = future_dates

        result_df = pd.DataFrame({
            'predicted': forecast_mean.values,
            'lower_ci': forecast_ci.iloc[:, 0].values,
            'upper_ci': forecast_ci.iloc[:, 1].values
        }, index=future_dates)

        save_data_to_csv(result_df, f'prediction_{timestamp}.xlsx', output_dir)

        # 尝试自动获取真实数据进行评估
        try:
            y_true = input_data.loc[future_dates]
            if len(y_true) != len(result_df):
                raise ValueError(f"真实数据长度({len(y_true)})必须与预测点数({len(result_df)})一致。")
            evaluate_and_save_metrics(y_true.values, result_df['predicted'].values, output_dir, timestamp)
        except KeyError:
            print(f"无法在 input_data 中找到未来日期范围 {future_dates[0]} 到 {future_dates[-1]} 的真实数据，跳过评估。")

        if GENERATE_PLOTS:
            plt.figure(figsize=(12, 6))
            recent_history = input_data[-points_per_day:]
            plt.plot(recent_history.index, recent_history, label='历史数据', color='blue')
            plt.plot(result_df.index, result_df['predicted'], label='预测值', color='red')
            plt.fill_between(
                result_df.index,
                result_df['lower_ci'],
                result_df['upper_ci'],
                color='red', alpha=0.2,
                label='95% 置信区间'
            )
            plt.title('未来一天的SARIMA预测结果 (95%置信区间)')
            plt.xlabel('时间')
            plt.ylabel('产气量')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'prediction_{timestamp}.png'))
            plt.close()
            print(f"成功生成预测图 (含95%置信区间)")

        return result_df

    except Exception as e:
        print(f"预测失败: {e}")
        traceback.print_exc()

        # 尝试使用替代方法
        try:
            print("\n尝试使用替代方法计算95%置信区间...")
            if prediction_date is None:
                last_date = input_data.index[-1]
                start_date = last_date + timedelta(minutes=30)
            else:
                start_date = pd.to_datetime(prediction_date)

            future_dates = pd.date_range(start=start_date, periods=points_per_day, freq='30T')
            pred = model.forecast(steps=points_per_day)
            pred_series = pd.Series(pred, index=future_dates)
            resid_std = np.std(model.resid)
            z_value = 1.96
            lower_ci = pred_series - z_value * resid_std
            upper_ci = pred_series + z_value * resid_std

            result_df = pd.DataFrame({
                'predicted': pred_series.values,
                'lower_ci': lower_ci.values,
                'upper_ci': upper_ci.values
            }, index=future_dates)

            save_data_to_csv(result_df, f'prediction_approx_{timestamp}.xlsx', output_dir)

            # 评估
            try:
                y_true = input_data.loc[future_dates]
                if len(y_true) != len(result_df):
                    raise ValueError(f"真实数据长度({len(y_true)})必须与预测点数({len(result_df)})一致。")
                evaluate_and_save_metrics(y_true.values, result_df['predicted'].values, output_dir, timestamp)
            except KeyError:
                print(f"无法在 input_data 中找到未来日期范围 {future_dates[0]} 到 {future_dates[-1]} 的真实数据，跳过评估。")

            if GENERATE_PLOTS:
                plt.figure(figsize=(12, 6))
                recent_history = input_data[-points_per_day:]
                plt.plot(recent_history.index, recent_history, label='历史数据', color='blue')
                plt.plot(result_df.index, pred_series, label='预测值', color='red')
                plt.fill_between(
                    result_df.index,
                    lower_ci,
                    upper_ci,
                    color='red', alpha=0.2,
                    label='95% 置信区间 (近似)'
                )
                plt.title('未来一天的SARIMA预测结果 (近似95%置信区间)')
                plt.xlabel('时间')
                plt.ylabel('产气量')
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'prediction_approx_{timestamp}.png'))
                plt.close()
                print(f"成功生成预测图 (含近似95%置信区间)")

            return result_df

        except Exception as e2:
            print(f"替代方法也失败: {e2}")
            return None


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SARIMA气体产量预测 - 使用训练好的模型进行预测')
    
    parser.add_argument('--input_file', type=str, required=False,
                        help='输入数据文件路径 (xlsx格式)')
    parser.add_argument('--date_column', type=str, default='date',
                        help='日期列名')
    parser.add_argument('--target_column', type=str, default='OT',
                        help='目标列名')
    parser.add_argument('--model_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='模型目录')
    parser.add_argument('--model_file', type=str, default=DEFAULT_MODEL_FILE,
                        help='模型文件名')
    parser.add_argument('--input_days', type=int, default=DEFAULT_INPUT_DAYS,
                        help='使用最近几天的数据进行预测')
    parser.add_argument('--prediction_date', type=str, default=None,
                        help='预测日期 (格式: YYYY-MM-DD), 默认为输入数据的最后一天的下一天')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='输出目录')
    
    return parser.parse_args()



def main():
    """主函数"""
    start_time = time.time()
    
    # 解析命令行参数
    args = parse_args()
    
    # 设置输出目录
    output_dir = setup_output_directory(args.output_dir)
    
    # 设置中文字体
    setup_font_for_chinese()
    
    # 加载模型信息
    model_info = load_model_info(args.model_dir)
    points_per_day = model_info.get('POINTS_PER_DAY', 48)
    sampling_interval = model_info.get('SAMPLING_INTERVAL', 30)
    
    model = load_model(model_path)
    
    if model is None:
        print("模型加载失败，无法进行预测")
        return
    
    input_data = load_input_data(input_file, date_column, target_column, sampling_interval)
    
    # 获取最近几天的数据
    recent_data = get_recent_data(input_data, args.input_days, points_per_day)
    print(f"使用最近 {args.input_days} 天的数据进行预测 (共 {len(recent_data)} 个点)")
    
    # 进行预测
    prediction = predict_with_95ci(model, recent_data, points_per_day, output_dir, args.prediction_date)
    
    if prediction is not None:
        print("预测成功!")
    else:
        print("预测失败!")
    
    # 打印运行时间
    end_time = time.time()
    print(f"\n程序运行总时间: {(end_time - start_time):.2f} 秒")

# python ARIMIA\joy\predict.py --prediction_date 2025-06-04
if __name__ == "__main__":
    # 加载输入数据
    input_file = r''
    model_path = r''
    date_column = 'timestamp'
    target_column = '瞬时流量'
    main() 