import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

file_name = ''
df =pd.read_excel(file_name)

# 特征选择（基于提供的字段）
features = [
    'predict', 'hour', 'weekday', 'day', 'day_part',
    'residual_lag1', 'SARIMA_lag1', 'residual_lag2', 'SARIMA_lag2',
    'residual_lag3', 'SARIMA_lag3', 'residual_lag24', 'SARIMA_lag24',
    'residual_lag48', 'SARIMA_lag48', 'rolling_8h_mean'
]
target = 'residual'

# 处理缺失值（滞后特征产生的缺失）
df_clean = df.dropna(subset=features+[target]).reset_index(drop=True)

# 时间序列分割（按时间顺序）
total_size = len(df_clean)
train_size = int(total_size * 0.7)
val_size = int(total_size * 0.15)

train_df = df_clean.iloc[:train_size]
val_df = df_clean.iloc[train_size:train_size+val_size]
test_df = df_clean.iloc[train_size+val_size:]

print(f"数据集大小: 总样本{total_size}, 训练集{train_size}, 验证集{len(val_df)}, 测试集{len(test_df)}")

# 准备数据
X_train = train_df[features]
y_train = train_df[target]
X_val = val_df[features]
y_val = val_df[target]
X_test = test_df[features]
y_test = test_df[target]

# 超参数搜索（使用时间序列交叉验证）
tscv = TimeSeriesSplit(n_splits=3)

param_grid = {
    'num_leaves': [31, 63],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [300, 500],
    'max_depth': [5, 7],
    'min_child_samples': [20, 50],
    'reg_alpha': [0, 0.1]
}

print("开始超参数搜索...")
lgb_model = lgb.LGBMRegressor(random_state=42, force_col_wise=True)
grid_search = GridSearchCV(
    estimator=lgb_model,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print("最优参数:", grid_search.best_params_)
best_params = grid_search.best_params_


final_model = lgb.LGBMRegressor(**best_params, random_state=42)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50)
    ]
)

# 保存模型
joblib.dump(final_model, 'residual_correction_model.pkl')
print("模型已保存为: residual_correction_model.pkl")

# 评估函数
def evaluate_model(model, X, y_true, sarima_pred, name):
    """评估模型并返回指标"""
    y_res_pred = model.predict(X)
    y_final_pred = sarima_pred + y_res_pred  # 加入残差修正
    
    rmse = np.sqrt(mean_squared_error(y_true, y_final_pred))
    mae = mean_absolute_error(y_true, y_final_pred)
    mape = np.mean(np.abs((y_true - y_final_pred) / np.clip(np.abs(y_true), 1e-5, None))) * 100
    mse = mean_squared_error(y_true, y_final_pred)  
    
    # 原始SARIMA的误差
    sarima_rmse = np.sqrt(mean_squared_error(y_true, sarima_pred))
    sarima_mse = mean_squared_error(y_true, sarima_pred)  
    
    print(f"\n{name}集评估结果:")
    print(f"最终预测 RMSE: {rmse:.4f} | MSE: {mse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}%")
    print(f"相比原始SARIMA提升: "
          f"RMSE提升: {(sarima_rmse - rmse)/sarima_rmse*100:.2f}% | "
          f"MSE提升: {(sarima_mse - mse)/sarima_mse*100:.2f}%")  
    
    return y_final_pred, rmse, mae, mape, mse

# 在三个数据集上评估
train_final_pred, train_rmse, train_mae, train_mape, train_mse = evaluate_model(
    final_model, X_train, y_train, train_df['predict'], "训练"
)

val_final_pred, val_rmse, val_mae, val_mape, val_mse = evaluate_model(
    final_model, X_val, y_val, val_df['predict'], "验证"
)

test_final_pred, test_rmse, test_mae, test_mape, test_mse = evaluate_model(
    final_model, X_test, y_test, test_df['predict'], "测试"
)

# 可视化结果
plt.figure(figsize=(15, 18))
plt.rcParams['font.sans-serif'] = ['SimSun']      

# 1. 特征重要性
plt.subplot(3, 1, 1)
lgb.plot_importance(final_model, max_num_features=15, importance_type='gain')
plt.title('特征重要性 (信息增益)')

# 2. 测试集预测对比
plt.subplot(3, 1, 2)
sample_data = test_df.iloc[:100]  # 取前100个样本展示
plt.plot(sample_data['date'], sample_data['Actual'], 'b-', label='真实值')
plt.plot(sample_data['date'], sample_data['predict'], 'r--', label='SARIMA预测')
plt.plot(sample_data['date'], test_final_pred[:100], 'g-.', label='校正后预测')
plt.legend()
plt.title('测试集预测效果对比')
plt.xlabel('时间')
plt.ylabel('值')
plt.xticks(rotation=45)

# 3. 残差分布
plt.subplot(3, 1, 3)
final_residuals = test_df['Actual'] - test_final_pred
plt.hist(final_residuals, bins=50, alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('最终预测残差分布')
plt.xlabel('残差值')
plt.ylabel('频次')

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300)
print("评估图表已保存为: model_evaluation.png")

# 最终评估报告
print("\n" + "="*50)
print("最终模型评估报告")
print("="*50)
print(f"训练集: RMSE={train_rmse:.4f}, MSE={train_mse:.4f}, MAE={train_mae:.4f}, MAPE={train_mape:.2f}%")
print(f"验证集: RMSE={val_rmse:.4f}, MSE={val_mse:.4f}, MAE={val_mae:.4f}, MAPE={val_mape:.2f}%")
print(f"测试集: RMSE={test_rmse:.4f}, MSE={test_mse:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%")

# 残差自相关分析
from statsmodels.graphics.tsaplots import plot_acf

plt.figure(figsize=(12, 6))
plot_acf(final_residuals, lags=48, alpha=0.05, title='最终残差自相关函数 (ACF)')
plt.savefig('residual_acf.png', dpi=300)
print("残差ACF图已保存为: residual_acf.png")