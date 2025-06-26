import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

file_name = ''
df = pd.read_excel(file_name)

# 特征选择（基于提供的字段）
features = ['predict', 'spectral_energy', 'spectral_spread']
target = 'Actual'

# 处理缺失值
df_clean = df.dropna(subset=features+[target]).reset_index(drop=True)

# 时间序列分割
points_per_day = 24 * 60 // 30  # 30分钟间隔，一天48个点
val_days = 7  # 验证集取7天
val_size = val_days * points_per_day  # 验证集总点数

total_size = len(df_clean)
test_start_idx = total_size - points_per_day  # 测试集起始位置
val_start_idx = test_start_idx - val_size    # 验证集起始位置

# 划分数据集
train_df = df_clean.iloc[:val_start_idx]        # 训练集: 开始 -> 验证集前
val_df = df_clean.iloc[val_start_idx:test_start_idx]  # 验证集: 一周数据
test_df = df_clean.iloc[test_start_idx:]         # 测试集: 最后一天

print(f"数据集大小: 总样本{total_size}, 训练集{len(train_df)}, 验证集{len(val_df)}, 测试集{len(test_df)}")

# 准备数据
X_train = train_df[features]
y_train = train_df[target]
X_val = val_df[features]
y_val = val_df[target]
X_test = test_df[features]
y_test = test_df[target]

# 超参数搜索（使用时间序列交叉验证）
tscv = TimeSeriesSplit(n_splits=3)

# XGBoost的超参数网格
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [300, 500],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [0.5, 1.0]
}

print("开始超参数搜索...")
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print("最优参数:", grid_search.best_params_)
best_params = grid_search.best_params_


final_model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    random_state=42,
    eval_metric='rmse',  
    **best_params
)


final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=50
)

# 保存模型
joblib.dump(final_model, 'XGB_Model.pkl')
print("模型已保存为: XGB_Model.pkl")

# 评估函数
def evaluate_model(model, X, y_true, sarima_pred, name):
    """评估模型并返回指标"""
    y_final_pred = model.predict(X) + sarima_pred  
    
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

# 特征重要性图
plt.figure(figsize=(15, 6))
xgb.plot_importance(final_model, max_num_features=15, importance_type='gain')
plt.title('特征重要性 (信息增益)')
plt.tight_layout()
plt.savefig('xgb_feature_importance.png', dpi=300)
print("特征重要性图已保存为: xgb_feature_importance.png")

# 其余可视化保持不变
plt.rcParams['font.sans-serif'] = ['SimSun'] 

# 测试集预测对比图
plt.figure(figsize=(15, 6))
sample_data = test_df.iloc[:100] 
plt.plot(sample_data['date'], sample_data['Actual'], 'b-', label='真实值')
plt.plot(sample_data['date'], sample_data['predict'], 'r--', label='SARIMA预测')
plt.plot(sample_data['date'], test_final_pred[:100], 'g-.', label='校正后预测')
plt.legend()
plt.title('测试集预测效果对比')
plt.xlabel('时间')
plt.ylabel('值')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('xgb_test_comparison.png', dpi=300)
print("测试集预测对比图已保存为: xgb_test_comparison.png")

# 残差分布图
plt.figure(figsize=(15, 6))
final_residuals = test_df['Actual'] - test_final_pred
plt.hist(final_residuals, bins=50, alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('最终预测残差分布')
plt.xlabel('残差值')
plt.ylabel('频次')
plt.tight_layout()
plt.savefig('xgb_residual_distribution.png', dpi=300) 
print("残差分布图已保存为: xgb_residual_distribution.png")

# 残差自相关分析
from statsmodels.graphics.tsaplots import plot_acf

plt.figure(figsize=(12, 6))
max_valid_lags = len(final_residuals) - 1
selected_lags = min(36, max_valid_lags)
plot_acf(final_residuals, lags=selected_lags, alpha=0.05, title='最终残差自相关函数 (ACF)')
plt.savefig('xgb_residual_acf.png', dpi=300)
print("残差ACF图已保存为: xgb_residual_acf.png")

# 最终评估报告
print("\n" + "="*50)
print("XGBoost最终模型评估报告")
print("="*50)
print(f"训练集: RMSE={train_rmse:.4f}, MSE={train_mse:.4f}, MAE={train_mae:.4f}, MAPE={train_mape:.2f}%")
print(f"验证集: RMSE={val_rmse:.4f}, MSE={val_mse:.4f}, MAE={val_mae:.4f}, MAPE={val_mape:.2f}%")
print(f"测试集: RMSE={test_rmse:.4f}, MSE={test_mse:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%")