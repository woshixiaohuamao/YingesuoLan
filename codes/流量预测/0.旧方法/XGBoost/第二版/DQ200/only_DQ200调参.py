import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from skopt import BayesSearchCV

# 数据准备 --------------------------------------------------------------
file_path = ''
df = pd.read_excel(file_path)

# 移除非特征列
features = df.drop(columns=['DQ200总流量','时间'])
target = df['DQ200总流量']

# 特征标准化 -----------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 时间序列交叉验证器设置 -----------------------------------------------
tscv = TimeSeriesSplit(n_splits=5)

# 超参数搜索空间定义 --------------------------------------------------
param_space = {
    'reg__n_estimators': (50, 300),
    'reg__max_depth': (3, 10),
    'reg__learning_rate': (0.01, 0.3, 'log-uniform'),
    'reg__subsample': (0.6, 1.0),
    'reg__colsample_bytree': (0.6, 1.0),
}

# 构造模型流水线 -------------------------------------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('reg', XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1))
])

# 自动化调参器配置 -----------------------------------------------------
opt = BayesSearchCV(
    estimator=pipeline,
    search_spaces=param_space,
    scoring='neg_mean_squared_error',
    cv=tscv,
    n_jobs=-1,
    n_iter=50,            # 控制调参次数
    verbose=1,
    random_state=42
)

# 开始调参和训练 --------------------------------------------------------
opt.fit(features, target)

# 输出最佳参数 ---------------------------------------------------------
print("Best parameters found:")
print(opt.best_params_)

# 获取最优模型并进行预测 -----------------------------------------------
best_model = opt.best_estimator_

# 在整个数据集上训练后预测
y_pred = best_model.predict(features)

import joblib # 保存模型到文件 
joblib_file = "xgb_dq200_part.pkl" 
joblib.dump(best_model, joblib_file)

# 综合评估指标
metrics = {
    "MSE": mean_squared_error(target, y_pred),
    "RMSE": np.sqrt(mean_squared_error(target, y_pred)),
    "MAE": mean_absolute_error(target, y_pred),
    "R²": r2_score(target, y_pred),
    "MAPE": mean_absolute_percentage_error(target, y_pred)
}

# 将评估指标保存为 .txt 文件，并指定编码为 utf-8
# 训练时间大约为11分钟
with open("evaluation_metrics.txt", "w", encoding='utf-8') as file:
    file.write("Model Evaluation Metrics\n")
    file.write("========================\n")
    for metric, value in metrics.items():
        file.write(f"{metric}: {value}\n")

print("评估指标已成功保存到 'evaluation_metrics.txt' 文件。")