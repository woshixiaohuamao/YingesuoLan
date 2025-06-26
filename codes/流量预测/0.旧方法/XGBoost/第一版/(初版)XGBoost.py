import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import xgboost as xgb

# 数据准备 --------------------------------------------------------------
# 目标变量应为'瞬时流量'
file_path = ''
df = pd.read_excel(file_path)
features = df.drop(columns=['瞬时流量', '时间'])  # 移除目标变量和时间列
target = df['瞬时流量']

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    features, 
    target,
    test_size=0.2,
    shuffle=True  # 保持时间顺序
)

# 特征标准化 -----------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)     

# 转换回DataFrame
train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# 模型配置 ------------------------------------------------------------
xgb_reg = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=139,        # 增加树的数量提升表现
    max_depth=4,             # 适当增加深度
    learning_rate=0.29999999999999993,      # 调低学习率
    subsample=0.9089509263201534,
    colsample_bytree=0.900635462955971,
    random_state=42,
    n_jobs=-1               # 使用全部CPU核心
)

# 模型训练（使用标准化后的数据）----------------------------------------
xgb_reg.fit(X_train_scaled, y_train)

# 预测与评估 ----------------------------------------------------------
y_pred = xgb_reg.predict(X_test_scaled)

# 综合评估指标
metrics = {
    "MSE": mean_squared_error(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "MAE": mean_absolute_error(y_test, y_pred),
    "R²": r2_score(y_test, y_pred),
    "MAPE": mean_absolute_percentage_error(y_test, y_pred)
}

# 格式化输出结果
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")