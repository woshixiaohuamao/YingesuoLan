import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.signal import argrelextrema
import joblib
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import classification_report

# ========== 1. 加载并构造训练数据 ========== #
def load_and_prepare_data(file_path, min_samples=50):
    df = pd.read_excel(file_path, parse_dates=['time'])
    df.set_index('time', inplace=True)

    # ✅ 计算总流量
    df['total_flow'] = df['DLDZ_DQ200_LLJ01_FI01.PV'] + df['DLDZ_AVS_LLJ01_FI01.PV']

    # ✅ 构造设备组合标签
    device_cols = [
        'DLDZ_AVS_KYJ01_YI01.PV', 'DLDZ_AVS_KYJ02_YI01.PV',
        'DLDZ_AVS_KYJ03_YI01.PV', 'DLDZ_AVS_KYJ04_YI01.PV',
        'DLDZ_AVS_KYJ05_YI01.PV'
    ]
    df['device_combo'] = df[device_cols].astype(str).agg(''.join, axis=1)

    # ✅ 统计每种组合出现次数，过滤稀有组合
    combo_counts = df['device_combo'].value_counts()
    valid_combos = combo_counts[combo_counts >= min_samples].index
    df = df[df['device_combo'].isin(valid_combos)]

    # ✅ 增加时间/滑动特征
    df['flow_mean_5'] = df['total_flow'].rolling(window=5, min_periods=1).mean()
    df['flow_std_10'] = df['total_flow'].rolling(window=10, min_periods=1).std()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute

    # 填充滑窗缺失值
    df.fillna(method='bfill', inplace=True)

    return df, device_cols

# ========== 2. 模型训练函数 ========== #
def train_model(df):
    features = ['total_flow', 'flow_mean_5', 'flow_std_10', 'hour', 'minute']
    X = df[features]
    y = df['device_combo']

    # 划分训练/测试
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ✅ 增加正则限制防止过拟合
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train, y_train)

    # ✅ 打印评估指标
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    print("\n📊 分类评估报告:")
    print(classification_report(y_test, clf.predict(X_test)))

    return clf


# ========== 3. 找到一天中4个关键点 ========== #
def find_important_time_points(flow_series):
    flow_array = flow_series.values

    # 极小值点（滑动窗口60分钟内最低）
    min_indices = argrelextrema(flow_array, np.less_equal, order=60)[0]
    if len(min_indices) < 3:
        min_indices = np.argsort(flow_array)[:3]
    else:
        min_indices = min_indices[np.argsort(flow_array[min_indices])[:3]]

    # 正常点（中午12点或全局平均）
    normal_index = 720 if len(flow_array) > 720 else int(len(flow_array) / 2)
    all_indices = [normal_index] + min_indices.tolist()
    return all_indices

# ========== 4. 推荐函数 ========== #
def recommend_devices(model, full_df, important_indices, device_cols):
    """
    参数说明：
    - model：已训练的随机森林模型
    - full_df：包含 'total_flow' 的完整预测 DataFrame
    - important_indices：选中的推荐时间索引
    - device_cols：设备列名（用于输出格式）
    """
    # 复制一份 DataFrame，防止污染原数据
    df = full_df.copy()

    # 计算预测数据中的辅助特征
    df['flow_mean_5'] = df['total_flow'].rolling(window=5, min_periods=1).mean()
    df['flow_std_10'] = df['total_flow'].rolling(window=10, min_periods=1).std()

    # 如果没有时间列就构造（你也可以提前准备好）
    if 'time' in df.columns:
        df['hour'] = pd.to_datetime(df['time']).dt.hour
        df['minute'] = pd.to_datetime(df['time']).dt.minute
    else:
        # 若 time 列不存在就用 index 构造
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute

    df.fillna(method='bfill', inplace=True)

    # 构造模型所需特征
    features = ['total_flow', 'flow_mean_5', 'flow_std_10', 'hour', 'minute']
    X_pred = df.iloc[important_indices][features]

    # 预测设备组合
    predicted_combos = model.predict(X_pred)

    results = []
    for i, combo in zip(important_indices, predicted_combos):
        combo_list = [int(c) for c in combo]
        results.append({
            "time_index": int(i),
            "flow": float(df.iloc[i]['total_flow']),
            "recommended_devices": dict(zip(device_cols, combo_list))
        })

    return results


# ========== 5. 主流程 ========== #
def main():
    # 路径配置
    historical_data_path = r''
    predicted_flow_path = r''
    model_path = 'device_model.pkl'

    # Step 1：加载与训练模型
    df, device_cols = load_and_prepare_data(historical_data_path)
    clf = train_model(df)
    joblib.dump(clf, model_path)
    print(f"模型已保存到：{model_path}")

    # Step 2: 读取预测数据
    df_pred = pd.read_excel(predicted_flow_path)

    # 自动计算 total_flow
    if 'total_flow' not in df_pred.columns:
        df_pred['total_flow'] = df_pred['DLDZ_DQ200_LLJ01_FI01.PV'] + df_pred['DLDZ_AVS_LLJ01_FI01.PV']

    # Step 3: 获取关键推荐点
    important_indices = find_important_time_points(df_pred['total_flow'])

    # ✅ Step 4: 推荐设备组合（传 full_df 给推荐函数）
    results = recommend_devices(clf, df_pred, important_indices, device_cols)

    # Step 5：输出结果
    print("\n📌 每日设备推荐组合如下：")
    for item in results:
        print(f"[Index {item['time_index']}] 流量: {item['flow']:.2f}, 推荐设备: {item['recommended_devices']}")

    # 可选：保存为json
    pd.DataFrame(results).to_json("daily_recommendation.json", orient='records', force_ascii=False)
    print("\n✅ 推荐已保存为 daily_recommendation.json")


if __name__ == "__main__":
    main()
