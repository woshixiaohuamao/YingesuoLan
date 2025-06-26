# change_point_detector.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy import stats

class ChangePointDetector:
    """
    突变点检测器，提供两种处理策略
    
    功能：
    1. 检测数据中的突变点
    2. 提供两种处理突变点的策略：
        - 策略一：剔除突变点后重新训练
        - 策略二：将突变点作为额外特征加入模型
    
    使用方法：
    detector = ChangePointDetector()
    results = detector.detect_and_handle(X, y, confidence_level=0.95)
    """
    
    def __init__(self, model=None):
        """
        初始化检测器
        
        参数：
        model: 自定义模型（需实现fit和predict方法），默认为RandomForestRegressor
        """
        self.model = model or RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.anomaly_mask = None
        self.anomaly_indices = None
        self.residuals = None
    
    def detect_change_points(self, X, y, confidence_level=0.95):
        """检测突变点"""
        # 检查输入数据是否有NaN
        if X.isnull().any().any():
            print("警告: 输入特征矩阵包含NaN值，正在填充0...")
            X = X.fillna(0)
        
        if y.isnull().any():
            print("警告: 目标变量包含NaN值，正在填充0...")
            y = y.fillna(0)
        
        # 训练初始模型
        self.model.fit(X, y)
        
        # 获取预测值和残差
        y_pred = self.model.predict(X)
        self.residuals = y - y_pred
        
        # 计算置信区间
        std_residual = np.std(self.residuals)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * std_residual
        
        # 创建置信区间
        lower_bound = y_pred - margin_of_error
        upper_bound = y_pred + margin_of_error
        
        # 识别突变点
        self.anomaly_mask = (y < lower_bound) | (y > upper_bound)
        self.anomaly_indices = np.where(self.anomaly_mask)[0]
        
        print(f"检测到 {len(self.anomaly_indices)} 个突变点 (占总数据 {len(self.anomaly_indices)/len(y):.2%})")
        
        return {
            'model': self.model,
            'y_pred': y_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'anomaly_mask': self.anomaly_mask,
            'anomaly_indices': self.anomaly_indices,
            'residuals': self.residuals
        }
    
    def strategy_remove_anomalies(self, X, y):
        """剔除突变点后重新训练"""
        if self.anomaly_mask is None:
            raise ValueError("请先运行 detect_change_points() 方法")
        
        # 确保没有NaN值
        if X.isnull().any().any():
            print("警告: 特征矩阵包含NaN值，正在填充0...")
            X = X.fillna(0)
        
        if y.isnull().any():
            print("警告: 目标变量包含NaN值，正在填充0...")
            y = y.fillna(0)
        
        X_clean = X[~self.anomaly_mask]
        y_clean = y[~self.anomaly_mask]
        
        print(f"剔除策略: 使用 {len(X_clean)} 个样本 (原始样本 {len(X)})")
        
        model = self.model.__class__(**self.model.get_params())
        model.fit(X_clean, y_clean)
        y_pred = model.predict(X)
        
        return model, y_pred
    
    def strategy_add_parameters(self, X, y):
        """将突变点作为额外特征加入模型"""
        if self.anomaly_indices is None:
            raise ValueError("请先运行 detect_change_points() 方法")
        
        # 确保没有NaN值
        if X.isnull().any().any():
            print("警告: 特征矩阵包含NaN值，正在填充0...")
            X = X.fillna(0)
        
        if y.isnull().any():
            print("警告: 目标变量包含NaN值，正在填充0...")
            y = y.fillna(0)
        
        # 创建突变点特征
        X_augmented = X.copy()
        
        # 添加突变点指示器 - 更安全的方法
        # 创建一个全零矩阵，然后设置突变点位置为1
        anomaly_matrix = np.zeros((len(X), len(self.anomaly_indices)))
        
        for i, idx in enumerate(self.anomaly_indices):
            # 确保索引在有效范围内
            if idx < len(anomaly_matrix):
                anomaly_matrix[idx, i] = 1
        
        # 将矩阵转换为DataFrame
        anomaly_df = pd.DataFrame(
            anomaly_matrix,
            columns=[f"anomaly_{i}" for i in range(len(self.anomaly_indices))],
            index=X.index
        )
        
        # 合并到原始特征矩阵
        X_augmented = pd.concat([X_augmented, anomaly_df], axis=1)
        
        # 确保没有NaN值
        if X_augmented.isnull().any().any():
            print("警告: 特征矩阵包含NaN值，正在填充0...")
            X_augmented = X_augmented.fillna(0)
        
        print(f"参数增强策略: 添加 {len(self.anomaly_indices)} 个突变点特征")
        
        model = self.model.__class__(**self.model.get_params())
        model.fit(X_augmented, y)
        y_pred = model.predict(X_augmented)
        
        return model, y_pred, X_augmented
    
    def detect_and_handle(self, X, y, confidence_level=0.95):
        """完整流程：检测突变点并应用两种处理策略"""
        # 检测突变点
        detection_results = self.detect_change_points(X, y, confidence_level)
        
        # 应用策略一
        remove_model, remove_pred = self.strategy_remove_anomalies(X, y)
        
        # 应用策略二
        add_model, add_pred, X_augmented = self.strategy_add_parameters(X, y)
        
        return {
            'detection': detection_results,
            'strategy_remove': {
                'model': remove_model,
                'y_pred': remove_pred
            },
            'strategy_add_feature': {
                'model': add_model,
                'y_pred': add_pred,
                'X_augmented': X_augmented
            }
        }