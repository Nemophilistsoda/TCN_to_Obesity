import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class ObesityDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []

    def load_transform(self, file_path, window_size, use_features=None):
        # 读取数据
        df = pd.read_csv(file_path)

        # 检查必要列是否存在
        required_columns = ['obesity_rate'] + (use_features if use_features else [])
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"数据文件中缺少以下列: {missing_columns}")

        # 特征处理
        if use_features:
            df = df[['obesity_rate'] + use_features]
        else:
            df = df[['obesity_rate']]

        # 归一化
        scaled = self.scaler.fit_transform(df)
        self.feature_names = df.columns.tolist()

        # 生成滑动窗口样本
        X, y = [], []
        for i in range(len(scaled) - window_size):
            X.append(scaled[i:i + window_size])
            y.append(scaled[i + window_size, 0])  # 预测下一时间步肥胖率

        # 转换为numpy数组并检查形状
        X = np.array(X)
        y = np.array(y)

        if len(X.shape) != 3:
            raise ValueError(f"滑动窗口数据生成错误，期望形状为 (样本数, 时间步长, 特征数)，实际得到 {X.shape}")

        # 在生成滑动窗口后添加验证
        if len(X) < 5:  # 至少需要5个样本
            raise ValueError(f"数据量不足，滑动窗口生成后仅有 {len(X)} 个样本")

        return X, y

    def _add_lag_features(self, df, lags=3):
        """添加滞后特征"""
        for lag in range(1, lags + 1):
            df[f'lag_{lag}'] = df['obesity_rate'].shift(lag)
        df.dropna(inplace=True)