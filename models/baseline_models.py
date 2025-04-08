# 基线模型（线性回归、ARIMA、Prophet）
import warnings
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import numpy as np


class BaselineModels:
    @staticmethod
    def arima_predict(data, order=(1,0,0), steps=None):
        """ARIMA预测"""
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = sm.tsa.ARIMA(data, order=order).fit()
                # 动态设置预测步长
                predict_steps = steps if steps else len(data)
                return model.forecast(steps=predict_steps)[:len(data)]  # 确保输出长度与输入一致
        except Exception as e:
            print(f"ARIMA预测失败: {str(e)}")
            return np.zeros(len(data))  # 返回与输入相同长度的零数组

    @staticmethod
    def linear_regression(X_train, y_train, X_test):
        """线性回归模型"""
        # 展平三维数据为二维 (样本数×时间步长, 特征数)
        X_train_2d = X_train.reshape(-1, X_train.shape[-1])
        X_test_2d = X_test.reshape(-1, X_test.shape[-1])

        # 扩展目标值以匹配特征维度
        y_train_expanded = np.repeat(y_train, X_train.shape[1])

        model = LinearRegression()
        model.fit(X_train_2d, y_train_expanded)

        # 对预测结果进行平均处理
        preds = model.predict(X_test_2d)
        return preds.reshape(len(X_test), -1).mean(axis=1)  # 按样本平均        return model.predict(X_test_2d)