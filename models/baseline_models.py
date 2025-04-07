# 基线模型（线性回归、ARIMA、Prophet）
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


class BaselineModels:
    @staticmethod
    def arima_predict(data, order=(1, 0, 0), steps=12):
        """ARIMA预测"""
        model = sm.tsa.ARIMA(data, order=order).fit()
        return model.forecast(steps=steps)

    @staticmethod
    def linear_regression_predict(X_train, y_train, X_test):
        """线性回归预测"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model.predict(X_test)