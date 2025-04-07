from models.baseline_models import BaselineModels
# 训练ARIMA
arima_pred = BaselineModels.arima_predict(train_data, order=(1,1,1), steps=12)
# 评估
from utils.evaluator import calculate_rmse
print("ARIMA RMSE:", calculate_rmse(test_data, arima_pred))