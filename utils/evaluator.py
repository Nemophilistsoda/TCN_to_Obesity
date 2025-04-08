# 可解释性分析（特征重要性、SHAP）

# evaluator.py
import numpy as np


def calculate_rmse(true, pred):
    return np.sqrt(np.mean((true - pred) ** 2))


def calculate_mae(true, pred):
    return np.mean(np.abs(true - pred))
