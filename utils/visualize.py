# visualize.py
import matplotlib.pyplot as plt
import shap
import torch
import numpy as np


def plot_comparison(y_true, y_pred, save_path):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='真实值', marker='o')
    plt.plot(y_pred, label='预测值', linestyle='--', marker='x')
    plt.xlabel('时间步')
    plt.ylabel('肥胖率')
    plt.title('真实值与预测值对比')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def shap_analysis(model, background, test_data, feature_names, save_path):
    # 移除冗余的numpy转换
    plt.figure()
    # 删除以下重复代码
    # shap.summary_plot(shap_values,
    #                  test_data.numpy(),
    #                  feature_names=feature_names,
    #                  plot_type="bar")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()