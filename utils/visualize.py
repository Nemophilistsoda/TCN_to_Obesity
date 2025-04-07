import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
import numpy as np

def plot_results(ground_truth, predictions, save_path="results/figures/result.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(ground_truth, label="真实值", marker='o')
    plt.plot(predictions, label="预测值", marker='x', linestyle='--')
    plt.xlabel("时间步")
    plt.ylabel("肥胖率")
    plt.title("肥胖率预测效果对比")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(features, importance, save_path="results/figures/feature_importance.png"):
    plt.figure(figsize=(8, 4))
    plt.barh(features, importance)
    plt.title("特征重要性分析")
    plt.savefig(save_path)
    plt.close()