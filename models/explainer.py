# 可解释性分析（特征重要性、SHAP）
import shap
import matplotlib.pyplot as plt


class SHAPExplainer:
    def __init__(self, model, background_data):
        self.explainer = shap.DeepExplainer(model, background_data)

    def analyze(self, sample_data, feature_names):
        shap_values = self.explainer.shap_values(sample_data)
        shap.summary_plot(shap_values, sample_data, feature_names=feature_names, plot_type="bar")
        plt.savefig("results/figures/shap_summary.png")