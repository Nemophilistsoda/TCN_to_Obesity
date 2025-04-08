"""
基层健康治理智能决策系统 - Pro版核心入口
功能：多模型对比 | 可解释性分析 | 政策模拟
"""
import argparse
import numpy as np
import torch
import pandas as pd
from models.tcn import AdvancedTCN
from models.baseline_models import BaselineModels
from utils.data_processor import ObesityDataProcessor
from utils.evaluator import calculate_rmse, calculate_mae
from utils.visualize import plot_comparison, shap_analysis
from utils.config_loader import load_config


def main():
    # 配置加载与参数解析
    parser = argparse.ArgumentParser(description='Obesity Policy AI Pro')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['train', 'predict', 'compare', 'explain'],
                        help='运行模式: train(训练)/predict(预测)/compare(模型对比)/explain(可解释分析)')
    parser.add_argument('--config', type=str, default='configs/policy_analysis.yaml',
                        help='配置文件路径')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理管道
    processor = ObesityDataProcessor()
    features, target = processor.load_transform(
        config['data_paths']['national'],
        window_size=config['model_params']['window_size'],
        use_features=config['data_params']['use_features']  # 使用配置文件中的特征列表
    )

    # 添加维度检查
    if len(features.shape) != 3 or features.shape[1] == 0:
        raise ValueError(f"特征数据维度不正确，期望形状为 (样本数, 时间步长, 特征数)，实际得到 {features.shape}")

    # 划分训练集/测试集
    split_idx = int(len(features) * config['data_params']['train_ratio'])
    X_train, X_test = features[:split_idx], features[split_idx:]
    y_train, y_test = target[:split_idx], target[split_idx:]

    # 多模式处理
    if args.mode == 'train' or args.mode == 'full':
        # 训练增强版TCN
        tcn_model = AdvancedTCN(
            input_size=X_train.shape[2],  # 使用特征维度作为输入大小
            hidden_size=config['model_params']['hidden_size'],
            num_layers=config['model_params']['num_layers']
        ).to(device)
        tcn_model.fit(X_train, y_train,
                      epochs=config['training']['epochs'],
                      lr=config['training']['learning_rate'],
                      batch_size=config['training']['batch_size'])
        torch.save(tcn_model.state_dict(), config['model_paths']['tcn'])

    if args.mode == 'predict' or args.mode == 'full':
        # 加载模型进行预测
        tcn_model.load_state_dict(torch.load(config['model_paths']['tcn']))

        # 确保输入数据维度正确
        if len(X_test.shape) == 2:
            X_test = np.expand_dims(X_test, axis=0)

        predictions = tcn_model.predict(X_test)

        # 保存预测结果
        np.savetxt(config['result_paths']['predictions'],
                   np.column_stack([y_test, predictions]),
                   header='Actual,Predicted', delimiter=',')

        # 可视化对比
        plot_comparison(y_test, predictions,
                        save_path=config['result_paths']['comparison_fig'])

    if args.mode == 'compare' or args.mode == 'full':
        # 多模型性能对比
        models = {
            'Linear Regression': lambda x, y: BaselineModels.linear_regression(X_train, y_train, x),
            'ARIMA': lambda x, y: BaselineModels.arima_predict(y_train, order=(1,0,0), steps=len(y)),
            'TCN': lambda x, y: tcn_model.predict(x)
        }

        results = {}
        for name, model in models.items():
            try:
                preds = model(X_test, y_test)  # 统一传递测试集
                results[name] = {
                    'RMSE': calculate_rmse(y_test, preds[:len(y_test)]),  # 截断预测结果
                    'MAE': calculate_mae(y_test, preds[:len(y_test)])
                }
            except Exception as e:
                print(f"{name} 模型错误: {str(e)}")
        print("\n模型性能对比:")
        print(pd.DataFrame(results).T)

    if args.mode == 'explain' or args.mode == 'full':
        # SHAP可解释性分析
        sample_size = min(100, len(X_train))  # 动态调整样本量
        background = X_train[np.random.choice(len(X_train), sample_size, replace=sample_size > len(X_train))]
        shap_analysis(tcn_model, background, X_test[:50],
                      feature_names=processor.feature_names,
                      save_path=config['result_paths']['shap_fig'])


if __name__ == "__main__":
    main()