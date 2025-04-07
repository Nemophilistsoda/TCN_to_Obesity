from models.simple_tcn import SimpleTCN
from utils.data_loader import load_data, load_config
from utils.visualize import plot_results


def main():
    # 加载配置
    config = load_config()

    # 加载数据
    X, y = load_data("data/national/national.csv", config['window_size'])

    # 训练模型
    model = SimpleTCN(window_size=config['window_size'])
    model.train(X, y, epochs=config['epochs'], lr=config['lr'])

    # 预测并保存结果
    test_X = X[-10:]  # 用最后10个样本测试
    preds = [model.predict(x) for x in test_X]
    np.savetxt("results/predictions/pred.csv", preds, delimiter=",")

    # 可视化
    plot_results(y[-10:], preds, save_path="results/figures/result.png")


if __name__ == "__main__":
    main()