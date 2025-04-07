from models.simple_tcn import SimpleTCN
from utils.data_loader import load_data, load_config
from utils.visualize import plot_results
import numpy as np

def main():
    # 加载配置
    config = load_config()

    # 加载数据
    # window_size = config['data_params']['window_size']  # 按实际层级访问
    X, y = load_data("data/national/national.csv", config['data_params']['window_size'])

    # 训练模型
    model = SimpleTCN(window_size=config['data_params']['window_size'])
    model.train(X, y, epochs=config['model']['epochs'], lr=config['model']['learning_rate'])

    # 预测并保存结果
    test_X = X[-10:]  # 用最后10个样本测试
    preds = [model.predict(x) for x in test_X]
    np.savetxt("results/predictions/pred.csv", preds, delimiter=",", encoding="utf-8")

    # 可视化
    plot_results(y[-10:], preds, save_path="results/figures/result.png")

    # 在main.py中添加数据探索代码
    config = load_config()
    local_data = np.genfromtxt(config['paths']['local_data'],
                               delimiter=',',
                               skip_header=1)
    local_values = local_data[:, 1]  # 假设第1列是月份，第2列是肥胖率

    print("基层数据统计：")
    print("均值:", np.mean(local_values))
    print("方差:", np.var(local_values))
    # 若方差<0.01，说明数据波动小，模型难以捕捉变化




if __name__ == "__main__":
    main()