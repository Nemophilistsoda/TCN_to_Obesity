import numpy as np
import yaml  # 需要安装PyYAML（pip install pyyaml）

def load_data(data_path, window_size):
    """加载数据并生成滑动窗口样本"""
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, 1])  # 假设第1列是肥胖率
        y.append(data[i+window_size, 1])
    return np.array(X), np.array(y)

def load_config(config_path="configs/params.yaml"):
    """加载配置参数"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config