import numpy as np


class SimpleTCN:
    """纯NumPy实现的极简TCN，适合CPU运行"""

    def __init__(self, window_size=12):
        self.window_size = window_size
        self.weights = np.random.randn(window_size) * 0.1  # 初始化权重

    def train(self, X, y, epochs=100, lr=0.01):
        """训练：X为输入序列，y为目标值"""
        for epoch in range(epochs):
            pred = np.dot(X, self.weights)
            loss = np.mean((pred - y) ** 2)
            # 梯度下降更新
            grad = 2 * np.dot(X.T, (pred - y)) / len(X)
            self.weights -= lr * grad
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, x):
        return np.dot(x, self.weights)