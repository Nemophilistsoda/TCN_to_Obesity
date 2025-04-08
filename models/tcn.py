import torch
import torch.nn as nn
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3,
                     dilation=dilation, padding=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        # 使用条件判断代替nn.Identity
        self.res = (nn.Conv1d(in_channels, out_channels, 1)
                   if in_channels != out_channels
                   else None)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        if self.res is not None:
            residual = self.res(residual)
        return x + residual


class AdvancedTCN(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i  # 扩张卷积指数增长
            layers.append(ResidualBlock(
                input_size if i == 0 else hidden_size,
                hidden_size,
                dilation
            ))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 如果输入是numpy数组，转换为tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.permute(0, 2, 1)  # [batch, features, seq_len]
        x = self.tcn(x)
        x = self.fc(x[:, :, -1])
        return x

    def fit(self, X_train, y_train, epochs=100, lr=0.01, batch_size=32):
        # 转换为PyTorch Dataset
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, X):
        """模型预测方法"""
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            # 将numpy数组转换为tensor
            X_tensor = torch.from_numpy(X).float()
            if torch.cuda.is_available():
                X_tensor = X_tensor.cuda()

            # 进行预测
            outputs = self(X_tensor)
            return outputs.cpu().numpy().flatten()