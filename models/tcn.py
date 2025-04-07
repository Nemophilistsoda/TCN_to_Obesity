# 增强版TCN（残差连接+扩张卷积）
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.res = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.res(x)


class AdvancedTCN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i  # 扩张因子指数增长
            layers.append(ResidualBlock(input_size if i == 0 else hidden_size, hidden_size, dilation))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, features, seq_len]
        x = self.tcn(x)
        x = x[:, :, -1]  # 取最后时间步
        return self.fc(x)