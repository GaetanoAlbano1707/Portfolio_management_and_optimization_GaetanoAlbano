import torch
import torch.nn as nn


class EIIE(nn.Module):
    def __init__(self, num_assets: int, time_window: int, num_features: int):
        super(EIIE, self).__init__()
        self.conv1 = nn.Conv2d(num_features, 2, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(2, 20, kernel_size=(1, time_window))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(20 * num_assets, num_assets + 1)

    def forward(self, state, last_action):
        x = self.conv1(state)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.flatten(x)
        out = torch.softmax(self.linear(x), dim=1)
        return out


class GPM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(GPM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, state, last_action):
        batch_size = state.size(0)
        x = state.permute(0, 2, 1, 3).reshape(batch_size, state.shape[2], -1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        out = torch.softmax(self.fc(x), dim=1)
        return out


class EI3(nn.Module):
    def __init__(self, num_assets: int, time_window: int, num_features: int):
        super(EI3, self).__init__()
        self.conv1 = nn.Conv2d(num_features, 8, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(1, time_window))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * num_assets, 64)
        self.fc2 = nn.Linear(64, num_assets + 1)

    def forward(self, state, last_action):
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        out = torch.softmax(self.fc2(x), dim=1)
        return out
