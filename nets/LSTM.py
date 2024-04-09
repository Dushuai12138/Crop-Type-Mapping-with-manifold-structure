# -*- coding: utf-8 -*-
# time: 2024/4/9 14:04
# file: LSTM.py
# author: Shuai


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LSTM(nn.Module):
    def __init__(self, lstm_inputs=12, lstm_outputs=64, num_classes=4, num_layers=1):
        super(LSTM, self).__init__()
        self.lstm_inputs = lstm_inputs
        self.lstm_outputs = lstm_outputs
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size=lstm_inputs, hidden_size=lstm_outputs, num_layers=num_layers, batch_first=True,
                            bidirectional=False)
        self.fc = nn.Linear(lstm_outputs, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        inputs = x.reshape(batch_size, 6, self.lstm_inputs, 128 * 128)
        inputs = torch.transpose(inputs, 2, 3)
        inputs = torch.transpose(inputs, 1, 2)
        inputs = inputs.reshape(batch_size * 128 * 128, 6, self.lstm_inputs)
        # [batch_Size, 128*128, 6, 12]

        inputs= self.lstm(inputs)[0][:, -1, :]

        out = self.fc(inputs)

        out = out.reshape(batch_size, 128*128, self.num_classes)
        out = torch.transpose(out, 1, 2)
        out = out.reshape(batch_size, self.num_classes, 128, 128)

        return out
