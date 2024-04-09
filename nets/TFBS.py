# -*- coding: utf-8 -*-
# time: 2024/1/29 20:38
# file: TFBS.py
# author: Shuai


import torch
import torch.nn as nn
from torch.nn import functional as F


class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)



class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.Up(up)
        return torch.cat((x, r), 1)


class Unet(nn.Module):

    def __init__(self, input=3, num_classes=1000):
        super(Unet, self).__init__()
        self.C1 = Conv(input, 64)
        self.D1 = DownSampling(64)
        self.C2 = Conv(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = Conv(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = Conv(512, 1024)

        self.U1 = UpSampling(1024)
        self.C6 = Conv(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = Conv(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = Conv(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = Conv(128, 64)

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, num_classes, 3, 1, 1)

    def forward(self, x):
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))


        return self.Th(self.pred(O4))


class LSTM_UNet(nn.Module):
    def __init__(self, lstm_inputs=12, lstm_outputs=64, num_classes=4):
        super(LSTM_UNet, self).__init__()
        self.lstm_inputs = lstm_inputs
        self.lstm_outputs = lstm_outputs
        self.lstm = nn.LSTM(input_size=lstm_inputs, hidden_size=lstm_outputs, num_layers=1, batch_first=True, bidirectional=False)
        self.unet = Unet(input=lstm_outputs, num_classes=num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        inputs = x.view(batch_size, 6, self.lstm_inputs, -1).permute(0, 3, 1, 2)
        inputs = inputs.reshape(batch_size * 128 * 128, 6, self.lstm_inputs)

        # [batch_Size, 128*128, 6, 12]
        inputs = self.lstm(inputs)[0][:, -1, :]

        outputs = inputs.view(batch_size, 128 * 128, self.lstm_outputs).permute(0, 2, 1)
        outputs = outputs.view(batch_size, self.lstm_outputs, 128, 128)

        out = self.unet(outputs)

        return out
