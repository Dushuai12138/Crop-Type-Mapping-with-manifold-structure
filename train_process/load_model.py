# -*- coding: utf-8 -*-
# time: 2024/1/15 10:46
# file: load_model.py
# author: Shuai


import torch
from nets.segformer import SegFormer
from nets.LSTM_segformer import LSTM_SegFormer
from nets.Unet import UNet
from nets.TFBS import LSTM_UNet
from nets.LSTM import LSTM
import torch.nn as nn


def load_model(model_name, input_features, num_classes, phi, lstm_outputs, pretrained, source_num=None, source_modelpath=None):
    if model_name == 'SegFormer':
        model = SegFormer(in_chans=input_features, num_classes=num_classes, phi=phi)

    if model_name == 'SegFormer_TLfromRGB':
        model = SegFormer(in_chans=3, num_classes=21, phi=phi, pretrained=pretrained)
        model.backbone.patch_embed1.proj = nn.Conv2d(input_features, 64, kernel_size=(7, 7), stride=(4, 4),
                                                     padding=(3, 3))
        embedding_dim = model.decode_head.linear_pred.in_channels
        model.decode_head.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=(1, 1), stride=(1, 1))

    if model_name == 'SegFormer_TLfromNE':
        model = SegFormer(in_chans=input_features, num_classes=source_num, phi=phi)
        model.load_state_dict(torch.load(source_modelpath))
        embedding_dim = model.decode_head.linear_pred.in_channels
        model.decode_head.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    if model_name == 'Unet':
        model = UNet(num_classes=num_classes, pretrained=False, backbone='resnet50')
        model.resnet.conv1 = nn.Conv2d(input_features, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                       bias=False)

    if model_name == 'Unet_TLfromRGB':
        model = UNet(num_classes=num_classes, pretrained=pretrained, backbone='resnet50')
        model.resnet.conv1 = nn.Conv2d(input_features, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                       bias=False)

    if model_name == 'Unet_TLfromNE':
        model = UNet(num_classes=source_num, pretrained=False, backbone='resnet50')
        model.resnet.conv1 = nn.Conv2d(input_features, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                       bias=False)
        model.load_state_dict(torch.load(source_modelpath))
        out = model.final.in_channels
        model.final = nn.Conv2d(out, num_classes, 1)

    if model_name == 'TFBS':
        model = LSTM_UNet(lstm_inputs=input_features, lstm_outputs=lstm_outputs, num_classes=num_classes)


    if model_name == 'TFBS_TLfromNE':
        model = LSTM_UNet(lstm_inputs=input_features, lstm_outputs=lstm_outputs, num_classes=source_num)
        model.load_state_dict(torch.load(source_modelpath))
        model.unet.pred = torch.nn.Conv2d(64, num_classes, 3, 1, 1)

    if model_name == 'LSTM':
        model = LSTM(lstm_inputs=input_features, lstm_outputs=lstm_outputs, num_classes=num_classes)

    if model_name == 'LSTM_TLfromNE':
        model = LSTM(lstm_inputs=input_features, lstm_outputs=lstm_outputs, num_classes=source_num)
        model.load_state_dict(torch.load(source_modelpath))
        model.fc = nn.Linear(lstm_outputs, num_classes)
        model.num_classes = num_classes

    return model
