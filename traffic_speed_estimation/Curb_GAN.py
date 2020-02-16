import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt
import imageio
import itertools
import numpy as np
import struct
from sa_building_block import Building_Block
from sublayers import Norm
import copy
from DynamicConv import DyConv


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Generator(nn.Module):
    # final_features is the num of features we want (usually is 1)
    def __init__(self, input_features, hidden_features, final_features, num_heads, dropout, N):
        super(Generator, self).__init__()
        self.N = N
        #self.Building_Block = Building_Block(input_features, hidden_features, num_heads, dropout)
        self.linear = nn.Linear(1, final_features)
        self.blocks = get_clones(Building_Block(1, 1, num_heads, dropout), N)
        self.norm = Norm(100 * 1)

        self.DyConv1 = DyConv(input_features, hidden_features)
        self.DyConv2 = DyConv(hidden_features, int(hidden_features / 2))
        self.DyConv3 = DyConv(int(hidden_features / 2), int(hidden_features / 4))
        self.DyConv4 = DyConv(int(hidden_features / 4), 1)

        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)
        self.bn3 = nn.BatchNorm1d(100)
        self.bn4 = nn.BatchNorm1d(100)

    def forward(self, x, adj, c):
        # input noise x <- (batch_size, sequence_length, pixel_num, init_features = 100 rand nums)
        # input adjacency matrix <- (batch_size, sequence_length, pixel_num, pixel_num)
        # input condition c <- (batch_size, sequence_length, pixel_num, condition_num)
        x = torch.cat((x, c), dim=3)

        # first x go through one GraphConvolution layer
        new_x = []
        for i in range(12):
            new_x.append(F.relu(self.bn1(self.DyConv1(x[:, i, :, :], adj[:, i, :, :]))))

        new_x = torch.stack(new_x)
        # x <- (batch_size, sequence_length, pixel_num, feature_num)
        x = new_x.permute(1, 0, 2, 3)

        new_x = []
        for i in range(12):
            new_x.append(F.relu(self.bn2(self.DyConv2(x[:, i, :, :], adj[:, i, :, :]))))

        new_x = torch.stack(new_x)
        # x <- (batch_size, sequence_length, pixel_num, feature_num)
        x = new_x.permute(1, 0, 2, 3)

        new_x = []
        for i in range(12):
            new_x.append(F.relu(self.bn3(self.DyConv3(x[:, i, :, :], adj[:, i, :, :]))))

        new_x = torch.stack(new_x)
        # x <- (batch_size, sequence_length, pixel_num, feature_num)
        x = new_x.permute(1, 0, 2, 3)

        new_x = []
        for i in range(12):
            new_x.append(F.relu(self.bn4(self.DyConv4(x[:, i, :, :], adj[:, i, :, :]))))

        new_x = torch.stack(new_x)
        # x <- (batch_size, sequence_length, pixel_num, feature_num)
        x = new_x.permute(1, 0, 2, 3)
        x = x.view(x.size()[0], x.size()[1], -1)

        # second, x go through many times building_blocks
        for i in range(self.N):
            x = self.blocks[i](x)
        '''
        # x shape: (batch_size, sequence_length, pixel_num * hidden_features)
        x = self.Building_Block(x, adj)
        '''
        x = self.norm(x)
        x = x.view(x.size()[0], x.size()[1], 100, -1)

        # x <- (batch_size, sequence_length, pixel_num, final_features)
        x = torch.tanh(self.linear(x))

        return x


class Discriminator(nn.Module):
    # final_features usually is 1, cuz we need one scalar
    def __init__(self, input_features, hidden_features, final_features, num_heads, dropout, N):
        super(Discriminator, self).__init__()
        self.N = N
        #self.Building_Block = Building_Block(input_features, hidden_features, num_heads, dropout)
        # Attention: 100 here should be adjusted if pixel num changes
        self.linear = nn.Linear(100 * 1, final_features)
        self.blocks = get_clones(Building_Block(1, 1, num_heads, dropout), N)
        self.norm = Norm(100 * 1)

        self.DyConv1 = DyConv(input_features, hidden_features)
        self.DyConv2 = DyConv(hidden_features, int(hidden_features * 2))
        self.DyConv3 = DyConv(int(hidden_features * 2), int(hidden_features * 4))
        self.DyConv4 = DyConv(int(hidden_features * 4), 1)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)
        self.bn3 = nn.BatchNorm1d(100)
        self.bn4 = nn.BatchNorm1d(100)

    def forward(self, x, adj, c):
        # input region x <- (batch_size, sequence_length, pixel_num, init_feature_num)
        # input adjacency matrix <- (batch_size, sequence_length, pixel_num, pixel_num)
        # input condition c <- (batch_size, sequence_length, pixel_num, condition_num)
        x = torch.cat((x, c), dim=3)

        # first x go through one GraphConvolution layer
        new_x = []
        for i in range(12):
            new_x.append(F.leaky_relu(self.bn1(self.DyConv1(x[:, i, :, :], adj[:, i, :, :])), 0.2))

        new_x = torch.stack(new_x)
        # x <- (batch_size, sequence_length, pixel_num, feature_num)
        x = new_x.permute(1, 0, 2, 3)

        new_x = []
        for i in range(12):
            new_x.append(F.leaky_relu(self.bn2(self.DyConv2(x[:, i, :, :], adj[:, i, :, :])), 0.2))

        new_x = torch.stack(new_x)
        # x <- (batch_size, sequence_length, pixel_num, feature_num)
        x = new_x.permute(1, 0, 2, 3)

        new_x = []
        for i in range(12):
            new_x.append(F.leaky_relu(self.bn3(self.DyConv3(x[:, i, :, :], adj[:, i, :, :])), 0.2))

        new_x = torch.stack(new_x)
        # x <- (batch_size, sequence_length, pixel_num, feature_num)
        x = new_x.permute(1, 0, 2, 3)

        new_x = []
        for i in range(12):
            new_x.append(F.leaky_relu(self.bn4(self.DyConv4(x[:, i, :, :], adj[:, i, :, :])), 0.2))

        new_x = torch.stack(new_x)
        # x <- (batch_size, sequence_length, pixel_num, feature_num)
        x = new_x.permute(1, 0, 2, 3)
        x = x.view(x.size()[0], x.size()[1], -1)

        # second, x go through many times building_blocks
        for i in range(self.N):
            x = self.blocks[i](x)

        x = self.norm(x)

        '''
        # x shape: (batch_size, sequence_length, pixel_num * hidden_features)
        x = self.Building_Block(x, adj)
        '''
        # outputs <- (batch_size, sequence_length, 1)
        outputs = torch.sigmoid(self.linear(x))
        # outputs <- (batch_size, 1)
        outputs = torch.mean(outputs, dim=1, keepdim=False)

        return outputs
