import torch
import torch.nn as nn
from torch.autograd import Variable
from DynamicConv import DyConv
from sublayers import MultiHeadAttention, FeedForward, Norm


class Building_Block(nn.Module):
    def __init__(self, input_features, hidden_features, num_heads, dropout):
        super(Building_Block, self).__init__()

        self.input_features = input_features
        self.hidden_features = hidden_features

        # defaulted seq_len = 12 here
        # defaulted region size = 10 * 10 = 100
        self.embed_dim = 100 * self.hidden_features

        self.attn = MultiHeadAttention(num_heads, self.embed_dim, dropout)

        self.pos_ffn = FeedForward(self.embed_dim, dropout=dropout)

        self.norm_1 = Norm(self.embed_dim)
        self.norm_2 = Norm(self.embed_dim)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):  # here x includes c
        # input x <- (batch_size, sequence_length, pixel_num * feature_num)
        x2 = self.norm_1(x)
        new_x = x + self.dropout_1(self.attn(x2, x2, x2))
        x2 = self.norm_2(new_x)
        new_x = new_x + self.dropout_2(self.pos_ffn(x2))
        # output new_x <- (batch_size, sequence_length, pixel_num * feature_num)
        return new_x
