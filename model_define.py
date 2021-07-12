#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
import sys

import torch
import math
import torch.nn as nn

# define Embedding(Tokenize)


# define Position Encoding
class PositionalEncoding_Fixed(nn.Module):
    """"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """"""
        super(PositionalEncoding_Fixed, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        if d_model > 2:
            pe[:, 0::2] = torch.sin(position * div_term)  # [:, 0::2]，[axis=0所有的数据，axis=2从0开始取值，间隔为2]
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.sin(position * div_term)  # [:, 0::2]，[axis=0所有的数据，axis=2从0开始取值，间隔为2]
            # pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """"""
        x = x + self.pe[:, x.size(1), :]
        return self.dropout(x)


class PositionalEncoding_Learnable(nn.Module):
    """"""
    def __init__(self):
        """"""
        super(PositionalEncoding_Learnable, self).__init__()
        pass

    def forward(self, X):
        pass

# define Encoder
class Encoder_TransformerEncoder(nn.Module):
    """"""
    def __init__(self, d_model, nhd=8, nly=6, dropout=0.1, hid=2048):
        """"""
        super(Encoder_TransformerEncoder, self).__init__()
        encoder = nn.TransformerEncoderLayer(d_model, nhead=nhd, dim_feedforward=hid, dropout=dropout)
        self.encoder_lays = nn.TransformerEncoder(encoder, nly, norm=nn.LayerNorm(d_model))

    def forward(self, X):
        """"""
        res = self.encoder_lays(X)
        return res

# define Decoder
class Decoder_MLP_Linear(nn.Module):
    """"""
    def __init__(self, d_model, tokenizer):
        """"""
        super(Decoder_MLP_Linear, self).__init__()
        self.tokenizer = tokenizer
        # detoken_len = self.tokenizer.detoken_len
        detoken_len = 160
        linear = nn.Linear(detoken_len, detoken_len // 2)
        relu = nn.ReLU()
        self.hidden = nn.Sequential(linear, relu)
        self.linear = nn.Linear(detoken_len // 2, 1)

    def forward(self, X):
        """"""
        batch_size = X.shape[0]
        de_token_list = []
        for i in range(batch_size):
            temp_data = X[i, :, :]
            temp_detoken = self.tokenizer.token_wrapper(temp_data, 'detoken')
            temp_detoken = temp_detoken[0, 0:160].unsqueeze(0)
            de_token_list.append(temp_detoken)
        res = torch.cat(de_token_list)
        res = self.hidden(res)
        res = self.linear(res)
        return res


class Decoder_Conv_Pooling(nn.Module):
    """"""
    def __init__(self):
        """"""
        super(Decoder_Conv_Pooling, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5))

    def forward(self, X):
        pass
# define Model
class PE_fixed_EC_transformer_DC_mlp_linear(nn.Module):
    """"""

    def __init__(self, d_model, tokenizer, nhd=8, nly=6, dropout=0.1, hid=2048):
        """"""
        self.model_name = self.__class__.__name__

        super(PE_fixed_EC_transformer_DC_mlp_linear, self).__init__()
        # position encoding
        self.position_encoding = PositionalEncoding_Fixed(d_model, dropout=dropout)

        # encoder
        self.encoder = Encoder_TransformerEncoder(d_model, nhd=nhd, nly=nly, dropout=dropout, hid=hid)

        # decoder
        self.decoder = Decoder_MLP_Linear(d_model, tokenizer)

    def forward(self, X):
        """"""
        res = self.position_encoding(X)
        res = self.encoder(res)
        res = self.decoder(res)
        return res


class PE_fixed_EC_transformer_DC_mlp_linear_with_Resconnection(nn.Module):
    """"""

    def __init__(self, d_model, tokenizer, nhd=8, nly=6, dropout=0.1, hid=2048):
        """"""
        self.model_name = self.__class__.__name__

        super(PE_fixed_EC_transformer_DC_mlp_linear_with_Resconnection, self).__init__()
        # position encoding
        self.position_encoding = PositionalEncoding_Fixed(d_model, dropout=dropout)

        # encoder
        self.encoder = Encoder_TransformerEncoder(d_model, nhd=nhd, nly=nly, dropout=dropout, hid=hid)

        # decoder
        self.decoder = Decoder_MLP_Linear(d_model, tokenizer)

    def forward(self, X):
        """"""
        x = X
        res = self.position_encoding(X)
        res = self.encoder(res)
        # add identical res-connection
        res = res + x

        res = self.decoder(res)
        return res


class PureMLP(nn.Module):
    """"""

    def __init__(self, in_dim: int = None) -> None:
        """"""
        self.model_name = self.__class__.__name__
        super(PureMLP, self).__init__()
        linear = nn.Linear
        activ = nn.Sigmoid
        # model define
        self.hidden1 = nn.Sequential(linear(in_dim, in_dim // 2), activ())
        self.hidden2 = nn.Sequential(linear(in_dim // 2, 1), activ())

    def forward(self, x: torch.tensor) -> None:
        """"""
        res = self.hidden1(x)
        res = self.hidden2(res)
        return res

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)
        return model


if __name__ == '__main__':
    pass
