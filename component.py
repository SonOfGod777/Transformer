# coding=utf-8
import torch
from torch import nn
from torch.autograd import Variable
import pandas as pd
import numpy as np
import math, copy


class Config(object):
    def __init__(self, vocab_size=None,
                 hidden_size=768,
                 num_hidden_layers=12,
                 intermediate_size=3072,
                 hidden_dropout_prob=0.1,
                 attention_dropout_prob=0.1,
                 max_position_ids=5000,
                 type_vocab_size=2,
                 dropout=0.1,
                 eps=1e-12,
                 attention_heads=8):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.max_position_ids = max_position_ids
        self.type_vocab_size = type_vocab_size
        self.dropout = dropout
        self.eps = eps
        self.attention_heads = attention_heads


class LayerNorm(nn.Module):
    def __init__(self, config):
        super(LayerNorm, self).__init__()
        feature = config.hidden_size
        self.eps = config.eps
        self.weight = nn.Parameter(torch.ones(feature))
        self.bias = nn.Parameter(torch.zeros(feature))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        return self.weight * (x-mean) / (std+self.eps) + self.bias


class TokenEmbedding(nn.Module):
    def __init__(self, config):
        super(TokenEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.hidden_size = config.hidden_size

    def forward(self, input_ids):
        token_embedding = self.token_embedding(input_ids)
        return token_embedding * math.sqrt(self.hidden_size)


class PositionEmbedding(nn.Module):
    def __init__(self, config):
        super(PositionEmbedding, self).__init__()
        max_len = config.max_position_ids
        hidden_size = config.hidden_size

        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        dim_size = torch.exp(torch.arange(0, hidden_size, 2) * -(torch.log(10000) / hidden_size))
        pe[:, 0::2] = torch.sin(position * dim_size)
        pe[:, 1::2] = torch.cos(position * dim_size)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return Variable(self.pe[:, :x.size(1)], requires_grad=False)


class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.LayerNorm = LayerNorm(config)
        self.position_embedding = PositionEmbedding(config)
        self.tokenEmbedding = TokenEmbedding(config)

    def forward(self, input_ids):
        token_embedding = self.tokenEmbedding(input_ids)
        position_embedding = self.position_embedding(token_embedding)
        embedding = token_embedding + position_embedding
        return self.LayerNorm(self.dropout(embedding))


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# mask 加到 q*v 结果上 ； en_mask:[batch,1,1,leng]  de_mask:[batch,1,leng,leng]
def attention(query, key, value, mask=None, dropout=None):
    attention_hidden_size = query.size(-1)
    score = torch.matmul(query, key.transpose(-1, -2))
    score = score / torch.sqrt(attention_hidden_size)
    if mask is not None:
        mask = (1.0 - mask.float()) * 1e9
        score -= mask
    prob = nn.Softmax(dim=-1)(score)
    if dropout is not None:
        prob = dropout(prob)
    return torch.matmul(prob, value), prob


class MultiAttention(nn.Module):
    def __init__(self, config):
        super(MultiAttention, self).__init__()
        self.hidden_size = config.hidden_size
        self.atten_heads = config.atten_heads
        assert self.hidden_size % self.atten_heads == 0
        self.atten_size = self.hidden_size // self.atten_heads
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.linears = clones(self.linear, 4)

    def forward(self, query, key, values, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batchs = query.size(0)
        query, key, values = [linear(x).view(batchs, -1, self.atten_heads, self.atten_size).transpose(1, 2) for linear, x in zip(self.linears, [query, key, values])]
        x, atten_q = attention(query, key, values, mask)
        x = x.transpose(1, 2).contiguous().view(batchs, -1, self.atten_size * self.atten_heads)
        return self.linears[-1](x)


class Dense(nn.Module):
    def __init__(self, config):
        super(Dense, self).__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.linear2 = nn.Linear(self.intermediate_size, self.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm = LayerNorm(config)
        self.mul_attention = MultiAttention(config)
        self.dense = Dense(config)

    def forward(self, input_em, sour_mask=None):
        input_em = self.norm(input_em + self.dropout(self.mul_attention(input_em, input_em, input_em, sour_mask)))
        return self.norm(input_em + self.dropout(self.dense(input_em)))


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm = LayerNorm(config)
        self.self_attention = MultiAttention(config)
        self.sour_attention = MultiAttention(config)
        self.dense = Dense(config)

    def forward(self, de_input, en_output, targ_mask=None, sour_mask=None):
        de_output = self.norm(de_input + self.dropout(self.self_attention(de_input, de_input, de_input, targ_mask)))
        de_output = self.norm(de_output + self.dropout(self.sour_attention(de_output, en_output, en_output, sour_mask)))
        return self.norm(de_output + self.dropout(self.dense(de_output)))


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = EncoderLayer(config)
        self.layers = clones(self.layer, config.num_hidden_layers)
        self.all_layers = []

    def forward(self, input_em, sour_mask, return_all_layers=True):
        for layer in self.layers:
            input_em = layer(input_em, sour_mask)

            if return_all_layers:
                self.all_layers.append(input_em)
        if not return_all_layers:
            self.all_layers.append(input_em)
        return self.all_layers




























