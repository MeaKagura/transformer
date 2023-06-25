import collections
import math
import re
import os
import numpy as np
import torch
import d2l.torch as d2l
from torch import nn, Tensor
from torch.functional import F
from torch.utils import data
from typing import List, Tuple, Optional


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens: int, dropout: int, max_steps=1000):
        super().__init__()
        # i: [max_steps, 1]
        i = torch.arange(max_steps, dtype=torch.float32).reshape(-1, 1)
        # j: [1, num_hiddens/2]
        j = torch.arange(0, num_hiddens, step=2, dtype=torch.float32).reshape(1, -1)
        # X: [max_steps, num_hiddens/2]
        X = i / torch.pow(10000, j / num_hiddens)
        # P: [max_steps, num_hiddens]
        self.P = torch.zeros(max_steps, num_hiddens)
        self.P[:, 0::2] = torch.sin(X)
        self.P[:, 1::2] = torch.cos(X)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: Tensor) -> Tensor:
        """
        :param X: [batch_size, num_steps, num_hiddens]
        :return: [batch_size, num_steps, num_hiddens]
        """
        X = X + self.P[:X.shape[1], :].unsqueeze(0).to(X.device)
        X = self.dropout(X)
        return X


# 点乘注意力
def masked_softmax(attention_scores: Tensor, valid_lens: Tensor) -> Tensor:
    """
    :param attention_scores: [batch_size, num_queries, num_keys]
    :param valid_lens: [batch_size,] or [batch_size, num_queries]
    :return attention_weights: [batch_size, num_queries, num_keys]
    """
    if valid_lens is None:
        return F.softmax(attention_scores, dim=-1)
    else:
        shape = attention_scores.shape
        if valid_lens.dim() == 1:
            valid_lens = valid_lens.repeat_interleave(shape[1]).reshape(-1, shape[1])
        mask = torch.arange(shape[2])[None, None, :].to(valid_lens.device) < valid_lens[:, :, None]
        attention_scores[~mask] = -1e6
        return F.softmax(attention_scores, dim=-1)


class DotProductAttention(nn.Module):
    def __init__(self, dropout: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor,
                valid_lens: Tensor) -> Tensor:
        """
        :param queries: [batch_size, num_queries, queries_size]
        :param keys: [batch_size, num_keys, keys_size]
        :param values: [batch_size, num_values, values_size]
        :param valid_lens: [batch_size,] or [batch_size, num_queries]
        :return attention_results: [batch_size, num_queries, values_size]
        """
        # attention_scores: [batch_size, num_queries, num_keys]
        attention_scores = torch.bmm(queries, keys.transpose(1, 2))
        attention_scores = attention_scores / math.sqrt(queries.shape[-1])
        # attention_weights: [batch_size, num_queries, num_keys]
        attention_weights = masked_softmax(attention_scores, valid_lens)
        return torch.bmm(self.dropout(attention_weights), values)


# 多头注意力
def transpose_qkv(qkv: Tensor, num_heads: int):
    """
    为了多注意力头的并行计算而变换形状
    :param qkv: [batch_size, num_steps, num_hiddens]
    :param num_heads: int
    :return qkv: [batch_size * num_heads, num_steps, num_hiddens/num_heads]
    """
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], num_heads, -1)
    qkv = qkv.permute(0, 2, 1, 3)
    qkv = qkv.reshape(-1, qkv.shape[2], qkv.shape[3])
    return qkv


def transpose_results(results: Tensor, num_heads: int):
    """
    为了多注意力头的并行计算而变换形状
    :param results: [batch_size * num_heads, num_steps, num_hiddens/num_heads]
    :param num_heads: int
    :return: qkv: [batch_size, num_steps, num_hiddens]
    """
    results = results.reshape(-1, num_heads, results.shape[1], results.shape[2])
    results = results.permute(0, 2, 1, 3)
    results = results.reshape(results.shape[0], results.shape[1], -1)
    return results


class MultiHeadAttention(nn.Module):
    def __init__(self, query_size: int, key_size: int, value_size: int,
                 num_hiddens: int, num_heads: int, dropout: int, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, valid_lens: Tensor) -> Tensor:
        """
        :param queries: [batch_size, num_queries, query_size]
        :param keys: [batch_size, num_keys, key_size]
        :param values: [batch_size, num_values, value_size]
        :param valid_lens: [batch_size,] or [batch_size, num_queries]
        :return attention_results: [batch_size, num_queries, num_hiddens]
        """
        # q/k/v: [batch_size * num_heads, num_q/k/v, num_hiddens/num_heads]
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        # valid_lens: [batch_size * num_heads,] or [batch_size, * num_heads, num_queries]
        if valid_lens is not None:
            valid_lens = valid_lens.repeat_interleave(self.num_heads, dim=0)
        # attention_results: [batch_size * num_heads, num_queries, num_hiddens/num_heads]
        attention_results = self.attention(queries, keys, values, valid_lens)
        # attention_results: [batch_size, num_queries, num_hiddens]
        attention_results = transpose_results(attention_results, self.num_heads)
        attention_results = self.W_o(attention_results)
        return attention_results


# 残差连接&归一化
class AddNorm(nn.Module):
    def __init__(self, ln_shape: List[int], dropout: int):
        super().__init__()
        self.ln = nn.LayerNorm(ln_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        :param X: [batch_size, num_steps, num_hiddens]
        :param Y: [batch_size, num_steps, num_hiddens]
        :return Y: [batch_size, num_steps, num_hiddens]
        """
        Y = self.dropout(Y)
        Y = X + Y
        Y = self.ln(Y)
        return Y


# 基于位置的前馈网络
class PositionWiseFFN(nn.Module):
    def __init__(self, num_inputs: int, num_hiddens: int, num_outputs: int):
        super().__init__()
        self.dense1 = nn.Linear(num_inputs, num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(num_hiddens, num_outputs)

    def forward(self, X: Tensor) -> Tensor:
        """
        :param X: [batch_size, num_steps, num_inputs]
        :return: [batch_size, num_steps, num_outputs]
        """
        X = self.dense1(X)
        X = self.relu(X)
        X = self.dense2(X)
        return X


# 编码器块
class EncoderBlock(nn.Module):
    def __init__(self, query_size: int, key_size: int, value_size: int, num_hiddens: int,
                 num_heads: int, ffn_inputs: int, ffn_hiddens: int,
                 ln_shape: List[int], dropout: int, bias: bool):
        super().__init__()
        self.attention = MultiHeadAttention(query_size, key_size, value_size, num_hiddens,
                                            num_heads, dropout, bias=bias)
        self.add_norm1 = AddNorm(ln_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_inputs, ffn_hiddens, num_hiddens)
        self.add_norm2 = AddNorm(ln_shape, dropout)

    def forward(self, X: Tensor, valid_lens: Tensor) -> Tensor:
        """
        :param X: [batch_size, num_steps, query_size and key_size and value_size]
        :param valid_lens: [batch_size,]
        :return: [batch_size, num_steps, num_hiddens]
        """
        Y = self.attention(X, X, X, valid_lens)
        X = self.add_norm1(X, Y)
        Y = self.ffn(X)
        return self.add_norm2(X, Y)


# 解码器块
class DecoderBlock(nn.Module):
    def __init__(self, query_size: int, key_size: int, value_size: int, num_hiddens: int,
                 num_heads: int, ffn_inputs: int, ffn_hiddens: int,
                 ln_shape: List[int], dropout: int, bias: bool, index: int):
        super().__init__()
        self.index = index
        self.attention1 = MultiHeadAttention(query_size, key_size, value_size, num_hiddens,
                                             num_heads, dropout, bias=bias)
        self.add_norm1 = AddNorm(ln_shape, dropout)
        self.attention2 = MultiHeadAttention(query_size, key_size, value_size, num_hiddens,
                                             num_heads, dropout, bias=bias)
        self.add_norm2 = AddNorm(ln_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_inputs, ffn_hiddens, num_hiddens)
        self.add_norm3 = AddNorm(ln_shape, dropout)

    def forward(self, X: Tensor, state: Tuple[Tensor, Tensor, List[Optional[Tensor]]]) \
            -> Tuple[Tensor, Tuple[Tensor, Tensor, List[Tensor]]]:
        """
        :param X: [batch_size, num_steps, query_size and key_size and value_size]
        :param state: List[enc_outputs, enc_valid_lens, List[dec_outputs]]
        :return: [batch_size, num_steps, num_hiddens],
                 List[enc_outputs, enc_valid_lens, List[dec_outputs]]
        """
        enc_outputs, enc_valid_lens, _ = state
        if state[2][self.index] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.index], X), dim=1)
        state[2][self.index] = key_values
        if self.training:
            dec_valid_lens = torch.arange(1, X.shape[1] + 1).repeat(X.shape[0], 1)
        else:
            dec_valid_lens = None
        Y = self.attention1(X, key_values, key_values, dec_valid_lens)
        X = self.add_norm1(X, Y)
        Y = self.attention2(X, enc_outputs, enc_outputs, enc_valid_lens)
        X = self.add_norm2(X, Y)
        Y = self.ffn(X)
        return self.add_norm3(X, Y), state


# 编码器
class Encoder(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, query_size: int, key_size: int,
                 value_size: int, num_hiddens: int, num_heads: int, ffn_inputs: int,
                 ffn_hiddens: int, ln_shape: List[int], dropout: int, bias: bool):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.add_module(f'block {i}',
                                   EncoderBlock(query_size, key_size, value_size,
                                                num_hiddens, num_heads, ffn_inputs,
                                                ffn_hiddens, ln_shape, dropout, bias))

    def forward(self, X: Tensor, valid_lens: Tensor) -> Tensor:
        """
        :param X: [batch_size, num_steps]
        :param valid_lens: [batch_size,]
        :return: [batch_size, num_steps, num_hiddens]
        """
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for block in self.blocks:
            X = block(X, valid_lens)
        return X


# 解码器
class Decoder(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, query_size: int, key_size: int,
                 value_size: int, num_hiddens: int, num_heads: int, ffn_inputs: int,
                 ffn_hiddens: int, ln_shape: List[int],
                 dropout: int, bias: bool):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.add_module(f'block {i}',
                                   DecoderBlock(query_size, key_size, value_size,
                                                num_hiddens, num_heads, ffn_inputs,
                                                ffn_hiddens, ln_shape, dropout, bias, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def forward(self, X: Tensor, state: Tuple[Tensor, Tensor, List[Tensor]]) \
            -> Tuple[Tensor, Tuple[Tensor, Tensor, List[Tensor]]]:
        """
        :param X: [batch_size, num_steps]
        :param state: List[enc_outputs, enc_valid_lens, List[dec_outputs]]
        :return: [batch_size, num_steps, vocab_size]
        """
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for block in self.blocks:
            X, state = block(X, state)
        X = self.dense(X)
        return X, state


class Transformer(nn.Module):
    def __init__(self, src_size: int, tgt_size: int, num_layers: int, query_size: int,
                 key_size: int, value_size: int, num_hiddens: int, num_heads: int,
                 ffn_inputs: int, ffn_hiddens: int, ln_shape: List[int],
                 dropout: int, bias: bool):
        super().__init__()
        self.num_layers = num_layers
        self.encoder = Encoder(src_size, num_layers, query_size, key_size, value_size,
                               num_hiddens, num_heads, ffn_inputs, ffn_hiddens,
                               ln_shape, dropout, bias)
        self.decoder = Decoder(tgt_size, num_layers, query_size, key_size, value_size,
                               num_hiddens, num_heads, ffn_inputs, ffn_hiddens,
                               ln_shape, dropout, bias)

    def forward(self, X: Tensor, Y: Tensor, enc_valid_lens: Tensor):
        """
        :param X: [batch_size, num_src_steps]
        :param Y: [batch_size, num_tgt_steps]
        :param enc_valid_lens: [batch_size,]
        :return: [batch_size, num_steps, vocab_size]
        """
        enc_output = self.encoder(X, enc_valid_lens)
        state = (enc_output, enc_valid_lens, [None] * self.num_layers)
        dec_output, state = self.decoder(Y, state)
        return dec_output


# # 损失函数
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred: Tensor, target: Tensor, valid_len: Tensor) -> Tensor:
        # pred: [batch_size, num_steps, num_inputs]
        # target: [batch_size, num_steps]
        weights = torch.ones(target.shape).to(pred.device)
        mask = torch.arange(target.shape[1], device=pred.device)[None,:] < valid_len[:, None]
        weights[~mask] = 0
        self.reduction = 'none'
        unweighted_loss = super().forward(pred.permute(0, 2, 1), target)
        weighted_loss = (unweighted_loss * weights).mean(dim=1).sum()
        return weighted_loss






