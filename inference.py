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

from tokenize_model.tokenize import chinese_tokenizer_load, english_tokenizer_load

import config


def greedy_decode(model, src_ids, src_valid_lens, src_tokenizer, tgt_tokenizer, device, max_len=config.max_len):
    model.to(device)
    model.eval()
    src_ids, src_valid_lens = src_ids.to(device), src_valid_lens.to(device)
    enc_output = model.encoder(src_ids, src_valid_lens)
    dec_input = torch.Tensor([src_tokenizer.bos_id()]).reshape(1, 1).to(device).type(torch.int64)
    state = (enc_output, src_valid_lens, [None] * model.num_layers)
    dec_outputs = []
    for i in range(max_len):
        dec_input, state = model.decoder(dec_input, state)
        pred_ids = torch.argmax(dec_input, dim=2)
        if pred_ids[0][0] == tgt_tokenizer.eos_id():
            break
        pred_token = tgt_tokenizer.decode_ids(pred_ids[0])
        dec_outputs.append(pred_token)
        dec_input = pred_ids
    return ' '.join(dec_outputs)


def batch_greedy_decode(model, src_ids, src_valid_lens, src_tokenizer, tgt_tokenizer, device, max_len=config.max_len):
    model.to(device)
    model.eval()
    src_ids, src_valid_lens = src_ids.to(device), src_valid_lens.to(device)
    enc_output = model.encoder(src_ids, src_valid_lens)
    dec_input = torch.Tensor([src_tokenizer.bos_id()]).repeat(src_ids.shape[0], 1).to(device).type(torch.int64)
    state = (enc_output, src_valid_lens, [None] * model.num_layers)
    batch_size, count = src_ids.shape[0], 0
    dec_outputs = [[] for _ in range(batch_size)]
    for _ in range(max_len):
        dec_input, state = model.decoder(dec_input, state)
        pred_ids = torch.argmax(dec_input, dim=2)
        for i in range(batch_size):
            if pred_ids[i][0] == tgt_tokenizer.eos_id():
                count += 1
            pred_token = tgt_tokenizer.decode_ids(pred_ids[i])
            dec_outputs[i].append(pred_token)
        dec_input = pred_ids
    return [' '.join(dec_output) for dec_output in dec_outputs]


# 计算 BLEU 得分
def bleu(pred_text: str, label_text: str, k: int):
    pred_tokens = pred_text.split(' ')
    label_tokens = label_text.split(' ')
    pred_len, label_len = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - label_len / pred_len))
    k = min(k, pred_len)
    for n in range(1, k + 1):
        label_n_gram = collections.defaultdict(int)
        num_matches = 0
        for i in range(len(label_tokens) - n + 1):
            label_n_gram[' '.join(label_tokens[i: i+n])] += 1
        for i in range(len(pred_tokens) - n + 1):
            if label_n_gram[' '.join(pred_tokens[i: i+n])] > 0:
                label_n_gram[' '.join(pred_tokens[i: i+n])] -= 1
                num_matches += 1
        score *= pow(num_matches / (pred_len - n + 1), pow(0.5, n))
    return score


def translate_one_sentence(model, device):
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
    sp_en = english_tokenizer_load()
    sp_cn = chinese_tokenizer_load()
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .','I\'m not leaving you .', 'You shouldn\'t have done that .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .', 'Je ne te quitte pas .', 'Tu n\'aurais pas dû faire ça .']
    for src_text, tgt_text in zip(engs, fras):
        src_ids = sp_en.EncodeAsIds(src_text)
        pred = greedy_decode(model, src_ids, len(src_ids), sp_en, sp_cn, device, max_len=config.max_len)
        print(f'source: \'{src_text}\', pred: \'{pred}\', bleu: {bleu(pred, tgt_text, 2)}')
