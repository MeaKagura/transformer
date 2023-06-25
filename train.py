import collections
import math
import re
import os
import logging
import numpy as np
import torch
import d2l.torch as d2l
import sacrebleu
import tqdm
from torch import nn, Tensor
from torch.functional import F
from torch.utils.data import DataLoader

import config
from data_process import NMTDataset
from inference import greedy_decode, batch_greedy_decode
from tokenize_model.tokenize import chinese_tokenizer_load, english_tokenizer_load


def train_transformer(model, loss_fn, optimizer, num_epochs, device):
    train_dataset = NMTDataset(config.train_data_path)
    dev_dataset = NMTDataset(config.dev_data_path)
    test_dataset = NMTDataset(config.test_data_path)

    logging.info("-------- Dataset Build! --------")

    train_iter = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                            collate_fn=train_dataset.collate_fn)
    dev_iter = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                          collate_fn=dev_dataset.collate_fn)
    test_iter = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                           collate_fn=test_dataset.collate_fn)

    logging.info("-------- Get Dataloader! --------")

    model.to(device)
    model.train()
    best_bleu_score = 0.0
    early_stop = config.early_stop
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        timer = d2l.Timer()
        for batch in train_iter:
            X, X_valid_lens, Y, Y_valid_lens = [x.to(device) for x in batch]
            Y_hat = model(X, Y[:, :-1], X_valid_lens)
            loss = loss_fn(Y_hat, Y[:, 1:], Y_valid_lens)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss, Y_valid_lens.sum())
        bleu_score = evaluate(model, dev_iter, device)
        logging.info(f'epoch: {epoch + 1:.3f}, loss: {metric[0] / metric[1]:.3f}, '
                     f'speed: {metric[1] / timer.stop():.3f} tokens/sec on {device}, '
                     f'bleu score: {bleu_score}')

        # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        if bleu_score > best_bleu_score:
            torch.save(model.state_dict(), config.model_path)
            best_bleu_score = bleu_score
            early_stop = config.early_stop
            logging.info("-------- Save Best Model! --------")
        else:
            early_stop -= 1
            logging.info("Early Stop Left: {}".format(early_stop))
        if early_stop == 0:
            logging.info("-------- Early Stop! --------")
            break


def evaluate(model, dev_iter, device):
    model.to(device)
    model.eval()
    sp_en = english_tokenizer_load()
    sp_cn = chinese_tokenizer_load()
    pred_texts = []
    tgt_texts = []
    for batch in dev_iter:
        X, X_valid_lens, Y, Y_valid_lens = [x.to(device) for x in batch]
        pred_text = batch_greedy_decode(model, X, X_valid_lens, sp_en, sp_cn, device)
        pred_texts.extend(pred_text)
        tgt_texts.extend(batch.tgt_text)
    bleu_score = sacrebleu.corpus_bleu(pred_texts, [tgt_texts], tokenize='zh')
    return float(bleu_score.score)
