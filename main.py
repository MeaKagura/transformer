import torch

import config
from model import Transformer, MaskedSoftmaxCELoss
from train import train_transformer
from inference import translate_one_sentence


num_hiddens, num_layers, dropout = config.d_model, config.n_layers, config.dropout
key_size, query_size, value_size, num_heads = config.d_k, config.d_q, config.d_v, config.n_heads
ffn_input, ffn_hiddens = config.d_model, config.d_ff
ln_shape = [config.d_model]
lr, num_epochs, device = config.lr, config.epoch_num, config.device

model = Transformer(config.src_vocab_size, config.tgt_vocab_size, num_layers, key_size, query_size,
                    value_size, num_hiddens, num_heads, ffn_input, ffn_hiddens,
                    ln_shape, dropout, bias=False)
loss_fn = MaskedSoftmaxCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

train_transformer(model, loss_fn, optimizer, num_epochs, device)

translate_one_sentence(model, device)