import torch
import os
import logging
import config
from model import Transformer, MaskedSoftmaxCELoss
from train import train_transformer
from inference import translate_one_sentence


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    if os.path.exists(log_path) is True:
        os.remove(log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def main():
    set_logger(config.log_path)

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

main()