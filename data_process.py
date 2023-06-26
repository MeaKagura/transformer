import torch
import json
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import config
from tokenize_model.tokenize import chinese_tokenizer_load, english_tokenizer_load

DEVICE = config.device


class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src_text, tgt_text, src_ids, src_valid_lens, tgt_ids, tgt_valid_lens):
        self.src_text = src_text
        self.tgt_text = tgt_text
        self.train_data = [src_ids, src_valid_lens, tgt_ids, tgt_valid_lens]

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        self.index += 1
        if self.index >= 4:
            raise StopIteration
        return self.train_data[self.index]
            

class NMTDataset(Dataset):
    def __init__(self, data_path):
        self.out_en_sent, self.out_cn_sent = self.get_dataset(data_path, sort=True)
        self.sp_eng = english_tokenizer_load()
        self.sp_chn = chinese_tokenizer_load()
        self.PAD = self.sp_eng.pad_id()  # 0
        self.BOS = self.sp_eng.bos_id()  # 2
        self.EOS = self.sp_eng.eos_id()  # 3

    @staticmethod
    def len_argsort(seq):
        """传入一系列句子数据(分好词的列表形式)，按照句子长度排序后，返回排序后原来各句子在数据中的索引下标"""
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, data_path, sort=False):
        """把中文和英文按照同样的顺序排序, 以英文句子长度排序的(句子下标)顺序为基准"""
        dataset = json.load(open(data_path, 'r'))
        out_en_sent = []
        out_cn_sent = []
        for idx, _ in enumerate(dataset):
            out_en_sent.append(dataset[idx][0])
            out_cn_sent.append(dataset[idx][1])
        if sort:
            sorted_index = self.len_argsort(out_en_sent)
            out_en_sent = [out_en_sent[i] for i in sorted_index]
            out_cn_sent = [out_cn_sent[i] for i in sorted_index]
        return out_en_sent, out_cn_sent

    def __getitem__(self, idx):
        eng_text = self.out_en_sent[idx]
        chn_text = self.out_cn_sent[idx]
        return [eng_text, chn_text]

    def __len__(self):
        return len(self.out_en_sent)

    def collate_fn(self, batch):
        src_text = [x[0] for x in batch]
        tgt_text = [x[1] for x in batch]

        src_tokens = [[self.BOS] + self.sp_eng.EncodeAsIds(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + self.sp_chn.EncodeAsIds(sent) + [self.EOS] for sent in tgt_text]

        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                   batch_first=True, padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)

        src_mask = torch.ones_like(batch_input) * self.PAD
        tgt_mask = torch.ones_like(batch_target) * self.PAD
        src_valid_lens = torch.sum(batch_input != src_mask, dim=1)
        tgt_valid_lens = torch.sum(batch_target != tgt_mask, dim=1)

        return Batch(src_text, tgt_text, batch_input, src_valid_lens, batch_target, tgt_valid_lens)
