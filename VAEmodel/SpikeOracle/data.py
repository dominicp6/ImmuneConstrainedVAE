import numpy as np

from tqdm.auto import tqdm

import json
import os

from collections import defaultdict

import Bio
from Bio import SeqIO

import torch
from torch.utils.data import Dataset


class StandardDataset(Dataset):

    def __init__(self, seq_len, max_seq_len=None, pad_to=None, conv=False, filename=None):
        _max_seq_len = seq_len if max_seq_len is None else max_seq_len
        _pad_to = seq_len if pad_to is None else pad_to
        self.tok = Tokenizer(f"..{os.sep}data{os.sep}One_hot_2.json", _max_seq_len, _pad_to, conv)

        self.seq_len = seq_len
        self._max_seq_len = _max_seq_len
        self._pad_to = _pad_to
        self._conv = conv

        self.viral_seqs = []
        self.max_len = 0

        self.seq_immuno_cat = None
        self.seq_immuno_cat_tokens = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}

        if filename is not None:
            with open(filename, "r") as file:
                self.viral_seqs = file.readlines()
            for idx, seq in enumerate(self.viral_seqs):
                self.viral_seqs[idx] = seq.replace("\n", "")

    def load_from_fasta(self, fasta_filename):
        viral_seq_dict = defaultdict(lambda: 0)
        self.viral_seqs = []

        for record in tqdm(SeqIO.parse(fasta_filename, "fasta")):
            seq = str(record.seq)
            viral_seq_dict[seq] += 1

        for seq in viral_seq_dict.keys():
            if len(seq) == self.seq_len:
                self.viral_seqs.append(seq)
            if len(seq) > self.max_len:
                self.max_len = len(seq)

    def splitoff(self, cnt):
        selection = np.random.choice(self.viral_seqs, cnt)
        ds = StandardDataset(self.seq_len, self.max_seq_len, self.pad_to, self.conv)
        ds.viral_seqs = selection
        self.viral_seqs = [seq for seq in self.viral_seqs if seq not in selection]
        return ds

    def save_to_file(self, filename):
        text = ""
        for seq in tqdm(self.viral_seqs):
            text += seq + "\n"

        with open(filename, "w") as file:
            file.write(text)

    @property
    def max_seq_len(self):
        return self._max_seq_len

    @max_seq_len.setter
    def max_seq_len(self, v):
        self.tok.max_seq_len = v
        self._max_seq_len = v

    @property
    def pad_to(self):
        return self._pad_to

    @pad_to.setter
    def pad_to(self, v):
        self.tok.pad_to = v
        self._pad_to = v

    @property
    def conv(self):
        return self._conv

    @conv.setter
    def conv(self, v):
        self.tok.conv = v
        self._conv = v

    def __getitem__(self, idx):
        seq = self.tok.tokenize(self.viral_seqs[idx])
        return (seq, 1) if self.seq_immuno_cat is None else (seq,
                                                             torch.tensor(self.seq_immuno_cat_tokens[
                                                                              self.seq_immuno_cat[self.viral_seqs[idx]]
                                                                          ]))

    def __len__(self):
        return len(self.viral_seqs)


class Tokenizer:
    def __init__(self, filename, max_seq_len, pad_to, conv):
        f = open(filename)
        self.enc_dict = json.load(f)
        self.dec_dict = {}
        self.aa_to_idx = {}
        for key, value in self.enc_dict.items():
            if len(key) == 1:
                v = int(torch.max(torch.tensor(value), dim=-1).indices)
                self.dec_dict[v] = key
                self.aa_to_idx[key] = v

        self.max_seq_len = max_seq_len
        self.pad_to = pad_to
        self.conv = conv

    def tokenize(self, sequence):
        enc = []
        sequence = sequence.ljust(self.pad_to, '-')
        for aa in sequence[:self.max_seq_len]:
            enc.append(self.enc_dict[aa])

        t = torch.tensor(enc)
        if self.conv:
            t = t.permute(1, 0)
        return t.float()

    def decode(self, batch):
        batch_size, seq_len, aa_dim = batch.size()

        batch_seq = []

        h = torch.max(batch, dim=-1).indices

        for b in range(batch_size):
            seq = ""
            for s in range(seq_len):
                seq += self.dec_dict[int(h[b][s])]
            batch_seq.append(seq)

        return batch_seq
