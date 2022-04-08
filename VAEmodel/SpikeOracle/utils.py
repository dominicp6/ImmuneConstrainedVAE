import numpy as np

from tqdm.auto import tqdm

import Bio
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def write_seqs_to_fasta(seqs, filename):
    records = []

    for seq, cnt in seqs.items():
        record = SeqRecord(Seq(seq), id=f"", description=f"{cnt}")
        record = SeqRecord(Seq(seq.replace("-", "")), id=f"", description=f"{cnt}")
        records.append(record)

    fd = open(filename, "w")
    SeqIO.write(records, fd, format="fasta")
    fd.close()

    return records


def calc_PWM(seqs, aa_to_idx):
    seq_cnt = len(seqs)
    seq_len = len(seqs[0])

    pwm = np.zeros((max(aa_to_idx.values()) + 1, seq_len))

    for seq in tqdm(seqs):
        for pos, aa in enumerate(seq):
            pwm[aa_to_idx[aa], pos] += 1

    return pwm / seq_cnt


def calc_entropy(dist):
    e = 0
    for p in dist:
        if p > 0:
            e -= p * np.log2(p)

    return e


def calc_entropy_vector(seqs, aa_to_idx):
    pwm = calc_PWM(seqs, aa_to_idx)

    _, seq_len = pwm.shape
    ev = np.zeros(seq_len)

    for pos in range(seq_len):
        ev[pos] = calc_entropy(pwm[:, pos])

    return ev
