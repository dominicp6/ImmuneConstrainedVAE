from collections import defaultdict

import pandas as pd
from Bio import SeqIO
import operator
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def diff_letters(a,b):
    return sum ( a[i] != b[i] for i in range(len(a)) )

def analyse_mutation_distribution(database):
    canonical_amino_acid_order = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L",
                                  "K", "M", "F", "P", "S", "T", "W", "Y", "V", "-"]
    residue_distributions = defaultdict(
        lambda: {"A": 0, "R": 0, "N": 0, "D": 0, "C": 0, "Q": 0, "E": 0, "G": 0, "H": 0, "I": 0, "L": 0,
                 "K": 0, "M": 0, "F": 0, "P": 0, "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0, "-": 0})

    # read in and preprocess database file
    fasta_sequences = SeqIO.parse(database, "fasta")
    list_of_sequences = [seq.seq for seq in fasta_sequences]
    fasta_sequences = SeqIO.parse(database, "fasta")
    list_of_counts = [int(seq.id.split('|')[0]) for seq in fasta_sequences]
    total_counts = sum(list_of_counts)
    L = len(list_of_sequences[0])  # get the length of a sequence

    for residue_index in tqdm(range(L)):
        for seq_idx, seq in enumerate(list_of_sequences):
            amino_acid = seq[residue_index]
            residue_distributions[residue_index][amino_acid] += list_of_counts[seq_idx]

    mode_sequence = "".join([max(residue_distributions[residue_index].items(), key=operator.itemgetter(1))[0] for residue_index in range(L)])

    list_of_number_of_mutations = []
    for seq in list_of_sequences:
        list_of_number_of_mutations.append(diff_letters(seq, mode_sequence))

    plt.hist(list_of_number_of_mutations, bins=30, range=[0, 30])
    print(pd.DataFrame(list_of_number_of_mutations).describe())
    plt.show()
    print(np.mean(list_of_number_of_mutations))
    print(1/np.mean(list_of_number_of_mutations))


if __name__ == "__main__":
    print(analyse_mutation_distribution(database='../data/spikeprot_final_dataset.afa'))