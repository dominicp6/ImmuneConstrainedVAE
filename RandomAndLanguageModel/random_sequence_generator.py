"""
Generates "random" spike protein sequences using different methods.

NB: 1_in_500_cleaned_unique.afa has 8880 sequences, of which 2274 are unique
"""

import numpy as np
import random
from Bio import SeqIO
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from tqdm import tqdm
from collections import defaultdict

sequence_length = 1000

valid_residue_types = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q',
                       'R', 'S', 'T', 'V', 'W', 'Y', '-']


def generate_random_ngram_sequences(fasta_file, N, L, outfile, n=3):
    text_seed = 'MFVFLVLLPLVSSQCVN'
    lm = MLE(n)
    fasta = SeqIO.parse(fasta_file, 'fasta')

    print('Reading data file')
    sequence_data = []
    for sequence in fasta:
        sequence_data.append([letter for letter in sequence.seq])

    print('Training model')
    train, vocab = padded_everygram_pipeline(n, sequence_data)
    lm.fit(train, vocab)

    print('Generating sequences')
    with open(outfile, "w") as f:
        for _ in tqdm(range(N)):
            while True:
                seq = lm.generate(L - (n - 1), text_seed='MFVFLVLLPLVSSQCVN'[0:n - 1])
                for aa in text_seed[0:n - 1][::-1]:
                    seq.insert(0, aa)
                seq = "".join(seq)
                if "</s>" not in seq:  # only generate sequences that dont have a stop symbol in them
                    print(">", file=f)
                    print(seq, file=f)
                    break  # break the while loop

    return lm


def generate_completely_random_sequences(N, L, outfile):
    with open(outfile, "w") as f:
        for _ in range(N):
            seq = "".join(np.random.choice(valid_residue_types, size=L))
            print(">", file=f)
            print(seq, file=f)


def generate_randomly_mutated_sequences(database_infile, N, outfile):
    canonical_amino_acid_order = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L",
                                  "K", "M", "F", "P", "S", "T", "W", "Y", "V", "-"]
    residue_distributions = defaultdict(
        lambda: {"A": 0, "R": 0, "N": 0, "D": 0, "C": 0, "Q": 0, "E": 0, "G": 0, "H": 0, "I": 0, "L": 0,
                 "K": 0, "M": 0, "F": 0, "P": 0, "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0, "-": 0})

    # read in and preprocess database file
    fasta_sequences = SeqIO.parse(database_infile, "fasta")
    list_of_sequences = [seq.seq for seq in fasta_sequences]
    fasta_sequences = SeqIO.parse(database_infile, "fasta")
    list_of_counts = [int(seq.id.split('|')[0]) for seq in fasta_sequences]
    total_counts = sum(list_of_counts)
    L=len(list_of_sequences[0])  # get the length of a sequence
    for residue_index in range(L):
        for seq_idx, seq in enumerate(list_of_sequences):
            amino_acid = seq[residue_index]
            residue_distributions[residue_index][amino_acid] += list_of_counts[seq_idx]

    normalised_residue_distribution = dict()
    for residue_index, residue_distribution in residue_distributions.items():
        normalised_residue_distribution[residue_index] = [count / total_counts for count in
                                                          residue_distribution.values()]

    generated_sequences = []
    for seq_number in tqdm(range(N)):
        num_mut = np.round(np.random.exponential(scale=8.810980))
        sequence = random.choice(list_of_sequences)
        if num_mut == 0:
            generated_sequences.append(sequence)
        else:
            num_mut_remaining = num_mut
            forbidden_indices = []
            while num_mut_remaining > 0:
                mut_index = np.random.randint(0, L)
                if mut_index not in forbidden_indices:
                    old_amino_acid = sequence[mut_index]
                    new_amino_acid = np.random.choice(canonical_amino_acid_order,
                                                      p=normalised_residue_distribution[mut_index])
                    if new_amino_acid != old_amino_acid:
                        num_mut_remaining -= 1
                        forbidden_indices.append(mut_index)
                        sequence = "".join([residue if residue_index != mut_index else new_amino_acid for residue_index, residue in enumerate(sequence)])

            generated_sequences.append(sequence)

    with open(outfile, "w") as f:
        for seq in generated_sequences:
            print(">", file=f)
            print(seq, file=f)


if __name__ == "__main__":
    from fasta_preprocessing_tools import reduce_to_unique_sequences

    # n_gram_list = [11, 13, 15, 17]
    # for n in n_gram_list:
    # lm = generate_random_ngram_sequences("../data/data_for_training_language_model.afa",
    #                                      # number of sequences to be generated (same as in experimental database)
    #                                      71397,
    #                                      # length of sequences to be generated (same as in experimental database)
    #                                      1299,
    #                                      # n gram number
    #                                      n=11,
    #                                      outfile=f'random_{11}gram_sequences_big')
    reduce_to_unique_sequences(infile=f"random_{11}gram_sequences_big",
                               outfile=f"random_{11}gram_sequences_big.unique",
                               data_directory='.')


    # generate_randomly_mutated_sequences("../data/spikeprot_final_dataset.afa", 71397, "generated_with_random_mutations")
    # reduce_to_unique_sequences(infile=f"generated_with_random_mutations",
    #                            outfile=f"generated_with_random_mutations.unique",
    #                            data_directory='.')
