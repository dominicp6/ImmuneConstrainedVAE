"""
A script to visualise how much each residue of the Spike protein is conserved across the population.

To compute spike protein variation at each residue one method is to follow Sikora et al
(except I think that we can improve on it! They fixed their spike proteins to be of a given length!):

Define a conservation score to be 1 minus the entropy of the observed amino acid distribution to the maximum possible
entropy at a given position: Score(i) = 1-\left(\frac{-\sum_k p_k(i)\log p_k(i)}{\log(21)}\right)

Note log(21) not log(20) because we include the empty code "-" as a possibility since we are dealing with aligned
sequences.
"""

from Bio import SeqIO
from collections import defaultdict
import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.realpath(__file__))  # path to this file (DO NOT CHANGE)

residue_distribution = defaultdict(lambda: defaultdict(lambda: {"A": 0, "R": 0, "N": 0, "D": 0, "C": 0, "Q": 0, "E": 0, "G": 0, "H": 0, "I": 0, "L": 0,
                         "K": 0, "M": 0, "F": 0, "P": 0, "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0, "-": 0}))

def calculate_entropy_of_count_distribution(count_distribution):
    normalised_distribution = count_distribution/sum(count_distribution)
    distribution_entropy = sum([probability * np.log(probability)
                                for probability in normalised_distribution if probability > 0])

    return distribution_entropy


def calculate_conservation_score(entropy, maximum_entropy):
    return 1 - entropy/maximum_entropy


def calculate_entropy_score(entropy, maximum_entropy):
    return entropy/maximum_entropy


# Function to calculate Chi-distance
def chi2_distance(A, B):
    # compute the chi-squared distance using above formula
    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b)
                        for (a, b) in zip(A, B)])

    return chi

def RMSD_distance(A, B):
    return np.sqrt(np.mean([(a-b)**2 for (a,b) in zip(A,B)]))

if __name__ == "__main__":
    # relative path of datasets
    data_dir = script_dir + '/data/spike_protein_sequences/'

    fasta_file = "../data/spikeprot_final_dataset.afa"
    fasta_sequences = SeqIO.parse(open(fasta_file), 'fasta')

    seq_length = 0
    for fasta in fasta_sequences:
        id_label, sequence = fasta.id, str(fasta.seq)
        seq_length = max(seq_length, len(sequence))
        sequence_count = int(id_label.split('|')[0])
        try:
            variant_name = id_label.split('|')[2]
        except:
            variant_name = 'default'
        for position, letter in enumerate(sequence):
            residue_distribution[variant_name][position][letter] += sequence_count

    count = 0
    variant_conservation_vectors = dict()
    for variant_name, variant_residue_distribution in residue_distribution.items():
        conservation_vector = np.zeros(seq_length)
        for position, count_dictionary in variant_residue_distribution.items():
            entropy_of_position = calculate_entropy_of_count_distribution(np.fromiter(count_dictionary.values(), dtype=np.int64))
            position_conservation_score = calculate_entropy_score(entropy_of_position, maximum_entropy=-np.log(21))
            if position_conservation_score <= 0.00000000000000000001:
                count += 1
                position_conservation_score = 0
            conservation_vector[position] = position_conservation_score
        variant_conservation_vectors[variant_name] = conservation_vector

    for variant_name, conservation_vector in variant_conservation_vectors.items():
        print(len(conservation_vector))
        if variant_name == "natural":
        # if variant_name != "synthetic":
            plt.plot(conservation_vector, linewidth=0.75, label=variant_name, c='k')

    print(count)
    #plt.legend()
    plt.xlabel('Residue Number', fontsize=20)
    #plt.ylabel('Normalised Positional Entropy', fontsize=15)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.margins(x=0, y=0)
    plt.ylim((0, 0.35))
    fig = plt.gcf()
    fig.set_size_inches(8.4, 3.5)
    fig.set_dpi(100)
    plt.savefig('natural_positional_entropy.pdf')

    natural_conservation_score = variant_conservation_vectors['natural']
    for variant_name, conservation_vector in variant_conservation_vectors.items():
        print(f'natural-{variant_name} RMSD distance: {RMSD_distance(natural_conservation_score, conservation_vector)}')





