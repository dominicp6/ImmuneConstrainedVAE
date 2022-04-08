"""
Tools to label SARS-COV2 spike sequences with the closest known variant.
"""
from Bio import pairwise2
import os
from Bio import SeqIO


def create_variant_sequences_dict(sequences_src):
    """
    Creates a dictionary withs keys the variant names and values the reference spike protein sequences.
    """
    names = []
    for entry in os.listdir(sequences_src):  # Read all sequences
        if os.path.isfile(os.path.join(sequences_src, entry)):
            names.append(entry)

    consensus_dict = {}
    for name in names:
        if name[:-6] not in consensus_dict.keys():
            for fasta in SeqIO.parse(os.path.join(sequences_src, name), "fasta"):
                fasta_name, sequence = fasta.id, str(fasta.seq)
            consensus_dict[name[:-6]] = sequence

    return consensus_dict


def label_fasta_file_sequences_with_closest_variant(infile, outfile, path_to_consensus_sequences, data_directory):
    """
    Reads an unlabeled fasta file, finds the closest known variant to each sequence and
    generates an output file with each sequence labelled.
    """

    variant_sequences_dict = create_variant_sequences_dict(path_to_consensus_sequences)

    variant_similarity = {}
    unlabeled_fasta_file = open(os.path.join(data_directory, infile), "r")

    with open(os.path.join(data_directory, outfile), 'w') as labeled_fasta_file:
        while True:
            frequency = unlabeled_fasta_file.readline()
            sequence = unlabeled_fasta_file.readline()
            if not sequence: break  # EOF

            for variant, ref_seq in variant_sequences_dict.items():
                alignment_score = pairwise2.align.globalxx(ref_seq, sequence, score_only=True)
                variant_similarity[variant] = alignment_score

            line1 = frequency.strip() + "|" + max(variant_similarity, key=variant_similarity.get) + "\n"
            line2 = sequence
            labeled_fasta_file.writelines([line1, line2])


if __name__ == "__main__":
    cons_sequences_src = os.path.join("..", "data", "spike_protein_sequences", "consensus_sequences")
    data_dir = os.path.join("..", "data", "spike_protein_sequences")

    label_fasta_file_sequences_with_closest_variant(infile="spikeprot0112.fasta.cleaned.downsampled.unique",
                                                    outfile="spikeprot0112.fasta.cleaned.downsampled.unique.labeled2",
                                                    path_to_consensus_sequences=cons_sequences_src,
                                                    data_directory=data_dir)
