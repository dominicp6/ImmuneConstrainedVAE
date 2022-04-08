import numpy as np
import os
from Bio import SeqIO
from DSSPparser import parseDSSP
import pandas as pd
from collections import defaultdict
from Bio import pairwise2


def get_fasta_sequence_from_file(fasta_file):
    """
    Reads a fasta file containing a single sequence and returns the sequence string.
    """
    number_of_sequences_found = 0
    sequence = None
    for sequence_entry in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(sequence_entry.seq)
        number_of_sequences_found += 1

    if number_of_sequences_found != 1:
        raise ValueError('fasta file should have only a single sequence')

    return sequence


def extract_solvent_accessibility_from_dssp_file(dssp_file):
    """
    Parses a dssp file to find the solvent accessibility of each residue.
    """
    dssp_parser = parseDSSP(dssp_file)
    dssp_parser.parse()
    dssp_parsed_dict = dssp_parser.dictTodataframe()
    dssp_parsed_dict = dssp_parsed_dict[['resnum', 'inscode', 'chain', 'aa', 'acc']]

    solvent_accessibility_of_residue = defaultdict(lambda: [])
    residue_number_to_amino_acid = defaultdict(lambda: '')
    for entry in dssp_parsed_dict.iterrows():
        amino_acid = entry[1].aa  # position [0] is ID, position [1] is row data

        # the ! character is used to mark sequence breaks in dssp files; we ignore it
        if amino_acid == "!":
            continue

        # In dssp files CYS amino acids are labelled with lower case letters
        if amino_acid.islower():
            # relabel lower case letter with one-letter code for CYS
            amino_acid = 'C'

        amino_acid_residue_number = int(entry[1].inscode)

        residue_number_to_amino_acid[amino_acid_residue_number] = amino_acid
        solvent_accessibility = entry[1].acc
        solvent_accessibility_of_residue[amino_acid_residue_number].append(int(solvent_accessibility))

    averaged_solvent_accessibility_of_residue = dict()
    for residue_number, solvent_accessibilities in solvent_accessibility_of_residue.items():
        averaged_solvent_accessibility_of_residue[residue_number] = np.median(solvent_accessibilities)

    dssp_dictionary = {'residue_number': list(averaged_solvent_accessibility_of_residue.keys()),
                       'solvent_accessibility': list(averaged_solvent_accessibility_of_residue.values()),
                       'amino_acid': [residue_number_to_amino_acid[residue]
                                      for residue in averaged_solvent_accessibility_of_residue.keys()]}

    return pd.DataFrame(dssp_dictionary)


def pandas_series_to_string(series):
    """
    Concatenates the entries of a pandas series into a string.
    """
    string = ""
    for entry in series:
        string += str(entry)

    return string


def align_sequences(sequence1, sequence2):
    alignments = pairwise2.align.globalxx(sequence1, sequence2)

    best_alignment = alignments[0]

    sequence1_aligned = best_alignment.seqA
    sequence2_aligned = best_alignment.seqB

    return sequence1_aligned, sequence2_aligned


def adjust_epitope_mask_to_aligned_database(reference_sequence, epitope_mask, aligned_fasta_database_file):
    """
    Imputes missing entries into an epitope mask if "-"s appear in the aligned reference sequence in
    aligned_fasta_database_file.
    """
    for sequence_entry in SeqIO.parse(aligned_fasta_database_file, "fasta"):
        sequence = str(sequence_entry.seq)
        stripped_sequence = sequence.replace('-', '')

        # if the sequence found is a subsequence of the reference then it is a match
        if stripped_sequence in reference_sequence:
            refined_epitope_mask = np.zeros(len(sequence))
            indices_of_hyphens = np.array([pos for pos, char in enumerate(sequence) if char == '-'])

            # extend length of epitope mask by number of hyphens found
            for index, _ in enumerate(sequence):
                number_of_hyphens_passed = len(indices_of_hyphens[indices_of_hyphens < index])
                if index in indices_of_hyphens:
                    # impute missing entries with the average values of adjacent entries
                    refined_epitope_mask[index] = (epitope_mask[index - number_of_hyphens_passed - 1] +
                                                   epitope_mask[index - number_of_hyphens_passed]) / 2
                else:
                    refined_epitope_mask[index] = epitope_mask[index - number_of_hyphens_passed]
            break

    try:
        return refined_epitope_mask
    except:
        raise Exception('The reference sequence does not appear in the provided fasta file.')


def get_epitope_mask(fasta_file,
                     dssp_file,
                     data_directory,
                     aligned_file,
                     normalise=True,
                     threshold=None):
    """
    Pairwise sequence aligns a reference fasta sequence to a dssp sequence and returns
    a solvent accessibility vector for the fasta sequence.

    :param fasta_file: The fasta file of the reference sequence.
    :param dssp_file:  The dssp file of the reference sequence (generated from a PDB file)
    :param data_directory: The relative path to the directory containing the fasta file and dssp file.
    :param aligned_file: The fasta database of aligned sequences which the mask will be applied to. Needed to make sure
                         that the size of the epitope mask vector is expanded to the length of the sequence-aligned
                         reference sequence.
    :param normalise: If true then normalises the epitope mask relative to the size of its maximum entry.
    :param threshold: [Optional] A float between 0 and 1. Values below this relative size are set to zero,
                       values above are set to 1. I.e. this makes the epitope mask a binary mask.
    :return: epitope mask
    """
    fasta = os.path.join(data_directory, fasta_file)
    dssp = os.path.join(data_directory, dssp_file)
    reference_sequence = get_fasta_sequence_from_file(fasta)
    dssp_solvent_accessibility = extract_solvent_accessibility_from_dssp_file(dssp)
    dssp_sequence = pandas_series_to_string(dssp_solvent_accessibility.amino_acid)

    aligned_reference_sequence, aligned_dssp_sequence = align_sequences(reference_sequence, dssp_sequence)

    position_in_dssp_sequence = 0
    epitope_mask = []

    # converts a dssp solvent accessibility vector into a vector defined relative to the original fasta sequence
    for index, dssp_amino_acid in enumerate(aligned_dssp_sequence):
        if aligned_reference_sequence[index] == "-":
            continue
        else:
            if dssp_amino_acid != "-":
                solvent_accessibility_of_amino_acid = dssp_solvent_accessibility.solvent_accessibility[
                    position_in_dssp_sequence]
                epitope_mask.append(solvent_accessibility_of_amino_acid)
                position_in_dssp_sequence += 1
            else:
                epitope_mask.append(0)

    # apply optional post-processing to the epitope mask
    epitope_mask = np.array(epitope_mask)
    if threshold is not None:
        assert 1 > threshold > 0, "threshold must be a float between 0 and 1"
        max_sa = np.max(epitope_mask)
        epitope_mask = [0 if entry / max_sa < threshold else 1
                        for entry in epitope_mask]
    if normalise:
        epitope_mask /= np.max(epitope_mask)

    # adjust epitope mask so that it aligns to a given database
    epitope_mask = adjust_epitope_mask_to_aligned_database(reference_sequence, epitope_mask, aligned_file)

    return epitope_mask, reference_sequence


if __name__ == "__main__":
    data_dir = os.path.join("..", "data", "spike_protein_pdb")
    aligned_file = os.path.join("..", "data", "spike_protein_sequences", "1_in_500_cleaned_aligned.afa")

    epitope_mask, reference_sequence = get_epitope_mask(fasta_file='reference_spike.fasta',
                                                        dssp_file='reference_spike.dssp',
                                                        aligned_file=aligned_file,
                                                        data_directory=data_dir)
