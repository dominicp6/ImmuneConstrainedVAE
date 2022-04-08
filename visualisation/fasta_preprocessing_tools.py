"""
This script provides functions which preprocess fasta datafiles by removing incomplete sequences,
collecting repeats, labelling with variant names, and aligning misaligned sequences with MUSCLE.
"""
import random

import pandas as pd
from Bio import SeqIO, pairwise2
from Bio.Align.Applications import MuscleCommandline
from collections import defaultdict
import os
import numpy as np
from tqdm import tqdm
import itertools

script_dir = os.path.dirname(os.path.realpath(__file__))  # path to this file (DO NOT CHANGE)

path_to_muscle_executable = '/home/dominic/miniconda3/pkgs/muscle-3.8.1551-h7d875b9_6/bin/muscle'
path_to_consensus_sequences = os.path.join(script_dir, "data", "spike_protein_sequences", "consensus_sequences")

# relative path of datasets
data_dir = os.path.join(script_dir, "data", "spike_protein_sequences")


def _get_substring(string, mask):
    return "".join([char for index, char in enumerate(string) if mask[index] is False])


def rescale_counts_based_on_reference_label(database, outfile, reference_label, number_of_reference_sequences):
    db_seq = SeqIO.parse(database, 'fasta')

    counts = defaultdict(list)

    for seq in db_seq:
        count = int(seq.id.split('|')[0])
        label = '|'.join(seq.id.split('|')[1:])
        counts[label].append(count)

    total_counts = {label: sum(sorted(count_list, reverse=True)[0:number_of_reference_sequences])
                        for label, count_list in counts.items()}
    labels_of_reference = [label for label in counts.keys() if reference_label in label]
    reference_counts = sum([total_counts[label] for label in labels_of_reference])

    db_seq = SeqIO.parse(database, 'fasta')

    with open(outfile, "w") as out_file:
        for seq in db_seq:
            if reference_label in seq.id:
                print(f">{seq.id}", file=out_file)
                print(seq.seq, file=out_file)
            else:
                count = int(seq.id.split('|')[0])
                label = "|".join(seq.id.split('|')[1:])
                adjusted_count = int(count * reference_counts/total_counts[label])
                new_id = str(adjusted_count)+'|'+label
                print(f">{new_id}", file=out_file)
                print(seq.seq, file=out_file)


def remove_redundant_empty_residues(database, outfile):
    db_seq = SeqIO.parse(database, 'fasta')
    for id, seq in enumerate(db_seq):
        if id == 0:
            remove_by_residue_mask = [True]*len(seq.seq)
        for residue_num, residue in enumerate(seq.seq):
            if residue != "-" and remove_by_residue_mask[residue_num] == True:
                if id < 10000: # if it is a very rare change then ignore it as a fluke
                    remove_by_residue_mask[residue_num] = False

    print(remove_by_residue_mask)
    db_seq = SeqIO.parse(database, 'fasta')

    with open(outfile, "w") as out_file:
        for seq in db_seq:
            print(f">{seq.id}", file=out_file)
            print(_get_substring(seq.seq, remove_by_residue_mask), file=out_file)


def remove_empty_residues(database, outfile):
    db_seq = SeqIO.parse(database, 'fasta')

    with open(outfile, "w") as out_file:
        for seq in db_seq:
            print(f">{seq.id}", file=out_file)
            print(seq.seq.replace("-", ""), file=out_file)



def randomise_database(database, outfile):
    db_seq = SeqIO.parse(database, 'fasta')

    sequences = []
    for seq in db_seq:
        sequences.append((seq.id, seq.seq))

    random.shuffle(sequences)

    with open(outfile, "w") as out_file:
        for seq_tuple in sequences:
            print(f"f>{seq_tuple[0]}", file=out_file)
            print(seq_tuple[1], file=out_file)


def check_all_sequences_have_same_length(database):
    db_seq = SeqIO.parse(database, 'fasta')

    max_length = 0
    error_found = False
    for seq_id, seq in tqdm(enumerate(db_seq)):
        if max_length == 0:
            max_length = len(seq.seq)
        else:
            if len(seq.seq) != max_length:
                print(f"Error: The sequence number {seq_id+1} has length {len(seq.seq)} whereas the first sequence has length {max_length}.")
                error_found = True

    if not error_found:
        print('All sequences have the same length!')


def remove_id_label_from_fasta_database(database, id_position, outfile):
    db_seq = SeqIO.parse(database, 'fasta')

    with open(outfile, 'w') as out_file:
        for seq in tqdm(db_seq):
            seq_id_list = seq.id.split('|')
            try:
                del seq_id_list[id_position]
                print(f">{'|'.join(seq_id_list)}", file=out_file)
                print(seq.seq, file=out_file)
            except:
                print(f">{'|'.join(seq_id_list)}", file=out_file)
                print(seq.seq, file=out_file)


def database_similarity(reference_database, example_database, save_unique=False, outfile=None, compare_by_label=False):
    def strip_alignment(seq):
        return seq.replace('-', "")

    db_ref = SeqIO.parse(reference_database, 'fasta')
    uni_seq_ref = set(strip_alignment(seq.seq) for seq in db_ref)

    db_exa = SeqIO.parse(example_database, 'fasta')
    if compare_by_label is False:
        uni_seq_exa = set(strip_alignment(seq.seq) for seq in db_exa)
    else:
        label_to_seqs = defaultdict(list)
        for seq in db_exa:
            label_to_seqs["".join(seq.id.split('|')[1:])].append(seq.seq)
            print(seq.id)
        print(label_to_seqs.keys())
        uni_seq_exa_by_label = {label: set(strip_alignment(seq) for seq in seqs)
                                for label,seqs in label_to_seqs.items()}

    if compare_by_label is False:
        number_overlap = len(uni_seq_exa.intersection(uni_seq_ref))
        num_exa = len(uni_seq_exa)
        fraction_overlap = number_overlap/num_exa

        print(f'Of the {num_exa} unique sequences in {example_database}, {number_overlap} of them also appear in {reference_database}.')
        print(f'This corresponds to an overlap percentage of {round(fraction_overlap*100,1)}%.')
    else:
        for label, seq_set in uni_seq_exa_by_label.items():
            number_overlap = len(seq_set.intersection(uni_seq_ref))
            num_exa = len(seq_set)
            fraction_overlap = number_overlap / num_exa

            print(
                f'Of the {num_exa} unique {label} sequences in {example_database}, {number_overlap} of them also appear in {reference_database}.')
            print(f'This corresponds to an overlap percentage of {round(fraction_overlap * 100, 1)}%.')

    if save_unique is True:
        original_sequences = uni_seq_exa.difference(uni_seq_ref)
        db_exa = SeqIO.parse(example_database, 'fasta')
        with open(outfile, 'w') as out_file:
            for seq in db_exa:
                if strip_alignment(seq.seq) in original_sequences:
                    print(f'>{seq.id}', file=out_file)
                    print(f'{seq.seq}', file=out_file)


def add_id_label_to_fasta_database(database, id_position, label, outfile):
    db_seq = SeqIO.parse(database, 'fasta')

    with open(outfile, 'w') as out_file:
        for seq in tqdm(db_seq):
            seq_id_list = seq.id.split('|')
            seq_id_list.insert(id_position, label)
            print(f">{'|'.join(seq_id_list)}", file=out_file)
            print(seq.seq, file=out_file)


def partition_fasta_database_into_chunks(database, chunk_size, outfile_prefix):
    """
    Splits a single database into multiple smaller databases each containing chunk_size number of sequences
    (except possibly the last chuck, which may be smaller).
    """
    db_seq = SeqIO.parse(database, 'fasta')

    file_index = 0
    saw_a_new_sequence = True
    with tqdm(total=0) as pbar:
        while saw_a_new_sequence:
            for _ in (True,):  # "breakable scope" idiom
                with open(f"{outfile_prefix}_{file_index}.fasta", "w") as out_file:
                    saw_a_new_sequence = False
                    for seq_number, seq in enumerate(db_seq):
                        saw_a_new_sequence = True
                        if seq_number < chunk_size:
                            print(f">{seq.id}", file=out_file)
                            print(seq.seq, file=out_file)
                        else:
                            break
                    file_index += 1
                    pbar.update(1)
                    break


def combine_two_databases(database1, database2, variant_database2, outfile, variant_database1=None):
    """
    Merges two fasta databases, with optional labelling to distinguish sequences originating from the two databases.

    :param database1: Path to first database.
    :param database2: Path to second database.
    :param variant_database2: The "variant" label for the sequences in database 2 (e.g. 'natural', 'VAEsynthetic', ...)
    :param outfile: The name for the outfile of the merged databases.
    :param variant_database1: The "variant" label for the sequences in database 1. If "None", then leave the label
    unchanged.
    """
    db1_seq = SeqIO.parse(database1, 'fasta')
    db2_seq = SeqIO.parse(database2, 'fasta')

    with open(outfile, "w") as out_file:
        for seq in db1_seq:
            if variant_database1 is not None:
                print(f">{seq.id}|{variant_database1}", file=out_file)
            else:
                print(f">{seq.id}", file=out_file)
            print(seq.seq, file=out_file)
        for seq in db2_seq:
            print(f">{seq.id}|{variant_database2}", file=out_file)
            print(seq.seq, file=out_file)

def remove_all_sequences_with_label(infile, outfile, label):
    db = SeqIO.parse(infile, 'fasta')
    seq_2_label = dict()
    seq_2_identifier = dict()
    for seq in db:
        seq_2_label[seq.seq] = seq.id.split('|')[1]
        seq_2_identifier[seq.seq] = seq.id

    seq_2_label = dict(sorted(seq_2_label.items(), key=lambda item: item[1], reverse=True))
    with open(outfile, "w") as out_file:
        for id, seq in enumerate(seq_2_label.keys()):
            if seq_2_label[seq] != label:
                print(f">{seq_2_identifier[seq]}", file=out_file)
                print(seq, file=out_file)
            else:
                continue

def remove_all_sequences_without_label(infile, outfile):
    db = SeqIO.parse(infile, 'fasta')
    seq_2_identifier = dict()
    for seq in db:
        try:
            seq.id.split('|')[1]
            seq_2_identifier[seq.seq] = seq.id
        except:
            pass

    with open(outfile, "w") as out_file:
        for id, seq in enumerate(seq_2_identifier.keys()):
            print(f">{seq_2_identifier[seq]}", file=out_file)
            print(seq, file=out_file)


def keep_top_N_most_common_sequences(infile, outfile, N):
    db = SeqIO.parse(infile, 'fasta')
    seq_2_count = dict()
    seq_2_identifier = dict()
    for seq in db:
        seq_2_count[seq.seq] = int(seq.id.split('|')[0])
        seq_2_identifier[seq.seq] = seq.id

    seq_2_count = dict(sorted(seq_2_count.items(), key=lambda item: item[1], reverse=True))
    with open(outfile, "w") as out_file:
        for id, seq in enumerate(seq_2_count.keys()):
            if id < N:
                print(f">{seq_2_identifier[seq]}", file=out_file)
                print(seq, file=out_file)
            else:
                break

def get_sequence_count_info(infile):
    db = SeqIO.parse(infile, 'fasta')
    seq_2_count = dict()
    seq_2_identifier = dict()
    for seq in db:
        seq_2_count[seq.seq] = int(seq.id.split('|')[0])
        seq_2_identifier[seq.seq] = seq.id

    print(f'The database has {len(seq_2_count.keys())} entries with a combined sequence count of {sum(seq_2_count.values())}')


def remove_incomplete_sequences_from_fasta(infile,
                                           outfile,
                                           length_cutoff=1200,
                                           invalid_amino_acids_cutoff=1,
                                           data_directory=data_dir):
    """
    Given an input fasta file, creates a new fasta file with corrupt sequences removed.
    Corrupt sequences are those which are either too short, or contain too many 'X's, indicating low data quality.

    :param infile: The file to be read.
    :param outfile:  The file to be created.
    :param length_cutoff:  Sequences below this length are removed.
    :param invalid_amino_acids_cutoff:  Sequences with this many or more 'X' amino acids are removed.
    :param data_directory: Path to directory of infile and outfile.
    """

    number_of_sequences = 0
    number_of_complete_sequences = 0
    with open(os.path.join(data_directory, outfile), "w") as outfile:
        with open(os.path.join(data_directory, infile), "r") as infile:
            for sequence in tqdm(infile):
                number_of_sequences += 1
                # if the line is not a descriptor line
                if sequence[0] != '>':
                    # if sequence is not corrupt
                    if len(sequence) > length_cutoff and sequence.count('X') < invalid_amino_acids_cutoff:
                        number_of_complete_sequences += 1
                        outfile.write(descriptor_line)
                        outfile.write(sequence)
                    else:
                        pass
                # temporarily store the descriptor
                else:
                    descriptor_line = sequence

    print(f'The database {infile} consists of {number_of_sequences} entries of which {number_of_complete_sequences}'
          f'are complete.')


def downsample_fasta_file(infile,
                          outfile,
                          downsample_factor=100,
                          data_directory=data_dir):
    """
    Downsamples a fasta file by extracting only every Nth sequence.

    :param infile: The fasta file to downsample.
    :param outfile: The name for the downsampled fasta file to be saved to disk.
    :param downsample_factor: The factor by which to downsample.
    :param data_directory: Path to directory of infile and outfile.
    """

    with open(os.path.join(data_directory, infile), "r") as original_fasta_file:
        with open(os.path.join(data_directory, outfile), "w") as downsampled_fasta_file:
            for line_index, line in tqdm(enumerate(original_fasta_file)):
                # factor of 2 because each sequence also has an id row in a fasta file
                print_every = 2 * downsample_factor
                if line_index % print_every == 0 or (line_index - 1) % print_every == 0:
                    print(line, file=downsampled_fasta_file)


def reduce_to_unique_sequences(infile,
                               outfile,
                               data_directory=data_dir):
    """
    Reads a fasta file, identifies the unique sequences and outputs these to new fasta file.
    Also constructs dictionaries of the counts of each unique sequence (sequence_count_data)
    as well as the of number of sequences of each length (sequence_length_dict).
    The new fasta file contains in its description a count of how common the sequence was.


    :param infile: The input fasta file to be processed.
    :param outfile: The name of the outputfile.
    :param data_directory: Path to directory of infile and outfile.
    :return: Dictionaries counting the number of each type of sequence and the number of
             of each length of sequence.
             e.g. ['ABC' : 1000, 'BCDE' : 500], [3: 1000, 4: 500]
    """
    fasta_sequences = SeqIO.parse(os.path.join(data_directory, infile), 'fasta')

    # Check is valid fasta file
    if not fasta_sequences:
        raise ValueError(f'{infile} is not a valid .fasta file')

    sequence_count_dict = defaultdict(lambda: 0)  # dict(sequence (str): counts of sequence (int))
    sequence_length_dict = defaultdict(lambda: 0)  # dict(length_of_sqn (int): counts with this length (int))
    sequence_date_dict = defaultdict(lambda: [])  # dict(sequence (str): median date that sequence was recorded)
    number_of_sequences = 0

    for seq_obj in tqdm(fasta_sequences):
        identifier, sequence = seq_obj.id, str(seq_obj.seq)

        # remove non-sequence character suffixes if they exist
        if not sequence[-1].isalpha() and sequence[-1] != '-':
            sequence = sequence[:-1]

        try:
            date_string = identifier.split('|')[2]
            date = np.datetime64(date_string)
            sequence_date_dict[sequence].append(date)
        except:
            pass

        sequence_count_dict[sequence] += 1
        sequence_length_dict[len(sequence)] += 1
        number_of_sequences += 1

    # sort sequences in decreasing order of frequency of occurrence
    sequence_count_dict = dict(sorted(sequence_count_dict.items(), key=lambda item: item[1], reverse=True))

    # compute median date of each sequence
    print('Computing median date for each sequence...')
    for sequence, date_list in tqdm(sequence_date_dict.items()):
        median_date = pd.Series(date_list, dtype='datetime64[ns]').quantile(0.5, interpolation="midpoint")
        sequence_date_dict[sequence] = median_date

    with open(os.path.join(data_directory, outfile), "w") as f:
        for seq, count in sequence_count_dict.items():
            seq_date = sequence_date_dict[seq]
            if seq_date is not None:
                print(f'>{count}|{seq_date}', file=f)
            else:
                print(f'>{count}', file=f)
            print(seq, file=f)

    print(f'Processed {infile}:')
    print(f'Found {number_of_sequences} sequences,')
    print(f'of which {len(sequence_count_dict)} are unique sequences.')

    return sequence_count_dict, sequence_length_dict


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


def reduce_and_align_sequences(infile: str,
                               outfile: str,
                               reduction_factor: int,
                               length_cutoff=1200,
                               invalid_amino_acids_cutoff=1,
                               data_directory=data_dir,
                               path_to_muscle_executable=path_to_muscle_executable):
    """
    Removes incomplete sequences from a fasta database, then downsamples, pools identical sequences, labels them with
    the closest matching covid spike protein variant and finally aligns them using MUSCLE.

    :param infile: The original fasta file to reduce and align.
    :param outfile: The name of the final fasta file.
    :param reduction_factor: Factor by which to downsample.
    :param length_cutoff: Sequences below this length are removed.
    :param invalid_amino_acids_cutoff: Sequences with at least this many invalid amino acids are removed.
    """
    # 1.
    print('Removing incomplete sequences...')
    remove_incomplete_sequences_from_fasta(infile=infile,
                                           outfile=infile + '.cleaned',
                                           length_cutoff=length_cutoff,
                                           invalid_amino_acids_cutoff=invalid_amino_acids_cutoff,
                                           data_directory=data_dir)

    # 2.
    print(f'Downsampling sequences by factor of {reduction_factor}...')
    downsample_fasta_file(infile=infile + '.cleaned',
                          outfile=infile + '.cleaned.downsampled',
                          downsample_factor=reduction_factor,
                          data_directory=data_dir)

    # 3.
    print('Getting unique sequences...')
    reduce_to_unique_sequences(infile=infile + '.cleaned.downsampled',
                               outfile=infile + '.cleaned.downsampled.unique',
                               data_directory=data_dir)

    # 4.
    print('Labelling with closest variant...')
    label_fasta_file_sequences_with_closest_variant(infile=infile + '.cleaned.downsampled.unique',
                                                    outfile=infile + '.cleaned.downsampled.unique.labeled',
                                                    data_directory=data_dir,
                                                    path_to_consensus_sequences=path_to_consensus_sequences)

    # 5.
    print('To complete the alignment step run this command in the terminal:')
    muscle_command = MuscleCommandline(path_to_muscle_executable,
                                       input=data_directory + infile + '.cleaned.downsampled.unique.labeled',
                                       out=data_directory + outfile)
    print(muscle_command)


if __name__ == "__main__":
    #database_similarity('./data/spike_protein_sequences/spikeprot_bigger_dataset.afa', './data/spike_protein_sequences/1_in_50_cleaned.fasta', save_unique=False, outfile=None)
    # keep_top_N_most_common_sequences('./data/spike_protein_sequences/half_of_training_data_trimmed.afa',
    #                                  './data/most_common_natural_4k.fasta', 4000)
    # remove_all_sequences_without_label('./data/spike_protein_sequences/half_of_training_data.afa',
    #                                    './data/spike_protein_sequences/half_of_training_data_trimmed.afa')
    # combine_two_databases('./data/generated_sequences/LanguageModel/random_11gram_top_2730',
    #                       variant_database1='language model',
    #                       database2='./data/generated_sequences/Random75/random_mutator_top_2730',
    #                       variant_database2='random mutator',
    #                       outfile='./data/generated_sequences/language_model_and_random_mutator_big.fasta')
    # combine_two_databases('./data/generated_sequences/language_model_and_random_mutator_big.fasta',
    #                       database2='./data/generated_sequences/VAE/Version 4/gen4_merged_conserved_regions.afa',
    #                       variant_database2='VAE model',
    #                       outfile='./data/generated_sequences/all_generated_sequences_merged_big.fasta')
    #check_all_sequences_have_same_length('./data/generated_sequences/all_generated_sequences_merged.fasta')
    # get_sequence_count_info('./data/generated_sequences/Random75/random75_original.fasta')
    # get_sequence_count_info('./data/generated_sequences/LanguageModel/11gram_original.fasta')
    # get_sequence_count_info('./data/generated_sequences/VAE/Version 4/gen4_merged_conserved_regions.afa')
    # print(MuscleCommandline(path_to_muscle_executable,
    #                   input='./gen_all_big_0,5.fasta',
    #                   out='./gen_all_big_0.afa'))
    # partition_fasta_database_into_chunks('./gen_all_big_2.afa',
    #                                      1000,
    #                                      'gen_all_big_2')
    # remove_redundant_empty_residues('./data/most_common_natural.fasta', './data/most_common_natural.fasta')
    # remove_redundant_empty_residues('./data/most_common_natural_4k.fasta', './data/most_common_natural_4k.fasta')
    # remove_empty_residues('./data/gen_and_natural_visualisation_1.fasta', './data/gen_and_natural_visualisation_1.fasta')
    # randomise_database('./data/gen_and_natural_visualisation_1.fasta', './data/gen_and_natural_visualisation_1_.fasta')
    #partition_fasta_database_into_chunks('./data/gen_and_natural_visualisation_1_.fasta', 1000, './data/gen_and_natural_1_chunk_')
    database_similarity('./data/spikeprot_final_dataset.afa',
                        './data/gen_and_nat_v_1_with_conserved_regions.fasta',
                        compare_by_label=True)


