from tqdm import tqdm
from Bio import pairwise2, SeqIO



def find_closest_natural_sequence(synthetic_sequence, natural_sequences_database):
    maximum_alignment_score = float('-inf')
    aligned_natural_seq = ''
    aligned_synthetic_seq = ''
    for natural_seq in tqdm(natural_sequences_database):
        alignment_score = pairwise2.align.globalxs(synthetic_sequence,
                                                   natural_seq.seq,
                                                   score_only=True,
                                                   open=-1,
                                                   extend=-0.1)

        if alignment_score > maximum_alignment_score:
            alignment = pairwise2.align.globalxs(synthetic_sequence,
                                                 natural_seq.seq,
                                                 one_alignment_only=True,
                                                 open=-1,
                                                 extend=-0.1)
            maximum_alignment_score = alignment[0].score
            aligned_synthetic_seq = alignment[0].seqA
            aligned_natural_seq = alignment[0].seqB
        else:
            pass

    return aligned_synthetic_seq, aligned_natural_seq, maximum_alignment_score



def generate_point_mutation_list(original_sequence, mutated_sequence):
    point_mutation_list = ""
    for res_index, residue_tuple in enumerate(zip(original_sequence, mutated_sequence)):
        orig_res = residue_tuple[0]
        mut_res = residue_tuple[1]
        if mut_res == "-" or orig_res == "-":
            # potentially an insertion or deletion -ignore
            continue
        elif mut_res != orig_res:
            # a mutation - add it to the list
            point_mutation_list+=f"{orig_res}{res_index+1}{mut_res}\n"
        else:
            # no mutation - ignore
            continue

    return point_mutation_list


def generate_point_mutation_lists_for_entire_database(database, exp_name):
    synthetic_db = SeqIO.parse(database, 'fasta')
    seq_to_count_dict = dict()
    seq_to_id_dict = dict()

    for seq in synthetic_db:
        variant = seq.id
        count = int(seq.id.split('|')[0])
        seq_to_count_dict[seq.seq] = count
        seq_to_id_dict[seq.seq] = variant

    sequences_in_descending_count_order = \
        [seq for seq, count in sorted(seq_to_count_dict.items(), key=lambda item: item[1], reverse=True)]

    for idx, seq in enumerate(sequences_in_descending_count_order):
        print(seq)
        print(seq_to_count_dict[seq])
        natural_db = SeqIO.parse('../data/spike_protein_sequences/1_in_500_cleaned_aligned.afa', 'fasta')
        aligned_synthetic, aligned_natural, minimum_alignment_score = find_closest_natural_sequence(seq, natural_db)

        print(f'Found a closely aligned sequence with alignment score {minimum_alignment_score}.')
        print(f'nat: {aligned_natural}')
        print(f'syn: {aligned_synthetic}')

        with open(f'./{exp_name}_point_mutations/{idx}.muts', 'w') as f:
            print(generate_point_mutation_list(original_sequence=aligned_natural, mutated_sequence=aligned_synthetic),
                  file=f)

        aligned_natural = aligned_natural.replace('-','X')  # replace dashes with Xs for natural sequences so DDGun can read them properly
        with open(f'./{exp_name}_point_mutations/{idx}_reference.fasta', 'w') as f:
            print(f'>Ref for seq {idx}', file=f)
            print(aligned_natural, file=f)
        with open(f'./{exp_name}_point_mutations/{idx}_synthetic.fasta', 'w') as f:
            print(f'>Seq {idx} | {seq_to_id_dict[seq]}', file=f)
            print(aligned_synthetic, file=f)




