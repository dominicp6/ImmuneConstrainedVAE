from Bio import SeqIO, pairwise2
from fasta_preprocessing_tools import combine_two_databases


def load_database(fasta_database):
    db = SeqIO.parse(fasta_database, 'fasta')

    return db


def get_conserved_regions_mask(database):

    index = 0
    for seq in database:
        try:
            variant_name = seq.id.split('|')[1]
        except:
            variant_name = 'none given'
        if variant_name == 'None':
            if index == 0:
                conserved_region_mask = [True]*len(seq.seq)
                reference_sequence = seq.seq

            else:
                for residue_index, residue in enumerate(seq.seq):
                    if residue == reference_sequence[residue_index]:
                        pass
                    else:
                        conserved_region_mask[residue_index] = False
            index += 1

    return reference_sequence, conserved_region_mask


def filter_by_conserved_regions(database, reference_sequence, conserved_region_mask):
    filtered_sequences = []
    for seq in database:
        try:
            variant_name = seq.id.split('|')[1]
        except:
            variant_name = 'none given'
        if variant_name != "natural":
            for residue_index, residue in enumerate(seq.seq):
                if conserved_region_mask[residue_index] is True:
                    if reference_sequence[residue_index] == residue:
                        pass
                    else:
                        break
                if residue_index == len(seq.seq) - 1:
                    filtered_sequences.append(seq)

    return filtered_sequences

def save_sequence_list(sequence_list, filename):
    with open(filename, 'w') as file:
        for seq in sequence_list:
            print(f">{seq.id}", file=file)
            print(seq.seq, file=file)



def save_sequences_that_obey_conserved_regions(input, output):
    # Load sequences
    sequences_database = load_database(input)

    # conserved regions
    ref_seq, conserved_region_mask = get_conserved_regions_mask(sequences_database)

    # filter by conserved regions
    sequences_database = load_database(input)
    filtered_seqs = filter_by_conserved_regions(sequences_database, ref_seq, conserved_region_mask)

    # save filtered sequences to file
    save_sequence_list(filtered_seqs, output)





if __name__ == "__main__":
    save_sequences_that_obey_conserved_regions('../data/spike_protein_sequences/gen_and_natural_visualisation_1.afa',
                                               "../data/gen_and_nat_v_1_with_conserved_regions.fasta")