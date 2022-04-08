# Adapted from https://github.com/xyjing-works/SequenceEncoding/blob/master/SequenceEncoding.py

import json
import numpy as np
import os.path
from Bio import SeqIO


class ProteinSequenceEncoder:
    valid_encoding_types = ['One_hot', 'One_hot_6_bit', 'Binary_5_bit', 'Hydrophobicity_matrix',
                            'Meiler_parameters', 'Acthely_factors', 'PAM250', 'BLOSUM62', 'Miyazawa_energies',
                            'Micheletti_potentials', 'AESNN3', 'ANN4D', 'ProtVec']

    valid_residue_types = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q',
                           'R', 'S', 'T', 'V', 'W', 'Y', 'X']

    def __init__(self,
                 encoding_type="One_hot",
                 mask=None,
                 encodings_directory=os.path.join("..", "data", "encodings"),
                 data_directory=os.path.join("..", "data", "spike_protein_sequences")):

        if encoding_type not in ProteinSequenceEncoder.valid_encoding_types:
            raise Exception(f"Encoding type {encoding_type} not found")

        self.encoding_type = encoding_type
        self.encodings_directory = encodings_directory
        self.data_directory = data_directory
        self.mask = mask

        self.encoding_dimension = None    # calculated automatically during encoding

    @staticmethod
    def read_encoding_file(path_to_sequence_encodings_file):
        """
        Helper function to parse a file of encoded sequences. Returns the sequence descriptors and the encodings
        found in the file.
        """
        descriptors = []
        encoded_sequences = None
        with open(path_to_sequence_encodings_file, "r") as seq_encodings_file:
            for line in seq_encodings_file:
                if line[0] == ">":
                    descriptors.append(int(''.join(filter(str.isdigit, line))))
                else:
                    if encoded_sequences is not None:
                        encoded_sequences = np.vstack(
                            (encoded_sequences, np.array([float(digit) for digit in line.split(',')])))
                    else:
                        encoded_sequences = np.array([float(digit) for digit in line.split(',')])

        return descriptors, encoded_sequences

    def get_ProtVec_encoding(self, ProtVec, seq, overlap=True):
        if overlap:
            encodings = []
            for i in range(len(seq) - 2):
                encodings.append({seq[i:i + 3]: ProtVec[seq[i:i + 3]]}) if ProtVec.__contains__(
                    seq[i:i + 3]) else encodings.append({seq[i:i + 3]: ProtVec["<unk>"]})
        else:
            encodings_1, encodings_2, encodings_3 = [], [], []
            for i in range(0, len(seq), 3):
                if i + 3 <= len(seq):
                    encodings_1.append({seq[i:i + 3]: ProtVec[seq[i:i + 3]]}) if ProtVec.__contains__(
                        seq[i:i + 3]) else encodings_1.append({seq[i:i + 3]: ProtVec["<unk>"]})
                if i + 4 <= len(seq):
                    encodings_2.append({seq[i + 1:i + 4]: ProtVec[seq[i + 1:i + 4]]}) if ProtVec.__contains__(
                        seq[i + 1:i + 4]) else encodings_2.append({seq[i + 1:i + 4]: ProtVec["<unk>"]})
                if i + 5 <= len(seq):
                    encodings_3.append({seq[i + 2:i + 5]: ProtVec[seq[i + 2:i + 5]]}) if ProtVec.__contains__(
                        seq[i + 2:i + 5]) else encodings_3.append({seq[i + 2:i + 5]: ProtVec["<unk>"]})

            encodings = [encodings_1, encodings_2, encodings_3]
        return encodings

    def get_encoding(self, seq: str, overlap=True, zero_void_residues=True):
        """
        Returns residue-encoding pairs of a sequence.

        e.g. seq = ['ACD'] and self.encoding_type = 'Binary_5_bit'
        returns [{'A': [1, 0, 0, 0, 1]}, {'C': [1, 0, 0, 1, 0]}, {'D': [1, 0, 1, 0, 0]}]
        """
        seq = seq.upper()
        with open(os.path.join(self.encodings_directory, f"{self.encoding_type}.json"), 'r') as load_f:
            encoding = json.load(load_f)
            self.encoding_dimension = encoding['dimension']
        encoding_data = []
        if self.encoding_type == "ProtVec":
            encoding_data = self.get_ProtVec_encoding(encoding, seq, overlap)
        else:
            for res in seq:
                # decides how to deal with void residues that arise from sequence alignment
                if res == "-":
                    if zero_void_residues:
                        encoding_data.append({res: [0] * self.encoding_dimension})
                        continue
                    else:
                        encoding_data.append({'X': encoding['X']})
                        continue
                if res not in ProteinSequenceEncoder.valid_residue_types:
                    res = "X"
                encoding_data.append({res: encoding[res]})

        return encoding_data

    def encode_single_sequence(self, seq: str):
        """
        Encodes a protein sequence string as a numeric vector.
        """
        if self.mask is None:
            encoded_seq = [digit for residue_encoding_dict in self.get_encoding(seq)
                           for amino_acid, encoding_vector in residue_encoding_dict.items()
                           for digit in encoding_vector]
        else:
            assert len(self.mask) >= len(seq), f"The mask (len {len(self.mask)}) must be at least " \
                                               f"as long as the sequence (len {len(seq)})."
            encoded_seq = []
            for index, residue_encoding_dict in enumerate(self.get_encoding(seq)):
                (amino_acid, encoding_vector), = residue_encoding_dict.items()
                encoding_vector = np.array(encoding_vector) * self.mask[index]
                encoded_seq.extend(encoding_vector)

        return encoded_seq

    @staticmethod
    def _convert_numeric_encoding_to_string_encoding(encoding):
        """
        e.g. [1, 2, 3] becomes ['1', '2', '3']
        """
        return ",".join([str(digit) for digit in encoding])

    def encode_from_fasta_file(self, fasta_file, outfilename):
        """
        Reads a fasta file of protein sequences and encodes each sequence.
        """
        fasta_sequences = SeqIO.parse(open(os.path.join(self.data_directory, fasta_file)), 'fasta')
        number_of_sequences = len(list(fasta_sequences))
        encoded_sequences = None
        descriptors = []

        fasta_sequences = SeqIO.parse(open(os.path.join(self.data_directory, fasta_file)), 'fasta')

        with open(os.path.join(self.data_directory, outfilename), "w") as out_file:
            for index, fasta in enumerate(fasta_sequences):
                identifier, sequence = fasta.description, str(fasta.seq)
                encoded_seq = self.encode_single_sequence(sequence)
                if index == 0:
                    encoded_sequences = np.zeros([number_of_sequences, len(encoded_seq)])
                try:
                    encoded_sequences[index, :] = encoded_seq
                except:
                    raise Exception(f'Not all sequences in {fasta_file} are the same length. In particular '
                                    f'the first sequence was length {encoded_sequences.shape[1]/self.encoding_dimension} '
                                    f'and sequence number {index+1} was length '
                                    f'{len(encoded_seq)/self.encoding_dimension}.')
                encoded_seq = self._convert_numeric_encoding_to_string_encoding(encoded_seq)
                descriptors.append(identifier)
                print(f'>{identifier}', file=out_file)
                print(encoded_seq, file=out_file)

        return descriptors, encoded_sequences


if __name__ == "__main__":
    test_spike_protein_sequence = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAISGTNGTKRFDNPVLPFNDGVYFASTEKSNI" \
                                  "IRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDG" \
                                  "YFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLS" \
                                  "ETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNV" \
                                  "YADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQ" \
                                  "SYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIDDTTDAVRDPQTLEILDITPCS" \
                                  "FGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSHRRARSVAS" \
                                  "QSIIAYTMSLGAENSVAYSNNSIAIPINFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQ" \
                                  "IYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSG" \
                                  "WTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILAR" \
                                  "LDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAI" \
                                  "CHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTHNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNI" \
                                  "QKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"

    # example One-Hot encoding
    one_hot_encoder = ProteinSequenceEncoder(encoding_type='One_hot')
    encoded_sequence = one_hot_encoder.encode_single_sequence(test_spike_protein_sequence)
    print(encoded_sequence)

    # example PAM250 encoding
    PAM250_encoder = ProteinSequenceEncoder(encoding_type='PAM250')
    encoded_sequence = PAM250_encoder.encode_single_sequence(test_spike_protein_sequence)
    print(encoded_sequence)
