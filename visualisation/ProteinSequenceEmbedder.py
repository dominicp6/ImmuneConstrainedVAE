import os
import numpy as np
from math import log10, floor
import umap.umap_ as umap  # make sure that you install "umap-learn" not "umap"
import matplotlib.pyplot as plt
from amino_acid_encoding import ProteinSequenceEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap

default_hyperparameters = \
    {'PCA': {'whiten': False},
     'tSNE': {'perplexity': 30, 'learning_rate': 200, 'n_iter': 1000,
              'metric': 'cosine', 'PCA_dim': 50, 'whiten': False},
     'UMAP': {'n_neighbors': 15, 'metric': 'cosine', 'learning_rate': 1, 'seed': 42}}


class ProteinSequenceEmbedder:

    def __init__(self,
                 encoding_type,
                 embedding_type,
                 encodings_directory=os.path.join("..", "data", "encodings"),
                 data_directory=os.path.join("..", "data", "spike_protein_sequences")):

        assert embedding_type == 'PCA' \
               or embedding_type == 'UMAP' \
               or embedding_type == 'tSNE', "embedding type must be either PCA, UMAP or tSNE"

        self.encoding_type = encoding_type
        self.embedding_type = embedding_type
        self.encodings_directory = encodings_directory
        self.data_directory = data_directory

        self.embedding_data = {'embeddings': [],
                               'frequencies': [],
                               'other_descriptors': [],
                               'meta_data': []}

    def _encode_sequences(self, infile, mask=None):
        """
        Encodes aligned sequences from a fasta file.

        Valid encoding types are:
        'One_hot', 'One_hot_6_bit', 'Binary_5_bit', 'Hydrophobicity_matrix', 'Meiler_parameters', 'Acthely_factors',
        'PAM250', 'BLOSUM62', 'Miyazawa_energies', 'Micheletti_potentials', 'AESNN3', 'ANN4D', 'ProtVec'

        Note:
        If a mask is required, then it should be generated by running the get_epitope_mask function from epitope_mask.py

        :param infile: the fasta file of aligned sequences.
        :param mask: an epitope mask to apply to the encoding (optional).
        :return:
        frequencies       - the number of times that each sequence appeared (used during visualisation)
        encoded_sequences - the encoded amino acid sequences
        """
        encoder = ProteinSequenceEncoder(encoding_type=self.encoding_type,
                                         data_directory=self.data_directory,
                                         encodings_directory=self.encodings_directory,
                                         mask=mask)

        descriptors, encoded_sequences = encoder.encode_from_fasta_file(infile, f'{infile}_{self.encoding_type}.txt')

        return descriptors, encoded_sequences

    @staticmethod
    def embed_with_PCA(encoded_sequences, hyperparameters=default_hyperparameters['PCA']):
        """
        Embeds a list of encoded sequences down to two dimensions using PCA.
        """
        reducer = PCA(n_components=2, whiten=hyperparameters['whiten'])
        sequence_embeddings = reducer.fit_transform(encoded_sequences)
        explained_variance = reducer.explained_variance_ratio_
        return sequence_embeddings, explained_variance

    @staticmethod
    def embed_with_tSNE(encoded_sequences, hyperparameters=default_hyperparameters['tSNE']):
        """
        Embeds a list of encoded sequences down to two dimensions using tSNE.
        """
        pca = PCA(n_components=hyperparameters['PCA_dim'], whiten=hyperparameters['whiten'])
        pca_reduced_embeddings = pca.fit_transform(encoded_sequences)
        reducer = TSNE(init='pca',
                       perplexity=hyperparameters['perplexity'],
                       learning_rate=hyperparameters['learning_rate'],
                       n_iter=hyperparameters['n_iter'],
                       metric=hyperparameters['metric']
                       )
        sequence_embeddings = reducer.fit_transform(pca_reduced_embeddings)
        return sequence_embeddings

    @staticmethod
    def embed_with_UMAP(encoded_sequences, hyperparameters=default_hyperparameters['UMAP']):
        """
        Embeds a list of encoded sequences down to two dimensions using UMAP.
        """
        reducer = umap.UMAP(metric=hyperparameters['metric'],
                            random_state=hyperparameters['seed'],
                            n_neighbors=hyperparameters['n_neighbors'],
                            learning_rate=hyperparameters['learning_rate'])
        sequence_embeddings = reducer.fit_transform(encoded_sequences)
        return sequence_embeddings

    def _get_sizes_legend_handles_and_labels(self, marker_size, marker_size_power_scaling):
        def round_sig(x, sig=2):
            return float('{:.{p}g}'.format(x, p=sig))

        max_size = round_sig(max(self.embedding_data['frequencies']), sig=2)
        min_size = round_sig(min(self.embedding_data['frequencies']), sig=2)
        numbers = [round_sig(num, sig=2) for num in np.logspace(np.log10(min_size), np.log10(max_size), num=5, base=10)]
        handles = tuple([plt.scatter([], [], s=(num)**(2/marker_size_power_scaling) * marker_size, marker='o', color='#555555') for num in numbers])
        labels = tuple([str(int(num)) for num in numbers])

        return handles, labels

    def plot_embedding_map(self,
                           marker_size=5.0,
                           descriptor_number=3,
                           color_map='Set2',
                           save_image=False,
                           marker_size_power_scaling=2,
                           hide_meta_data=False,
                           title=None,
                           title_fontsize=20,
                           axis_fontsize=12,
                           axis_title_fontsize=15,
                           legend_fontsize=13,
                           legend_title_fontsize=14):
        """
        Plots a 2D map from a list of embeddings and their associated sequence frequency.

        :param color_map: colour map to use for the plot
        :param marker_size: relative point markerSize
        :param descriptor_number: identifies which descriptor will be used when colouring sequences
                                  2 - date of sequence
                                  3 - variant label of sequence
        """

        marker_sizes = [(frequency)**(2/marker_size_power_scaling) * marker_size for frequency in self.embedding_data['frequencies']]
        plt.rc('legend', fontsize=legend_fontsize)
        plt.rcParams['legend.title_fontsize'] = legend_title_fontsize

        # TODO: fix this for dates
        label_to_class_number = dict()
        class_numbers = []
        for label in [descriptor[descriptor_number-2] for descriptor in self.embedding_data['other_descriptors']]:
            if label not in label_to_class_number.keys():
                label_to_class_number[label] = len(label_to_class_number)
            class_numbers.append(label_to_class_number[label])

        fig, ax = plt.subplots()

        alpha = [0.25 if class_number != 2 else 0.05 for class_number in class_numbers]

        x_natural = [self.embedding_data['embeddings'][idx][0] for idx in range(len(class_numbers))
                     if class_numbers[idx] == 0]
        y_natural = [self.embedding_data['embeddings'][idx][1] for idx in range(len(class_numbers))
                     if class_numbers[idx] == 0]
        natural_marker_sizes = [size for idx, size in enumerate(marker_sizes)
                          if class_numbers[idx] == 0]
        x_non_natural = [self.embedding_data['embeddings'][idx][0] for idx in range(len(class_numbers))
                     if class_numbers[idx] != 0]
        y_non_natural = [self.embedding_data['embeddings'][idx][1] for idx in range(len(class_numbers))
                     if class_numbers[idx] != 0]
        non_natural_marker_sizes = [size for idx, size in enumerate(marker_sizes)
                          if class_numbers[idx] != 0]
        non_natural_class_numbers = [class_numbers[idx] for idx in range(len(class_numbers)) if class_numbers[idx] != 0]
        # plot
        # scatter = ax.scatter(self.embedding_data['embeddings'][:, 0],
        #                      self.embedding_data['embeddings'][:, 1],
        #                      c=class_numbers,                 # variant class information
        #                      s=marker_sizes,
        #                      alpha=alpha,
        #                      cmap=plt.get_cmap(color_map))



        scatter1 = ax.scatter(x_natural,
                   y_natural,
                   s=natural_marker_sizes,
                   alpha=0.05,
                   color='black')
        my_color_map = LinearSegmentedColormap.from_list("mycmap", [(227/256, 227/256, 30/256), (227/256,30/256,30/256), (34/256, 70/256, 152/256)])
        scatter2 = ax.scatter(x_non_natural,
                             y_non_natural,
                             c=non_natural_class_numbers,                 # variant class information
                             s=non_natural_marker_sizes,
                             alpha=0.75,
                             cmap=my_color_map)# plt.get_cmap(color_map))

        # legends
        labels_legend = ax.legend(handles=scatter2.legend_elements()[0],
                                  labels=['random-mutator', 'language model', 'VAE model'],#label_to_class_number.keys(),
                                  loc="lower left",
                                  title="Variant")     # variant legend
        ax.add_artist(labels_legend)
        handles, size_labels = self._get_sizes_legend_handles_and_labels(marker_size, marker_size_power_scaling)
        plt.legend(handles, size_labels, loc="lower right", title="Frequency", labelspacing=2.5, borderpad=2, handletextpad=1.5)          # frequency legend

        # title, labels and aspect ratio
        ax.set_aspect('equal', 'datalim')
        fig.set_size_inches(15, 15)
        plt.margins(0.25)
        if title is not None:
            ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlabel(f'{self.embedding_type} 1', fontsize=axis_title_fontsize)
        ax.set_ylabel(f'{self.embedding_type} 2', fontsize=axis_title_fontsize)

        plt.xticks(fontsize=axis_fontsize)
        plt.yticks(fontsize=axis_fontsize)

        # annotate with meta data
        if not hide_meta_data:
            ax.text(0.95, 0.95, '\n'.join(self.embedding_data['meta_data']),
                           horizontalalignment='right', verticalalignment='top',
                           transform=plt.gca().transAxes)

        if save_image:
            plt.savefig(f'{self.embedding_type}_{self.embedding_data["meta_data"]}.pdf',
                        format='pdf',
                        bbox_inches='tight')

        plt.show()

    @staticmethod
    def _parse_sequence_descriptors(descriptors):
        frequencies = []
        other_descriptors = []
        for seq_number, descriptor in enumerate(descriptors):
            split_descriptor = descriptor.split('|')
            try:
                frequencies.append(float(split_descriptor[0]))
            except:
                raise Exception('No frequency value found for sequence in fasta file.')

            if len(split_descriptor) > 1:
                other_descriptors.append(split_descriptor[1:])
                assert len(split_descriptor[1:]) == len(other_descriptors[0]), \
                    f"Not all sequences have the same number of descriptors " \
                    f"(sequence 1 has {len(other_descriptors[0])+1} descriptors, whilst sequence" \
                    f" {seq_number+1} has {len(split_descriptor)} descriptors)"

        return frequencies, other_descriptors

    def embed_sequences(self, infile, hyperparameters, mask):

        # check hyperparameters
        hyperparameters = self.validate_hyperparameters(hyperparameters)

        # encode
        descriptors, encoded_sequences = self._encode_sequences(infile, mask)

        # get frequency, median date and variant label from its descriptor
        frequencies, other_descriptors = self._parse_sequence_descriptors(descriptors)

        if mask is not None:
            mask_present = True
        else:
            mask_present = False

        meta_data = [self.embedding_type, f'encoding: {self.encoding_type}', f'mask: {mask_present}']
        meta_data.extend([f'{param}: {value}' for param, value in hyperparameters.items()])

        if self.embedding_type == 'PCA':
            sequence_embeddings, explained_variance = self.embed_with_PCA(encoded_sequences, hyperparameters)
            meta_data.extend([f'{round(variance, 3)}' for variance in explained_variance])
        elif self.embedding_type == 'tSNE':
            sequence_embeddings = self.embed_with_tSNE(encoded_sequences, hyperparameters)
        elif self.embedding_type == 'UMAP':
            sequence_embeddings = self.embed_with_UMAP(encoded_sequences, hyperparameters)
        else:
            raise Exception('Invalid method. Must be PCA, tSNE or UMAP.')

        self.embedding_data['embeddings'] = sequence_embeddings
        self.embedding_data['frequencies'] = frequencies
        self.embedding_data['other_descriptors'] = other_descriptors

        self.embedding_data['meta_data'] = meta_data

        return self.embedding_data

    def validate_hyperparameters(self, hyperparameter_dict):

        if hyperparameter_dict is None:
            hyperparameter_dict = default_hyperparameters[self.embedding_type]
        else:
            assert set(hyperparameter_dict.keys()).issubset(default_hyperparameters[self.embedding_type]), \
                f"Invalid entries for hyperparameters. " \
                f"Permitted dict keys are {default_hyperparameters[self.embedding_type]}"
            unspecified_hyperparameters = \
                set(default_hyperparameters[self.embedding_type]).difference(hyperparameter_dict.keys())
            for hyperparameter in unspecified_hyperparameters:
                hyperparameter_dict[hyperparameter] = default_hyperparameters[self.embedding_type][hyperparameter]

        return hyperparameter_dict

    def generate_embedding_map(self,
                               infile,
                               hyperparameters=None,
                               marker_size=1,
                               marker_size_power_scaling=2,
                               descriptor_number=3,
                               color_map='Set2',
                               mask=None,
                               save_image=False,
                               hide_meta_data=False,
                               title=None,
                               title_fontsize=20,
                               axis_fontsize=12,
                               axis_title_fontsize=15,
                               legend_fontsize=13,
                               legend_title_fontsize=14):
        """
        Plots a 2D representation of the dimensionality-reduced embeddings of the encoded sequences in a fasta database.

        :param hyperparameters: Dictionary of hyperparameters for the embedding algorithm.
        :param infile: a database of aligned fasta sequences.
        :param marker_size:  relative point markerSize.
        :param descriptor_number: identifies which descriptor will be used when colouring sequences
                                  2 - date of sequence
                                  3 - variant label of sequence
        :param mask:        An epitope mask to be applied to the encoded sequences (optional).
        """

        self.embed_sequences(infile=infile, hyperparameters=hyperparameters, mask=mask)
        self.plot_embedding_map(marker_size=marker_size,
                                marker_size_power_scaling=marker_size_power_scaling,
                                descriptor_number=descriptor_number,
                                color_map=color_map,
                                save_image=save_image,
                                hide_meta_data=hide_meta_data,
                                title=title,
                                title_fontsize=title_fontsize,
                                axis_fontsize=axis_fontsize,
                                axis_title_fontsize=axis_title_fontsize,
                                legend_fontsize=legend_fontsize,
                                legend_title_fontsize=legend_title_fontsize)


if __name__== "__main__":
    parent_dir = os.path.abspath('..')  # path to parent directory
    data_dir = parent_dir + '/data'  # path to datasets

    for method in ['tSNE', 'PCA']:
        for encoding_type in ['BLOSUM62']:
            embedder = ProteinSequenceEmbedder(encoding_type=encoding_type, embedding_type=method)
            embedder.generate_embedding_map(infile='gan0,5.afa', marker_size=6, marker_size_power_scaling=4,
                                            descriptor_number=3, color_map='tab20', save_image=True, hide_meta_data=True,
                                            axis_fontsize=23, axis_title_fontsize=30, legend_fontsize=18,
                                            legend_title_fontsize=20)