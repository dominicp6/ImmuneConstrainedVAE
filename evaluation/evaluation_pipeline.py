from fasta_preprocessing_tools import combine_two_databases
from Bio.Align.Applications import MuscleCommandline
from filter_by_conserved_regions import save_sequences_that_obey_conserved_regions
from get_point_mutation_list import generate_point_mutation_lists_for_entire_database


def execute_pipeline(path_to_muscle_executable='/home/dominic/miniconda3/pkgs/muscle-3.8.1551-h7d875b9_6/bin/muscle'):
    print('----------------------------------- EVALUATION PIPELINE ---------------------------------------------------')
    folder_to_generated_seqs = input("The relative path to the generated sequences folder: ")
    experiment_name = input("The experiment name: ")
    print('1. Merging generated sequences ----------------------------------------------------------------------------')
    skip_option = input('Skip?: ')
    if skip_option == "y" or skip_option == "Y" or skip_option == "yes":
        pass
    else:
        print(f'Check that the file names in {folder_to_generated_seqs} are:')
        print(f'{experiment_name}_high.fasta')
        print(f'{experiment_name}_intermediate.fasta')
        print(f'{experiment_name}_low.fasta')
        input('Press Enter to confirm...')
        print('Merging the experiment files')
        combine_two_databases(database1=folder_to_generated_seqs+f'/{experiment_name}_high.fasta',
                              database2=folder_to_generated_seqs+f'/{experiment_name}_intermediate.fasta',
                              variant_database1='high',
                              variant_database2='intermediate',
                              outfile=folder_to_generated_seqs+f'/.{experiment_name}_high_intermediate.fasta')
        combine_two_databases(database1=folder_to_generated_seqs+f'/.{experiment_name}_high_intermediate.fasta',
                              database2=folder_to_generated_seqs+f'/{experiment_name}_low.fasta',
                              variant_database2='low',
                              outfile=folder_to_generated_seqs+f'/{experiment_name}_merged.fasta')
        print('Files successfully merged.')

    print('2. Merging with natural sequences -------------------------------------------------------------------------')
    skip_option = input('Skip?: ')
    if skip_option == "y" or skip_option == "Y" or skip_option == "yes":
        pass
    else:
        print('Confirm that a downsampled natural fasta file is at:')
        downsampled_natural_file_location = '../data/natural_downsampled.afa'
        print(downsampled_natural_file_location)
        input('Press Enter to confirm')
        print('Merging with natural sequences')
        combine_two_databases(database1=folder_to_generated_seqs+f'/{experiment_name}_merged.fasta',
                              database2=downsampled_natural_file_location,
                              variant_database2='natural',
                              outfile=folder_to_generated_seqs+f'/{experiment_name}_merged_natural.fasta')
        print('Files successfully merged.')

    print('3. Aligning sequences -------------------------------------------------------------------------------------')
    skip_option = input('Skip?: ')
    if skip_option == "y" or skip_option == "Y" or skip_option == "yes":
        pass
    else:
        muscle_command = MuscleCommandline(path_to_muscle_executable,
                                           input=folder_to_generated_seqs+f'/{experiment_name}_merged_natural.fasta',
                                           out=folder_to_generated_seqs+f'/{experiment_name}_merged_natural_aligned.afa')
        print('Execute the MUSCLE command:')
        print(muscle_command)
        input('Press Enter once the sequences have finished aligning..')

    print('4. Filter sequences down to those that satisfy conserved regions ------------------------------------------')
    skip_option = input('Skip?: ')
    if skip_option == "y" or skip_option == "Y" or skip_option == "yes":
        pass
    else:
        save_sequences_that_obey_conserved_regions(
            input=folder_to_generated_seqs + f'/{experiment_name}_merged_natural_aligned.afa',
            output=folder_to_generated_seqs + f'/{experiment_name}_merged_conserved_regions.afa')
        print('Successfully filtered sequences.')

    print('5. Generate the point mutation list for the sequences -----------------------------------------------------')
    skip_option = input('Skip?: ')
    if skip_option == "y" or skip_option == "Y" or skip_option == "yes":
        pass
    else:
        generate_point_mutation_lists_for_entire_database(database=folder_to_generated_seqs+f'/{experiment_name}_merged_conserved_regions.afa',
                                                          exp_name=experiment_name)
        print('Successfully generated point mutations.')
        print('')
        print('Now you may run DDGun and identify the most stable sequences.')


if __name__ == "__main__":
    execute_pipeline()