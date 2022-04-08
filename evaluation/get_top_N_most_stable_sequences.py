import pandas as pd
import os
import shutil


def get_top_N_most_stable_sequences(path_to_sequences, ddgun_csv_file, N):
    ddgun_results_df = pd.read_csv(ddgun_csv_file)
    columns = ddgun_results_df.columns.tolist()
    ddgun_results_df = ddgun_results_df.sort_values(columns[1])
    N_most_stable_seq_ids = ddgun_results_df[columns[0]].head(N)
    N_most_stable_ddgun_results = ddgun_results_df[columns[1]]

    print(N_most_stable_seq_ids)
    print(N_most_stable_ddgun_results)

    for seq_id in N_most_stable_seq_ids:
        try:
            os.mkdir(f'{path_to_sequences}_{N}_most_stable')
        except:
            pass # directory already exists
        shutil.copy(os.path.join(path_to_sequences, f'{seq_id}_synthetic.fasta'),
                    os.path.join(f'{path_to_sequences}_{N}_most_stable', f'{seq_id}_synthetic.fasta'))

    for row_idx, seq_id in enumerate(N_most_stable_seq_ids):
        seq = None
        with open(os.path.join(f'{path_to_sequences}_{N}_most_stable', f'{seq_id}_synthetic.fasta'), 'r') as f:
            f.readline()  # skip the header
            seq = f.readline().replace('-', '')

        with open(os.path.join(f'{path_to_sequences}_{N}_most_stable', f'{seq_id}_synthetic.fasta'), 'w') as f:
            print(f'>seq_id: {seq_id}, ddgun_score: {ddgun_results_df.iloc[row_idx][columns[1]]}', file=f)
            print(seq, file=f)



if __name__ == "__main__":

    #foldername = 'gen2_point_mutations'
    #ddg_sum_file = 'ddg_sum_table.csv'
    #foldername = '11gram_point_mutations'
    #foldername = 'random75_point_mutations'
    #foldername = 'gen3_point_mutations'
    foldername = 'gen4_point_mutations'
    ddg_sum_file = foldername + '_ddg_sum_table.csv'

    get_top_N_most_stable_sequences('./' + foldername, ddg_sum_file, 10)


