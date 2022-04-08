import pandas as pd
def get_delta_delta_g_from_file(filename):

    total_delta_delta_g = 0

    with open(filename, 'r') as file:
        for line_number, line in enumerate(file):
            if line_number == 0:
                continue
            else:
                line_segments = line.split('\t')
                total_delta_delta_g += float(line_segments[-1])

    return total_delta_delta_g


if __name__ == "__main__":

    ddg_sum_dict={}
    #foldername = 'gen2_point_mutations'
    #foldername = 'gen3_point_mutations'
    #foldername = '11gram_point_mutations'
    #foldername = 'random75_point_mutations'
    foldername = 'gen4_point_mutations'
    for seq_number in range(601):
        filename = './' + foldername +'/'+ str(seq_number)+'.txt'
        ddg_sum = get_delta_delta_g_from_file(filename)
        ddg_sum_dict[str(seq_number)] = ddg_sum
    print(ddg_sum_dict)
    ddg_sum_df = pd.DataFrame.from_dict(ddg_sum_dict,orient='index', columns=['ddg_sum'])
    print(ddg_sum_df)
    ddg_sum_df.to_csv(foldername + '_ddg_sum_table.csv', index=True)
