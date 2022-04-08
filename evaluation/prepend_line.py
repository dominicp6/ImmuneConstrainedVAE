import os

def prepend_line(file_name, line):
    """ Insert given string as a new line at the beginning of a file """
    # define name of temporary dummy file
    dummy_file = file_name + '.bak'
    # open original file in read mode and dummy file in write mode
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        # Write given line to the dummy file
        write_obj.write(line + '\n')
        # Read lines from original file one by one and append them to the dummy file
        for line in read_obj:
            write_obj.write(line)
    # remove original file
    os.remove(file_name)
    # Rename dummy file as the original file
    os.rename(dummy_file, file_name)


def pop_top_line(file_name):
    with open(file_name, 'r+') as fp:
        # read an store all lines into list
        lines = fp.readlines()
        # move file pointer to the beginning of a file
        fp.seek(0)
        # truncate the file
        fp.truncate()

        # start writing lines except the first line
        # lines[1:] from line 2 to last line
        fp.writelines(lines[1:])


if __name__ == "__main__":
    for i in range(90):
        pop_top_line(f'./synthetic_point_mutations/{i}_reference.fasta')
        pop_top_line(f'./synthetic_point_mutations/{i}_synthetic.fasta')
        prepend_line(f'./synthetic_point_mutations/{i}_reference.fasta', f'>REF_for_VAE_{i}')
        prepend_line(f'./synthetic_point_mutations/{i}_synthetic.fasta', f'>VAE_{i}')