import subprocess

for i in SOMETHING:
    consensus_sequence_fasta=i['consensus_sequence']
    muts_file = i['muts']
    executable = ['python3', './ddgun_seq.py', str(consensus_sequence_fasta+.'fasta'),
                    str(muts_file+.'muts'), '>', str(i +'.txt')]
    subprocess.call(executable)
