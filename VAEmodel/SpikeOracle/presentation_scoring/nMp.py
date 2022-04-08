from collections import defaultdict


def eval_peptides_nMp(task, MHC_list):
    netmhcpan_peptides = defaultdict(lambda: [None] * len(MHC_list))

    for mhc_idx, mhc_name in enumerate(MHC_list):
        # print(mhc_name)
        mhc_name_2 = mhc_name.replace(":", "").replace("HLA-", "")

        filename = f"../netMHCpan/{task}_{mhc_name_2}.pep.out"
        file = open(filename, "r")
        line_nr = 1
        lines = file.readlines()

        for line in lines:

            if line_nr >= 49 and line[5:8] == "HLA":
                peptide = line[22:(22 + 9)]
                rank_el = float(line[96:(96 + 8)])

                netmhcpan_peptides[peptide][mhc_idx] = rank_el
                # print(f"peptide: {peptide}   rank_el: {rank_el}")
            line_nr += 1
        file.close()

        # print("\n")

    return dict(netmhcpan_peptides)


def score_seq_nMp(seq, MHC_list, nMp_peptide_scores):
    seq = seq.replace("-", "")
    score = 0

    epitopes = set()
    for position in range(len(seq) - 9):
        epitope = seq[position:(position + 9)]
        entry = nMp_peptide_scores[epitope]

        # print(f"position: {position} peptide: {seq[position:(position+9)]}")
        for mhc_rank in entry:
            if mhc_rank < 2.0:
                score += 1
                epitopes.add(epitope)

    return score / len(MHC_list), epitopes
