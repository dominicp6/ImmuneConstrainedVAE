import torch
import pMHC
from pMHC.data import MhcAllele
from pMHC.data.utils import get_input_rep_PSEUDO, move_dict_to_device, convert_example_to_batch


def score_seq_IB(model, seq, MHC_list, input_score_dict, show=False, weak_antigenic_threshold=None, strong_antigenic_threshold=None):
    seq = seq.replace("-", "")
    res = 0
    res_weak = 0
    res_strong = 0
    avg_pred = 0
    for position in range(len(seq) - 9):
        n_flank = seq[max(position - 15, 0):position]
        peptide = seq[position:position + 9]
        c_flank = seq[position + 9:min(position + 9 + 15, len(seq))]

        input = f"{n_flank}_{peptide}_{c_flank}"
        if not input in input_score_dict:
            input_score_dict[input] = []
            for mhc_idx, mhc_name in enumerate(MHC_list):
                mhc_seq = MhcAllele.mhc_alleles[mhc_name].pseudo_seq
                example = get_input_rep_PSEUDO(n_flank, peptide, c_flank, mhc_seq, model)

                pred = float(torch.sigmoid(model(move_dict_to_device(convert_example_to_batch(example), model))))
                input_score_dict[input].append(pred)
                avg_pred += pred

                if pred > 0.5:
                    res += 1

                if weak_antigenic_threshold is not None and pred > weak_antigenic_threshold[mhc_name]:
                    res_weak += 1

                if strong_antigenic_threshold is not None and pred > strong_antigenic_threshold[mhc_name]:
                    res_strong += 1

                if show:
                    offset = "    " if pred < 0.5 else ""
                    print(f"{offset} {position} {mhc_name}: {n_flank}_{peptide}_{c_flank}: {pred:.3f}")

        else:
            preds = input_score_dict[input]

            for mhc_idx, mhc_name in enumerate(MHC_list):
                pred = preds[mhc_idx]
                avg_pred += pred

                if pred > 0.5:
                    res += 1

                if weak_antigenic_threshold is not None and pred > weak_antigenic_threshold[mhc_name]:
                    res_weak += 1

                if strong_antigenic_threshold is not None and pred > strong_antigenic_threshold[mhc_name]:
                    res_strong += 1

                if show:
                    offset = "    " if pred < 0.5 else ""
                    print(f"{offset} {position} {mhc_name}: {n_flank}_{peptide}_{c_flank}: {pred:.3f}")

    return res / len(MHC_list), res_weak / len(MHC_list), res_strong / len(MHC_list), avg_pred / (len(seq) - 9)