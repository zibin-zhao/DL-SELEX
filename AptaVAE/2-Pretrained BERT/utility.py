import Levenshtein as lev
import numpy as np


__all__ = ['compute_edit_distance', 'onehot_to_seq']

def compute_edit_distance(seq1, seq2):
    """
    This function computes the Edit distance between two sequences
    """
    return lev.distance(seq1, seq2)

def onehot_to_seq(onehot_seq):
    bases = ['<', 'A', 'T', 'C', 'G', 'N', '>']
    seq = ''
    for i in range(0, len(onehot_seq), 7):      # as we flatten the vector, hence every 5 samples is a base
        try:
            #print(len(onehot_seq))
            seq += bases[np.argmax(onehot_seq[i:i+7])]
        except IndexError:
            print(f"Error at index {i} with onehot_seq length {len(onehot_seq)}")
            break
    return seq