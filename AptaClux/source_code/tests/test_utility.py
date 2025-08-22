import os
import sys
import types

# Stub external dependencies so that utility.py can be imported without the
# heavy optional packages actually being installed.
# numpy
numpy = types.ModuleType("numpy")

def argmax(seq):
    return max(range(len(seq)), key=lambda i: seq[i])


def array(seq):
    return list(seq)

numpy.argmax = argmax
numpy.array = array
sys.modules.setdefault("numpy", numpy)

# Other optional modules
for name in ["Levenshtein", "torch"]:
    sys.modules.setdefault(name, types.ModuleType(name))

# matplotlib
matplotlib = types.ModuleType("matplotlib")
matplotlib.__path__ = []  # make it a package
sys.modules.setdefault("matplotlib", matplotlib)
sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("pyplot"))

# sklearn.manifold.TSNE
sklearn = types.ModuleType("sklearn")
sklearn.__path__ = []
manifold = types.ModuleType("sklearn.manifold")
setattr(manifold, "TSNE", object)
sklearn.manifold = manifold
sys.modules.setdefault("sklearn", sklearn)
sys.modules.setdefault("sklearn.manifold", manifold)

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utility import onehot_to_seq, onehot_to_2d


def test_onehot_to_seq_known_vector():
    onehot = np.array([
        1, 0, 0, 0,  # A
        0, 1, 0, 0,  # T
        0, 0, 1, 0,  # C
        0, 0, 0, 1,  # G
    ])
    assert onehot_to_seq(onehot) == "ATCG"


def test_onehot_to_seq_non_multiple_length():
    onehot = np.array([
        1, 0, 0, 0,  # A
        0, 1, 0, 0,  # T
        1, 0,        # Partial encoding for A
    ])
    result = onehot_to_seq(onehot)
    assert result == "ATA"


def test_onehot_to_2d_symbol_conversion():
    onehot = np.array([
        1, 0, 0,  # '.'
        0, 1, 0,  # '('
        0, 0, 1,  # ')'
    ])
    assert onehot_to_2d(onehot) == ".()"
