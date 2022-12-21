import sys
sys.path.append('./python')
import numpy as np
import pytest
import needle as ndl
np.random.seed(0)

_DEVICES = [ndl.cpu(), pytest.param(
    ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]


def test_dagmm():
    pass
