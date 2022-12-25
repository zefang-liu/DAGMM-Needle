import sys
sys.path.append('./python')
import numpy as np
import pytest
import needle as ndl
from needle import backend_ndarray as nd
np.random.seed(0)

_DEVICES = [ndl.cpu(), pytest.param(
    ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]


TRAIN = [True, False]
@pytest.mark.parametrize("train", TRAIN)
def test_kdd_cup_dataset(train):
    dataset = ndl.data.KDDCUPDataset(train=train)

    if train:
        assert len(dataset) == 395216
    else:
        assert len(dataset) == 98805

    example = dataset[np.random.randint(len(dataset))]
    assert(isinstance(example, tuple))

    X, y = example
    assert isinstance(X, np.ndarray)
    assert X.shape == (118,)


BATCH_SIZES = [1, 16]
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("train", TRAIN)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_kdd_cup_dataloader(batch_size, train, device):
    dataset = ndl.data.KDDCUPDataset(train=train)
    dataloader = ndl.data.DataLoader(dataset, batch_size)

    X = None
    y = None
    for batch in dataloader:
        X, y = batch
        break

    assert X is not None and y is not None
    assert isinstance(X.cached_data, nd.NDArray)
    assert isinstance(X, ndl.Tensor)
    assert isinstance(y, ndl.Tensor)
    assert X.dtype == 'float32'
