import sys
sys.path.append('./python')
import numpy as np
import torch
import pytest
import needle as ndl
np.random.seed(0)

_DEVICES = [ndl.cpu(), pytest.param(
    ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]

op_cholesky_params = [
    3, 5, 10,
]
@pytest.mark.parametrize("n_dim", op_cholesky_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
def test_op_cholesky(n_dim, backward, device):
    from sklearn.datasets import make_spd_matrix
    A_array = make_spd_matrix(n_dim=n_dim, random_state=0)

    A = ndl.Tensor(A_array, device=device)
    y = ndl.cholesky(A)
    y_sum = y.sum()

    if backward:
        y_sum.backward()

    A_tensor = torch.Tensor(A_array).float()
    A_tensor.requires_grad = True
    y_tensor = torch.linalg.cholesky(A_tensor)
    y_sum_tensor = y_tensor.sum()

    if backward:
        y_sum_tensor.backward()

    err_out = np.linalg.norm(y_sum_tensor.detach().numpy() - y_sum.numpy())
    assert err_out < 1e-2, "outputs match %s, %s" % (y_sum, y_sum_tensor)

    if backward:
        err_grad = np.linalg.norm(A_tensor.grad.numpy() - A.grad.numpy())
        assert err_grad < 1e-2, "input grads match"
