import sys
sys.path.append('./python')
import numpy as np
import torch
import pytest
import needle as ndl
np.random.seed(0)

_DEVICES = [ndl.cpu(), pytest.param(
    ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]


op_squeeze_params = [
    ((3, 1), None),
    ((3, 1), 0),
    ((3, 1), 1),
    ((3, 1, 4), 1),
]
@pytest.mark.parametrize("x_shape, dim", op_squeeze_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
def test_op_squeeze(x_shape, dim, backward, device):
    np.random.seed(0)
    x_array = np.random.randn(*x_shape)

    x = ndl.Tensor(x_array, device=device)
    y = ndl.squeeze(x, dim=dim)
    y_sum = y.sum()

    if backward:
        y_sum.backward()

    x_tensor = torch.Tensor(x_array).float()
    x_tensor.requires_grad = True
    if dim:
        y_tensor = torch.squeeze(x_tensor, dim=dim)
    else:
        y_tensor = torch.squeeze(x_tensor)
    y_sum_tensor = y_tensor.sum()

    if backward:
        y_sum_tensor.backward()

    err_out = np.linalg.norm(y_sum_tensor.detach().numpy() - y_sum.numpy())
    assert err_out < 1e-2, "outputs match %s, %s" % (y_sum, y_sum_tensor)

    if backward:
        err_grad = np.linalg.norm(x_tensor.grad.numpy() - x.grad.numpy())
        assert err_grad < 1e-2, "input grads match"


op_unsqueeze_params = [
    ((3,), 0),
    ((3,), 1),
    ((3, 4), 0),
    ((3, 4), 1),
    ((3, 4), 2),
]
@pytest.mark.parametrize("x_shape, dim", op_unsqueeze_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
def test_op_unsqueeze(x_shape, dim, backward, device):
    np.random.seed(0)
    x_array = np.random.randn(*x_shape)

    x = ndl.Tensor(x_array, device=device)
    y = ndl.unsqueeze(x, dim=dim)
    y_sum = y.sum()

    if backward:
        y_sum.backward()

    x_tensor = torch.Tensor(x_array).float()
    x_tensor.requires_grad = True
    y_tensor = torch.unsqueeze(x_tensor, dim=dim)
    y_sum_tensor = y_tensor.sum()

    if backward:
        y_sum_tensor.backward()

    err_out = np.linalg.norm(y_sum_tensor.detach().numpy() - y_sum.numpy())
    assert err_out < 1e-2, "outputs match %s, %s" % (y_sum, y_sum_tensor)

    if backward:
        err_grad = np.linalg.norm(x_tensor.grad.numpy() - x.grad.numpy())
        assert err_grad < 1e-2, "input grads match"


op_norm_params = [
    ((3,), 0),
    ((3, 4), 0),
    ((3, 4), 1),
]
@pytest.mark.parametrize("x_shape, dim", op_norm_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
def test_op_norm(x_shape, dim, backward, device):
    np.random.seed(0)
    x_array = np.random.randn(*x_shape)

    x = ndl.Tensor(x_array, device=device)
    y = ndl.norm(x, dim=dim)
    y_sum = y.sum()

    if backward:
        y_sum.backward()

    x_tensor = torch.Tensor(x_array).float()
    x_tensor.requires_grad = True
    y_tensor = torch.norm(x_tensor, dim=dim)
    y_sum_tensor = y_tensor.sum()

    if backward:
        y_sum_tensor.backward()

    err_out = np.linalg.norm(y_sum_tensor.detach().numpy() - y_sum.numpy())
    assert err_out < 1e-2, "outputs match %s, %s" % (y_sum, y_sum_tensor)

    if backward:
        err_grad = np.linalg.norm(x_tensor.grad.numpy() - x.grad.numpy())
        assert err_grad < 1e-2, "input grads match"


op_cosine_similarity_params = [
    ((3,), 0),
    ((3, 4), 0),
    ((3, 4), 1),
]
@pytest.mark.parametrize("x_shape, dim", op_cosine_similarity_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
def test_op_cosine_similarity(x_shape, dim, backward, device):
    np.random.seed(0)
    x1_array = np.random.randn(*x_shape)
    x2_array = np.random.randn(*x_shape)

    x1 = ndl.Tensor(x1_array, device=device)
    x2 = ndl.Tensor(x2_array, device=device)
    y = ndl.cosine_similarity(x1, x2, dim=dim)
    y_sum = y.sum()

    if backward:
        y_sum.backward()

    x1_tensor = torch.Tensor(x1_array).float()
    x2_tensor = torch.Tensor(x2_array).float()
    x1_tensor.requires_grad = True
    x2_tensor.requires_grad = True
    y_tensor = torch.cosine_similarity(x1_tensor, x2_tensor, dim=dim)
    y_sum_tensor = y_tensor.sum()

    if backward:
        y_sum_tensor.backward()

    err_out = np.linalg.norm(y_sum_tensor.detach().numpy() - y_sum.numpy())
    assert err_out < 1e-2, "outputs match %s, %s" % (y_sum, y_sum_tensor)

    if backward:
        err_grad = np.linalg.norm(x1_tensor.grad.numpy() - x1.grad.numpy()) \
            + np.linalg.norm(x2_tensor.grad.numpy() - x2.grad.numpy())
        assert err_grad < 1e-2, "input grads match"


op_pairwise_distance_params = [
    ((3,), 0),
    ((3, 4), 1),
]
@pytest.mark.parametrize("x_shape, dim", op_pairwise_distance_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
def test_op_pairwise_distance(x_shape, dim, backward, device):
    np.random.seed(0)
    x1_array = np.random.randn(*x_shape)
    x2_array = np.random.randn(*x_shape)

    x1 = ndl.Tensor(x1_array, device=device)
    x2 = ndl.Tensor(x2_array, device=device)
    y = ndl.pairwise_distance(x1, x2, dim=dim)
    y_sum = y.sum()

    if backward:
        y_sum.backward()

    x1_tensor = torch.Tensor(x1_array).float()
    x2_tensor = torch.Tensor(x2_array).float()
    x1_tensor.requires_grad = True
    x2_tensor.requires_grad = True
    y_tensor = torch.pairwise_distance(x1_tensor, x2_tensor)
    y_sum_tensor = y_tensor.sum()

    if backward:
        y_sum_tensor.backward()

    err_out = np.linalg.norm(y_sum_tensor.detach().numpy() - y_sum.numpy())
    assert err_out < 1e-2, "outputs match %s, %s" % (y_sum, y_sum_tensor)

    if backward:
        err_grad = np.linalg.norm(x1_tensor.grad.numpy() - x1.grad.numpy()) \
            + np.linalg.norm(x2_tensor.grad.numpy() - x2.grad.numpy())
        assert err_grad < 1e-2, "input grads match"


op_softmax_params = [
    ((3, 4), 0),
    ((3, 4), 1),
    ((4, 5), 1),
]
@pytest.mark.parametrize("x_shape, dim", op_softmax_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
def test_op_softmax(x_shape, dim, backward, device):
    np.random.seed(0)
    x_array = np.random.randn(*x_shape)

    x = ndl.Tensor(x_array, device=device)
    y = ndl.softmax(x, dim=dim)
    y_sum = y.sum()

    if backward:
        y_sum.backward()

    x_tensor = torch.Tensor(x_array).float()
    x_tensor.requires_grad = True
    y_tensor = torch.softmax(x_tensor, dim=dim)
    y_sum_tensor = y_tensor.sum()

    if backward:
        y_sum_tensor.backward()

    err_out = np.linalg.norm(y_sum_tensor.detach().numpy() - y_sum.numpy())
    assert err_out < 1e-2, "outputs match %s, %s" % (y_sum, y_sum_tensor)

    if backward:
        err_grad = np.linalg.norm(x_tensor.grad.numpy() - x.grad.numpy())
        assert err_grad < 1e-2, "input grads match"


op_bmm_params = [
    ((2, 4, 4), (2, 4, 4)),
    ((2, 3, 4), (2, 4, 5)),
]
@pytest.mark.parametrize("A_shape, B_shape", op_bmm_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
def test_op_bmm(A_shape, B_shape, backward, device):
    np.random.seed(0)
    A_array = np.random.randn(*A_shape)
    B_array = np.random.randn(*B_shape)

    A = ndl.Tensor(A_array, device=device)
    B = ndl.Tensor(B_array, device=device)
    y = ndl.bmm(A, B)
    y_sum = y.sum()

    if backward:
        y_sum.backward()

    A_tensor = torch.Tensor(A_array).float()
    B_tensor = torch.Tensor(B_array).float()
    A_tensor.requires_grad = True
    B_tensor.requires_grad = True
    y_tensor = torch.bmm(A_tensor, B_tensor)
    y_sum_tensor = y_tensor.sum()

    if backward:
        y_sum_tensor.backward()

    err_out = np.linalg.norm(y_sum_tensor.detach().numpy() - y_sum.numpy())
    assert err_out < 1e-2, "outputs match %s, %s" % (y_sum, y_sum_tensor)

    if backward:
        err_grad = np.linalg.norm(A_tensor.grad.numpy() - A.grad.numpy()) \
            + np.linalg.norm(B_tensor.grad.numpy() - B.grad.numpy())
        assert err_grad < 1e-2, "input grads match"


op_cholesky_params = [
    (4, 4),
    (2, 4, 4),
]
@pytest.mark.parametrize("A_shape", op_cholesky_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
def test_op_cholesky(A_shape, backward, device):
    from sklearn.datasets import make_spd_matrix
    if len(A_shape) == 2:
        A_array = make_spd_matrix(n_dim=A_shape[0], random_state=0)
    else:
        A_array = np.zeros(A_shape)
        for i in range(A_shape[0]):
            A_array[i] = make_spd_matrix(n_dim=A_shape[1], random_state=0)

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
