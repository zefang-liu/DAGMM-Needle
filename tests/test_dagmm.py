import sys
sys.path.append('./python')
import numpy as np
import pytest
import needle as ndl
np.random.seed(0)

_DEVICES = [ndl.cpu(), pytest.param(
    ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]


op_dagmm_params = [
    (2, 120, 3, 4),
]
@pytest.mark.parametrize("N, X, Z, K", op_dagmm_params)
@pytest.mark.parametrize("device", _DEVICES)
def test_dagmm(N, X, Z, K, device):
    from apps.models import DAGMM

    np.random.seed(0)
    model = DAGMM(device=device)

    x_array = np.random.randn(N, X)
    x = ndl.Tensor(x_array, device=device)

    z_c, x_r, z, gamma = model(x)
    assert z_c.shape == (N, 1)
    assert x_r.shape == (N, X)
    assert z.shape == (N, Z)
    assert gamma.shape == (N, K)

    phi, mu, Sigma = model.get_gmm_parameters(gamma, z)
    assert phi.shape == (K, 1)
    assert mu.shape == (K, Z)
    assert Sigma.shape == (K, Z, Z)

    E = model.get_sample_energy(z, phi, mu, Sigma)
    assert E.shape == (N,)

    reconstruction_loss = model.get_reconstruction_loss(x, x_r)
    sample_energy_loss = model.get_sample_energy_loss(E)
    penalty_loss = model.get_penalty_loss(Sigma)
    loss = model.get_loss(x, x_r, z, phi, mu, Sigma)
