import sys
sys.path.append('./python')
import numpy as np
import torch
import pytest
import needle as ndl
from apps.models import DAGMM
from scipy.special import softmax
np.random.seed(0)

dagmm_params = [
    (16, 120, 3, 4),
]

_DEVICES = [ndl.cpu(), pytest.param(
    ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]


def get_gmm_parameters(gamma, z):
    N = gamma.shape[0]
    phi = torch.sum(gamma, dim=0) / N

    mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0)
    mu = mu / torch.sum(gamma, dim=0).unsqueeze(-1)

    z_mean = (z.unsqueeze(1) - mu.unsqueeze(0))
    sigma = torch.sum(
        gamma.unsqueeze(-1).unsqueeze(-1) * z_mean.unsqueeze(-1)
        * z_mean.unsqueeze(-2), dim=0
    ) / torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

    return phi, mu, sigma


@pytest.mark.parametrize("N, X, Z, K", dagmm_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_gmm_parameters(N, X, Z, K, device):
    np.random.seed(0)
    gamma_array = softmax(np.random.randn(N, K), axis=1)
    z_array = np.random.randn(N, Z)

    gamma = ndl.Tensor(gamma_array, device=device)
    z = ndl.Tensor(z_array, device=device)
    phi, mu, sigma = DAGMM.get_gmm_parameters(gamma, z)

    assert phi.shape == (K,)
    assert mu.shape == (K, Z)
    assert sigma.shape == (K, Z, Z)

    gamma_tensor = torch.Tensor(gamma_array).float()
    z_tensor = torch.Tensor(z_array).float()
    phi_tensor, mu_tensor, sigma_tensor = get_gmm_parameters(gamma_tensor, z_tensor)

    err_phi = np.linalg.norm(phi_tensor.detach().numpy() - phi.numpy())
    assert err_phi < 1e-2, "phi match %s, %s" % (phi, phi_tensor)

    err_mu = np.linalg.norm(mu_tensor.detach().numpy() - mu.numpy())
    assert err_mu < 1e-2, "mu match %s, %s" % (mu, phi_tensor)

    err_sigma = np.linalg.norm(sigma_tensor.detach().numpy() - sigma.numpy())
    assert err_sigma < 1e-2, "sigma match %s, %s" % (sigma, sigma_tensor)


def get_sample_energy(phi, mu, sigma, zi, K):
    e = torch.tensor(0.0)
    cov_eps = torch.eye(mu.shape[1]) * (1e-12)

    for k in range(K):
        miu_k = mu[k].unsqueeze(1)
        d_k = zi - miu_k

        inv_cov = torch.inverse(sigma[k] + cov_eps)
        e_k = torch.exp(-0.5 * d_k.T @ inv_cov @ d_k)
        e_k = e_k / torch.sqrt(torch.abs(torch.det(2 * np.pi * sigma[k])))
        e_k = e_k * phi[k]
        e += e_k.squeeze()

    return -torch.log(e)


@pytest.mark.parametrize("N, X, Z, K", dagmm_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_sample_energy(N, X, Z, K, device):
    np.random.seed(0)
    gamma_array = softmax(np.random.randn(N, K), axis=1)
    z_array = np.random.randn(N, Z)

    gamma = ndl.Tensor(gamma_array, device=device)
    z = ndl.Tensor(z_array, device=device)
    phi, mu, sigma = DAGMM.get_gmm_parameters(gamma, z)
    E = DAGMM.get_sample_energy(z, phi, mu, sigma)

    gamma_tensor = torch.Tensor(gamma_array).float()
    z_tensor = torch.Tensor(z_array).float()
    phi_tensor, mu_tensor, sigma_tensor = get_gmm_parameters(gamma_tensor, z_tensor)
    E_tensor = []

    for i in range(z_tensor.shape[0]):
        zi_tensor = z_tensor[i].unsqueeze(1)
        ei_tensor = get_sample_energy(phi_tensor, mu_tensor, sigma_tensor, zi_tensor, K)
        E_tensor.append(ei_tensor)

    E_tensor = torch.stack(E_tensor, dim=0)

    err_E = np.linalg.norm(E_tensor.detach().numpy() - E.numpy())
    assert err_E < 1e-2, "phi match %s, %s" % (E, E_tensor)


@pytest.mark.parametrize("N, X, Z, K", dagmm_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_model_dagmm(N, X, Z, K, device):
    np.random.seed(0)
    model = DAGMM(device=device)

    x_array = np.random.randn(N, X)
    x = ndl.Tensor(x_array, device=device)

    z_c, x_r, z, gamma = model(x)
    assert z_c.shape == (N, 1)
    assert x_r.shape == (N, X)
    assert z.shape == (N, Z)
    assert gamma.shape == (N, K)

    phi, mu, sigma = model.get_gmm_parameters(gamma, z)
    assert phi.shape == (K,)
    assert mu.shape == (K, Z)
    assert sigma.shape == (K, Z, Z)

    E = model.get_sample_energy(z, phi, mu, sigma)
    assert E.shape == (N,)

    loss, loss_items = model.get_loss(x, x_r, z, phi, mu, sigma)
    reconstruction_loss, sample_energy_loss, penalty_loss = loss_items
