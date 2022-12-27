import sys
sys.path.append('./python')
import numpy as np
import torch
import pytest
import needle as ndl
from apps.models import DAGMM
from scipy.special import softmax
np.random.seed(0)

_ERROR_THRESHOLD = 1e-4
_DEVICES = [ndl.cpu(), pytest.param(
    ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]

dagmm_params = [
    (16, 120, 3, 4),
]


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
    assert err_phi < _ERROR_THRESHOLD, "phi match %s, %s" % (phi, phi_tensor)

    err_mu = np.linalg.norm(mu_tensor.detach().numpy() - mu.numpy())
    assert err_mu < _ERROR_THRESHOLD, "mu match %s, %s" % (mu, phi_tensor)

    err_sigma = np.linalg.norm(sigma_tensor.detach().numpy() - sigma.numpy())
    assert err_sigma < _ERROR_THRESHOLD, "sigma match %s, %s" % (sigma, sigma_tensor)


def get_sample_energy(phi, mu, sigma, zi, K):
    energy = torch.tensor(0.0)

    for k in range(K):
        mu_k = mu[k].unsqueeze(1)
        zi_mean = zi - mu_k
        sigma_inv = torch.inverse(sigma[k])

        energy_k = torch.exp(-0.5 * zi_mean.T @ sigma_inv @ zi_mean)
        energy_k = energy_k / torch.sqrt(torch.det(2 * np.pi * sigma[k]))
        energy_k = energy_k * phi[k]
        energy += energy_k.squeeze()

    return -torch.log(energy)


@pytest.mark.parametrize("N, X, Z, K", dagmm_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_sample_energy(N, X, Z, K, device):
    np.random.seed(0)
    gamma_array = softmax(np.random.randn(N, K), axis=1)
    z_array = np.random.randn(N, Z)

    gamma = ndl.Tensor(gamma_array, device=device)
    z = ndl.Tensor(z_array, device=device)
    phi, mu, sigma = DAGMM.get_gmm_parameters(gamma, z)
    energy = DAGMM.get_sample_energy(z, phi, mu, sigma)

    gamma_tensor = torch.Tensor(gamma_array).float()
    z_tensor = torch.Tensor(z_array).float()
    phi_tensor, mu_tensor, sigma_tensor = get_gmm_parameters(gamma_tensor, z_tensor)
    energy_tensor = []

    for i in range(z_tensor.shape[0]):
        zi_tensor = z_tensor[i].unsqueeze(1)
        ei_tensor = get_sample_energy(phi_tensor, mu_tensor, sigma_tensor, zi_tensor, K)
        energy_tensor.append(ei_tensor)

    energy_tensor = torch.stack(energy_tensor, dim=0)

    err_energy = np.linalg.norm(energy_tensor.detach().numpy() - energy.numpy())
    assert err_energy < _ERROR_THRESHOLD, "energy match %s, %s" % (energy, energy_tensor)


@pytest.mark.parametrize("N, X, Z, K", dagmm_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_model_dagmm(N, X, Z, K, device):
    np.random.seed(0)
    model = DAGMM(x_dim=X, gamma_dim=K, device=device)

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

    energy = model.get_sample_energy(z, phi, mu, sigma)
    assert energy.shape == (N,)

    loss, loss_items = model.get_loss(x, x_r, energy, sigma)
    reconstruction_loss, sample_energy_loss, penalty_loss = loss_items
