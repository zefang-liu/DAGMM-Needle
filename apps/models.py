import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import numpy as np
np.random.seed(0)


class DAGMM(nn.Module):
    def __init__(self, x_dim=118, gamma_dim=4, lambda1=0.1, lambda2=0.005, device=None, dtype="float32"):
        ### BEGIN YOUR SOLUTION
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, 60, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(60, 30, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(30, 10, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(10, 1, device=device, dtype=dtype),
        )

        self.decoder = nn.Sequential(
            nn.Linear(1, 10, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(10, 30, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(30, 60, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(60, x_dim, device=device, dtype=dtype),
        )

        self.estimation = nn.Sequential(
            nn.Linear(3, 10, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(10, gamma_dim, device=device, dtype=dtype),
            nn.Softmax(),
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        N = x.shape[0]
        z_c = self.encoder(x)
        x_r = self.decoder(z_c)

        z_euclidean = ndl.relative_distance(x, x_r, dim=1)
        z_cosine = ndl.cosine_similarity(x, x_r, dim=1)
        z = ndl.stack(
            (z_c.reshape(shape=(N,)), z_euclidean, z_cosine),
            axis=1)

        gamma = self.estimation(z)
        return z_c, x_r, z, gamma
        ### END YOUR SOLUTION

    @staticmethod
    def get_gmm_parameters(gamma, z):
        ### BEGIN YOUR SOLUTION
        N, K = gamma.shape
        _, Z = z.shape

        gamma_sum = gamma.sum(axes=0).reshape(shape=(K, 1))  # (K, 1)
        phi = gamma_sum.squeeze() / N  # (K,)
        mu = gamma.T @ z / gamma_sum.broadcast_to(shape=(K, Z))  # (K, Z)
        z_mean = z.reshape(shape=(N, 1, Z)).broadcast_to(shape=(N, K, Z)) \
            - mu.reshape(shape=(1, K, Z)).broadcast_to(shape=(N, K, Z))  # (N, K, Z)
        z_mean = z_mean.reshape(shape=(N, K, Z, 1))  # (N, K, Z, 1)
        sigma = (
            gamma.reshape(shape=(N, K, 1, 1)).broadcast_to(shape=(N, K, Z, Z))
            * ndl.bmm(z_mean, z_mean.T)
        ).sum(axes=0) / gamma_sum.reshape(
            shape=(K, 1, 1)).broadcast_to(shape=(K, Z, Z))  # (K, Z, Z)

        return phi, mu, sigma
        ### END YOUR SOLUTION

    @staticmethod
    def get_sample_energy(z, phi, mu, sigma):
        ### BEGIN YOUR SOLUTION
        from numpy import pi
        N, Z = z.shape
        K, Z = mu.shape

        z_mean = z.reshape(shape=(N, 1, Z)).broadcast_to(shape=(N, K, Z)) \
            - mu.reshape(shape=(1, K, Z)).broadcast_to(shape=(N, K, Z))  # (N, K, Z)
        z_mean = z_mean.reshape(shape=(N, K, Z, 1))  # (N, K, Z, 1)
        sigma_inv = ndl.inv(sigma)  # (K, Z, Z)
        sigma_inv = sigma_inv.reshape(shape=(1, K, Z, Z)).broadcast_to(
            shape=(N, K, Z, Z))  # (N, K, Z, Z)
        sigma_det = ndl.det(sigma * 2 * pi).reshape(shape=(1, K)).broadcast_to(
            shape=(N, K))  # (N, K)
        phi = phi.reshape(shape=(1, K)).broadcast_to(shape=(N, K))  # (N, K)

        energy = -ndl.log((
            phi * (
                ndl.exp(ndl.bmm(ndl.bmm(z_mean.T, sigma_inv), z_mean).squeeze() * (-0.5))
                / (sigma_det ** 0.5)
            )
        ).sum(axes=1))  # (N,)
        return energy
        ### END YOUR SOLUTION

    @staticmethod
    def get_reconstruction_loss(x, x_r):
        ### BEGIN YOUR SOLUTION
        out = ndl.norm(x - x_r, dim=1)
        return out.sum(axes=0) / out.shape[0]
        ### END YOUR SOLUTION

    @staticmethod
    def get_sample_energy_loss(E):
        ### BEGIN YOUR SOLUTION
        return E.sum(axes=0) / E.shape[0]
        ### END YOUR SOLUTION

    @staticmethod
    def get_penalty_loss(sigma):
        ### BEGIN YOUR SOLUTION
        return ndl.diagonal(sigma ** (-1)).sum()
        ### END YOUR SOLUTION

    def get_loss(self, x, x_r, energy, sigma):
        ### BEGIN YOUR SOLUTION
        reconstruction_loss = self.get_reconstruction_loss(x, x_r)
        sample_energy_loss = self.get_sample_energy_loss(energy)
        penalty_loss = self.get_penalty_loss(sigma)
        loss = reconstruction_loss + self.lambda1 * sample_energy_loss \
            + self.lambda2 * penalty_loss
        return loss, (reconstruction_loss, sample_energy_loss, penalty_loss)
        ### END YOUR SOLUTION
