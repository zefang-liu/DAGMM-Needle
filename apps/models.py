import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import numpy as np
np.random.seed(0)


class DAGMM(nn.Module):
    def __init__(self, lambda1=0.1, lambda2=0.005, device=None, dtype="float32"):
        ### BEGIN YOUR SOLUTION
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.encoder = nn.Sequential(
            nn.Linear(120, 60, device=device, dtype=dtype),
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
            nn.Linear(60, 120, device=device, dtype=dtype),
        )

        self.estimation = nn.Sequential(
            nn.Linear(3, 10, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(10, 4, device=device, dtype=dtype),
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
        phi = gamma_sum / N  # (K, 1)
        mu = gamma.transpose() @ z / gamma_sum.broadcast_to(shape=(K, Z))  # (K, Z)
        z_mean = z.reshape(shape=(N, 1, Z)).broadcast_to(shape=(N, K, Z)) \
            - mu.reshape(shape=(1, K, Z)).broadcast_to(shape=(N, K, Z))  # (N, K, Z)
        Sigma = (
            gamma.reshape(shape=(N, K, 1, 1)).broadcast_to(shape=(N, K, Z, Z))
            * z_mean.reshape(shape=(N, K, Z, 1)).broadcast_to(shape=(N, K, Z, Z))
            * z_mean.reshape(shape=(N, K, 1, Z)).broadcast_to(shape=(N, K, Z, Z))
        ).sum(axes=0) / gamma_sum.reshape(shape=(K, 1, 1)).broadcast_to(shape=(K, Z, Z))  # (K, Z, Z)

        return phi, mu, Sigma
        ### END YOUR SOLUTION

    @staticmethod
    def get_sample_energy():
        ### BEGIN YOUR SOLUTION
        pass
        ### END YOUR SOLUTION

    @staticmethod
    def get_reconstruction_loss(x, x_r):
        ### BEGIN YOUR SOLUTION
        out = ndl.norm(x, x_r)
        return ndl.summation(out) / out.shape[0]
        ### END YOUR SOLUTION

    @staticmethod
    def get_sample_energy_loss():
        ### BEGIN YOUR SOLUTION
        pass
        ### END YOUR SOLUTION

    @staticmethod
    def get_penalty_loss():
        ### BEGIN YOUR SOLUTION
        pass
        ### END YOUR SOLUTION

    @staticmethod
    def get_loss():
        ### BEGIN YOUR SOLUTION
        pass
        ### END YOUR SOLUTION
