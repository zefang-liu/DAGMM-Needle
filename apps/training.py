import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
from models import *
import time
import datetime


### KDD Cup Training ###
def epoch_general_kdd_cup(model, dataloader, opt=None,
                          verbose=True, print_step=500):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss.

    Args:
        model: nn.Module instance
        dataloader: Dataloader instance
        opt: Optimizer instance (optional)
        verbose: print losses (bool)
        print_step: print step (int)

    Returns:
        avg_loss: average loss over dataset
        energies: sample energies
        targets: labels
    """
    np.random.seed(0)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()

    total_num = 0
    total_loss = 0
    energies = []
    targets = []
    start_time = time.time()

    for i_batch, (x, y) in enumerate(dataloader):
        batch_size = x.shape[0]
        total_num += batch_size

        z_c, x_r, z, gamma = model(x)
        phi, mu, sigma = model.get_gmm_parameters(gamma, z)
        energy = model.get_sample_energy(z, phi, mu, sigma)
        loss, _ = model.get_loss(x, x_r, energy, sigma)
        total_loss += loss.numpy() * batch_size
        energies.append(energy.numpy())
        targets.append(y.numpy())

        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()

        if verbose and (i_batch + 1) % print_step == 0:
            elapsed_time = time.time() - start_time
            log_time = str(datetime.timedelta(seconds=round(elapsed_time)))
            log = f'Batch [{i_batch + 1}/{len(dataloader)}]: loss = {total_loss / total_num:.4f}, ' \
                  f'time = {log_time}'

            print(log)

    avg_loss = total_loss / total_num
    energies = np.concatenate(energies, axis=0)
    targets = np.concatenate(targets, axis=0)

    return avg_loss, energies, targets
    ### END YOUR SOLUTION


def train_kdd_cup(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
                  lr=0.001, weight_decay=0.001, verbose=True, print_step=500):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: nn.Module instance
        dataloader: Dataloader instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        verbose: print losses (bool)
        print_step: print step (int)

    Returns:
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(0)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(
        params=model.parameters(), lr=lr,
        weight_decay=weight_decay)
    avg_loss = None
    energies = None
    targets = None
    start_time = time.time()

    for epoch in range(n_epochs):
        avg_loss, energies, targets = epoch_general_kdd_cup(
            model=model, dataloader=dataloader, opt=opt,
            verbose=verbose, print_step=print_step)

        if verbose:
            elapsed_time = time.time() - start_time
            log_time = str(datetime.timedelta(seconds=round(elapsed_time)))
            log = f'Batch [{epoch + 1}/{n_epochs}]: loss = {avg_loss:.4f}, ' \
                  f'time = {log_time}'
            print(log)

    return avg_loss, energies, targets
    ### END YOUR SOLUTION


def evaluate_kdd_cup(model, dataloader, verbose=True, print_step=500):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: nn.Module instance
        dataloader: Dataloader instance
        verbose: print losses (bool)
        print_step: print step (int)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(0)
    ### BEGIN YOUR SOLUTION
    avg_loss, energies, targets = epoch_general_kdd_cup(
        model=model, dataloader=dataloader, opt=None,
        verbose=verbose, print_step=print_step)
    return avg_loss, energies, targets
    ### END YOUR SOLUTION
