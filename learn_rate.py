import pandas as pd
import pickle
import numpy as np
import jax
from jax import value_and_grad
from jax.experimental.optimizers import adam, sgd
import jax.numpy as jnp
from jax.ops import index, index_add, index_update

from utils import read_count_matrix, count_matrix_to_probs, DIST_COL
from evaluate_rate import plot_prog_p

LL='ll'
L2='l2'

def fit_R(count_matrix: pd.DataFrame, init_R=None, n_iters=100, thresh=1e-5, lr=0.1, method=LL, return_progress=False):
    """
    Given count matrix fit a rate matrix according to
    countinous time markov model.

    Parameters
    ----------
    count_matrix : pd.DataFrame 
    init_R : np.ndarray, optional
        If None init R randomly, otherwise use this.
    n_iters : int, optional
        # of optimization iterations, by default 100
    thresh : [type], optional
        [description], by default 1e-5
    """
    prog_dict = {}
    if init_R:
        r = init_R
    else:
        r = np.random.uniform(-1., 0, (2,))
    opt_init, opt_update, get_params = adam(lr)
    if method == LL:
        grad_fn = value_and_grad(log_likelihood_loss, argnums=0)
    else:
        grad_fn = value_and_grad(l2_loss, argnums=0)
    probs_mat = count_matrix_to_probs(count_matrix)
    opt_state = opt_init(r)
    for i in range(n_iters):
        # loss, grads = grad_fn(get_params(opt_state), probs_mat)
        loss, grads = grad_fn(get_params(opt_state), count_matrix)
        prog_dict[i] = (get_params(opt_state), loss)
        print(f"iter {i}: {loss:.4f}, R = {get_params(opt_state)}")
        opt_state = opt_update(i, grads, opt_state) 
    prog_dict[DIST_COL] = probs_mat[DIST_COL].to_numpy()
    if return_progress:
        return prog_dict
    return get_params(opt_state)

def log_likelihood_loss(r, probs_mat: pd.DataFrame, eps=1e-8):
    """
    Computes the negative log-likelihood loss.

    Parameters
    ----------
    r : jax array
    probs_mat : pd.DataFrame
    """
    predicted_count = r_to_predicted_count(r, probs_mat[DIST_COL])
    probs_emp = probs_mat.iloc[:, [1,2,3,4]].to_numpy()
    ll = jnp.multiply(probs_emp, jnp.log2(predicted_count))
    return -(jnp.sum(ll) / probs_emp.sum())

def l2_loss(r, probs_mat) -> float:
    """
    L2 loss. Compute predicted count matrix for r
    and then apply l2 norm.

    Parameters
    ----------
    r : jax array
        2D array. First coordinate is MM (M->M), second coordinate is UU (U->U).
        The other two cells are -r.
    probs_mat : pd.DataFrame
    """
    predicted_count = r_to_predicted_count(r, probs_mat[DIST_COL])
    return jnp.mean((probs_mat.drop([DIST_COL], axis=1).to_numpy() - predicted_count) ** 2)

def r_to_predicted_count(r, dists) -> jnp.ndarray:
    """
    Compute P(n) = e^(r*n)
    for all n in dists.


    Parameters
    ----------
    r : jax array
    dists : pd.Series
    """
    R = jnp.array([r[0], -r[0],-r[1], r[1]])
    max_d = dists.max()
    r_per_dist = jnp.outer(dists.to_numpy() / max_d, R)
    predicted = jnp.zeros_like(r_per_dist)
    for i in range(r_per_dist.shape[0]):
        predicted = index_update(predicted, index[i], jax.scipy.linalg.expm(r_per_dist[i].reshape(2,2)).reshape(-1))
    return predicted


if __name__ == '__main__':
    sample = './data/full.samples/Heart-Cardiomyocyte-Z0000044G.2CpGs.pat.gz_count_matrix.tsv'
    count_mat = read_count_matrix(sample)
    probs = count_matrix_to_probs(count_mat)
    dists = count_mat[DIST_COL]
    r = fit_R(count_mat, n_iters=10)
    plot_prog_p(probs.drop(DIST_COL, axis=1).to_numpy(), r_to_predicted_count(r, dists), dists, 'real_deal')
    # with open(sample + '.r_vals.pkl', 'wb') as f:
    #     pickle.dump(r, f, pickle.HIGHEST_PROTOCOL)
    # predicted_probs_mat = r_to_predicted_count(r, dists)
    # with open(sample + '.r_probs.pkl', 'wb') as f:
    #     pickle.dump(predicted_probs_mat, f, pickle.HIGHEST_PROTOCOL)