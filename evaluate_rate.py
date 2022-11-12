import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.linalg import expm


def generate_p(R, ns):
    """
    Parameters
    ----------
    R : np.array s.t. P(n)=exp(nR), shape=(4,)
    ns : np.array

    Returns
    -------
    pd.DataFrame, P(n) for each n in ns.
    """
    max_ns = ns.max()
    Ps = pd.DataFrame(np.array([expm(n / max_ns * R).reshape(4) for n in ns]))
    Ps.insert(loc=0, column='dist', value=ns)
    return Ps


def plot_prog_p(emp_Ps, est_Ps, ns, sim_name=None):
    """
    Plots graphs of P(n)_ij vs n, for P(n)=exp(nR), and scatter of empirical P(n).
    Parameters
    ----------
    emp_Ps : np.array, empirical P(n) for each n in ns.
    est_Ps : np.array, s.t. P(n)=exp(nR) for the estimated R
    ns : np.array
    sim_name : string, png name
    """
    clr = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]
    for i in range(4):
        if i in [0, 2]:
            plt.plot(ns, est_Ps[:, i], label='est,p:M->M' if i==0 else 'est,q:U->U', c=clr[i])
            plt.scatter(ns, emp_Ps[:, i], label='emp,p:M->M' if i==0 else 'emp,q:U->U', c=clr[i])
        else: plt.plot(ns, est_Ps[:, i], c=clr[i])

    plt.title(f'Progress of P(n) vs n, {sim_name}')
    plt.xlabel('n')
    plt.ylabel('P(n)')
    # plt.xticks(ns, ns)
    plt.ylim((0, 1))
    plt.legend()
    if sim_name:
        plt.savefig(f'plots/simulations/{sim_name}.png')
    else:
        plt.show()
    plt.clf()


if __name__ == '__main__':
    from learn_rate import fit_R
    get_R = lambda a, b: np.array([[-a, a], [b, -b]])
    a_s = [1]
    b_s = [.1, .25, .5, 1, 2, 5, 10]
    for a in a_s:
        for b in b_s:
            # generate data
            ns = np.arange(1, 10, 1)
            emp_Ps = generate_p(get_R(a, b), ns)
            # fit R
            R = np.asarray(fit_R(emp_Ps, n_iters=100))
            R = np.array([[R[0], -R[0]], [-R[1], R[1]]])
            print(R)
            # R = get_R(1, 2)
            est_Ps = generate_p(R, ns)
            plot_prog_p(emp_Ps.values[:, 1:], est_Ps.values[:, 1:], ns, f'a={a}, b={b}')
    c=0