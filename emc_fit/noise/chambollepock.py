import logging
import numpy as np

logModule = logging.getLogger(__name__)


def chambolle_pock_tv(data, Lambda, n_it=100, return_all=False):
    """
    Chambolle-Pock algorithm for Total Variation regularization.
    The following objective function is minimized :
        ||K*x - d||_2^2 + Lambda*TV(x)

    Adapted from Pierre Paleo: https://github.com/pierrepaleo/spire
    Take operators K(x) as identity
    Lambda : weight of the TV penalization (the higher Lambda, the more sparse is the solution)
    L : norm of the operator [P, Lambda*grad] = 3.6 for identity
    n_it : number of iterations
    return_all: if True, an array containing the values of the objective function will be returned

    """

    sigma = 1.0 / 3.6
    tau = 1.0 / 3.6
    en = np.zeros(n_it)

    x = np.zeros_like(data)
    p = np.zeros_like(np.array(np.gradient(x)))
    q = np.zeros_like(data)
    x_tilde = np.zeros_like(x)

    for k in range(0, n_it):
        # Update dual variables
        # For anisotropic TV, the prox is a projection onto the L2 unit ball.
        # For anisotropic TV, this is a projection onto the L-infinity unit ball.
        arg = p + sigma * np.array(np.gradient(x_tilde))
        p = np.minimum(np.abs(arg), Lambda) * np.sign(arg)
        q = (q + sigma * x_tilde - sigma * data) / (1.0 + sigma)
        # Update primal variables
        x_old = x
        x = x + tau * np.sum(p, axis=0) - tau * q
        x_tilde = x + (x - x_old)
        # Calculate norms
        if return_all:
            fidelity = 0.5 * np.linalg.norm(x - data)
            tv = np.sum(np.abs(np.array(np.gradient(x))))
            energy = 1.0 * fidelity + Lambda * tv
            en[k] = energy
            if k % 10 == 0:
                logModule.info(f"{k} : energy {energy} \t fidelity {fidelity} \t TV {float(tv):.3f}")
    # constrain to >= 0
    x = np.clip(x, 0, np.max(x))
    if return_all:
        return en, x
    else:
        return x

