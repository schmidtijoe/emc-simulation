import numpy as np
import logging

logModule = logging.getLogger(__name__)


def fit_data(data: np.ndarray, te: np.ndarray, quad: bool = True) -> (np.ndarray, np.ndarray):
    eps = 1e-7
    # get data shape
    shape = data.shape
    # get data into m X ETL format assuming t in last dimension
    data = np.reshape(data, [-1, te.shape[0]])
    # log
    if quad:
        y = np.square(data)
    else:
        y = data
    y_obs = np.log(np.clip(y, 1, np.max(y)))

    # build parameter matrix
    A = np.ones([te.shape[0], 2])
    A[:, 1] = te

    B = np.dot(np.transpose(A), A)
    B = np.linalg.inv(B)

    params = np.dot(np.dot(B, np.transpose(A)), np.transpose(y_obs))

    s = np.exp(np.reshape(params[0, :], shape[:-1]))
    r2 = - np.reshape(params[1, :], shape[:-1])
    if quad:
        r2 /= 2
    r2 = np.clip(r2, 0, np.max(r2))
    t2 = np.divide(1e3, r2, where=r2 > eps, out=np.zeros_like(r2))
    t2 = np.clip(t2, 0, 1e3)
    return t2, s
