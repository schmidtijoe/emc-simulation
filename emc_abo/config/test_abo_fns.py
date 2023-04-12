import numpy as np
import itertools
import timeit
import matplotlib.pyplot as plt


def calc_l2_diff_loss_mat(signal_curves: np.ndarray = np.random.random(size=(20, 10))):
    l2_matrix = signal_curves[:, np.newaxis, :] - signal_curves
    l2_matrix = 1.0 - np.linalg.norm(l2_matrix, axis=-1)
    return l2_matrix


def calc_einsum_dist(signal_curves: np.ndarray = np.random.random(size=(20, 10))):
    # want to calculate euclidean dist (l2)
    sources = signal_curves[:, np.newaxis, :] - signal_curves
    l2_matrix = np.einsum('ijk, ijk -> ij', sources, sources)
    return 1.0 - np.sqrt(l2_matrix)


def calc_l2_diff_loss_iter(signal_curves: np.ndarray = np.random.random(size=(20, 10))):
    dim = signal_curves.shape[0]
    l2_matrix = np.zeros((dim, dim))
    for i, j in itertools.product(np.arange(dim), np.arange(dim)):
        l2_matrix[i, j] = 1 - np.linalg.norm(signal_curves[i] - signal_curves[j], axis=-1)
    return l2_matrix


def calc_dot_loss(signal_curves: np.ndarray = np.random.random(size=(20, 10))):
    # need normalized curves
    dim = signal_curves.shape[0]
    dot_matrix = np.zeros((dim, dim))
    for i, j in itertools.product(np.arange(dim), np.arange(dim)):
        dot_matrix[i, j] = np.dot(signal_curves[i], signal_curves[j])
    return dot_matrix


def calc_einsum_loss(signal_curves: np.ndarray = np.random.random(size=(20, 10))):
    # need normalised curves
    en_matrix = np.einsum('ik, jk -> ij', signal_curves, signal_curves)
    return en_matrix


def calc_pearson_loss(signal_curves: np.ndarray = np.random.random(size=(20, 10))):
    corr_matrix = np.corrcoef(signal_curves)
    # set diagonal and upper half 0, square
    obj_matrix = corr_matrix
    # additionally we are not interested in the correlation within a t2 value, between different b1 effectivities
    return obj_matrix


if __name__ == '__main__':
    test_curves = np.square(np.random.random(size=(20, 10)))
    norm = np.linalg.norm(test_curves, axis=-1, keepdims=True)
    test_curves = np.divide(test_curves, norm, where=norm > 1e-12, out=np.zeros_like(test_curves))

    l2_m = calc_l2_diff_loss_mat
    dot_eins = calc_einsum_dist
    dot_iter = calc_dot_loss
    pears = calc_pearson_loss
    eins = calc_einsum_loss

    test_fns = [l2_m, dot_iter, dot_eins, pears, eins]
    test_labels = ['l2 matrix', 'dot iter', 'l2 dist einsum', 'pearson corr', 'einsum func']
    test_num = 5

    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(3, 2)
    for k in range(test_num):
        test_func = test_fns[k]
        result = timeit.timeit(stmt='test_func()', number=5000, setup="from __main__ import test_func")

        ax = fig.add_subplot(gs[k])
        ax.axis(False)
        ax.set_title(f"{test_labels[k]} - time: {result:.4f} s")
        ax.imshow(test_func(test_curves))

        min = np.unravel_index(np.argmin(test_func(test_curves)), shape=test_func(test_curves).shape)
        ax.scatter(*min, color='r', label=f"found minimum at {min}")
        ax.legend()
    plt.savefig("testing_function_execution_speed_2.png", bbox_inches='tight')
    plt.show()



