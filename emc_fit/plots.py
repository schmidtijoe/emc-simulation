import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging

# set logger
logModule = logging.getLogger(__name__)


def set_mpl(n: int) -> np.ndarray:
    # set mpl
    plt.style.use("ggplot")
    colors = cm.viridis(np.linspace(0.05, 0.95, n))
    return colors


def plot_curve_selection(data: np.ndarray, noise_mean: float):
    n = 100
    colors = set_mpl(n)
    int_size = data.size / data.shape[-1]
    curves = np.reshape(data, (-1, data.shape[-1]))[np.random.randint(int_size, size=n)]
    x_ax = np.arange(1, curves.shape[-1] + 1)
    # plot selection
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot()
    for k in range(n):
        ax.plot(x_ax, curves[k], color=colors[k])
    ax.plot(x_ax, np.full_like(x_ax, noise_mean), color="#ff5c33", linewidth=3, label="noise mean")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_ortho_view(data: np.ndarray):
    logModule.info("plot ortho view")
    echo = 1
    echoImg = data[:, :, :, echo]
    shape = np.array([*echoImg.shape]) / 2
    x, y, z = shape.astype(int)

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 3, width_ratios=[18, 10, 1])

    ax_axial = fig.add_subplot(gs[:, 0])
    ax_axial.axis(False)
    img = ax_axial.imshow(echoImg[:, :, z], extent=[0, echoImg.shape[0], 0, echoImg.shape[1]])

    ax_coronal = fig.add_subplot(gs[0, 1])
    ax_coronal.axis(False)
    ax_coronal.imshow(echoImg[:, y, :].T, extent=[0, echoImg.shape[0], 0, echoImg.shape[2]])

    ax_saggital = fig.add_subplot(gs[1, 1])
    ax_saggital.axis(False)
    ax_saggital.imshow(echoImg[x, :, :].T, extent=[0, echoImg.shape[1], 0, echoImg.shape[2]])

    ax_cb = fig.add_subplot(gs[:, 2])
    ax_cb.grid(False)
    ax_cb.set_ylabel("Intensity")

    fig.colorbar(img, cax=ax_cb)
    plt.tight_layout()
    plt.show()


def plot_denoized(origData: np.ndarray, denoizedData: np.ndarray):
    # build data histogramm
    data_hist, data_bins = np.histogram(origData, bins=1000)
    data_bins = data_bins[1:] - np.diff(data_bins)

    y_lim = np.max(origData) * 0.7
    x_hist, x_bins = np.histogram(denoizedData, bins=1000)
    x_bins = x_bins[1:] - np.diff(x_bins)

    z = int(origData.shape[2] / 2)
    if origData.shape.__len__() > 3:
        origData = origData[:, :, :, 1]
        denoizedData = denoizedData[:, :, :, 1]

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(2, 6, width_ratios=[8, 1, 8, 1, 8, 1], height_ratios=[3, 1])

    ax = fig.add_subplot(gs[0, 0])
    ax.axis(False)
    ax.grid(False)
    img = ax.imshow(origData[:, :, z], clim=(0, y_lim))
    ax = fig.add_subplot(gs[0, 1])
    plt.colorbar(img, ax=ax)

    ax = fig.add_subplot(gs[0, 2])
    ax.axis(False)
    ax.grid(False)
    img = ax.imshow(denoizedData[:, :, z], clim=(0, y_lim))
    ax = fig.add_subplot(gs[0, 3])
    plt.colorbar(img, ax=ax)

    ax = fig.add_subplot(gs[0, 4])
    ax.axis(False)
    ax.grid(False)
    img = ax.imshow((origData[:, :, z] - denoizedData[:, :, z]) / origData[:, :, z])
    ax = fig.add_subplot(gs[0, 5])
    plt.colorbar(img, ax=ax)

    ax = fig.add_subplot(gs[1, :2])
    ax.set_ylim(0, 1.2 * np.max(data_hist[100:]))
    ax.fill_between(data_bins, data_hist)

    ax = fig.add_subplot(gs[1, 2:4])
    ax.fill_between(x_bins, x_hist)
    ax.set_ylim(0, 1.2 * np.max(x_hist[20:]))

    plt.tight_layout()
    plt.show()
