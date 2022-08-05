import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from natsort import natsorted
import nibabel as nib
from emc_fit.noise.distributions import NcChi

logModule = logging.getLogger(__name__)


def extract_chi_noise_characteristics_from_nii(niiData: np.ndarray,
                                               corner_fraction: float = 8.0,
                                               visualize: bool = True,
                                               mask: str = "",
                                               cmap: str = "gray"):
    """
    Input slice dim or 3d nii file with sample free corners, aka noise
    dimensions assumed: [x, y, z]

    :param mask: filename for custom mask file, provided as .nii or .npy
    :param visualize: show plots of masking and extraction
    :param cmap: colormap in case of visualize is true
    :param corner_fraction: fraction of the x/y dimension corners to use for noise handling
    :param niiData: path to nii file
    :return:
    """
    shape = niiData.shape
    # init mask
    mask_array = np.zeros_like(niiData, dtype=bool)
    # check for mask input
    if mask:
        mask_path = Path(mask).absolute()
        if mask_path.suffix == ".nii":
            mask_array = nib.load(mask_path).get_fdata()
        elif mask_path.suffix == ".npy":
            mask_array = np.load(str(mask_path))
        else:
            logModule.error("mask file ending not recognized: "
                            "give file as .nii or .npy")
            exit(-1)
    else:
        corner_idx = int(mask_array.shape[0] / corner_fraction)
        mask_array[:corner_idx, :corner_idx] = True
        mask_array[:corner_idx, -corner_idx:] = True
        mask_array[-corner_idx:, :corner_idx] = True
        mask_array[-corner_idx:, -corner_idx:] = True
    # cast shapes to 3D
    if mask_array.shape.__len__() < 3:
        mask_array = mask_array[:, :, np.newaxis]
    if mask_array.shape.__len__() > 3:
        mask_array = np.reshape(mask_array, [*mask_array.shape[:2], -1])
    if shape.__len__() < 3:
        if shape.__len__() < 2:
            logModule.error("input data dimension < 2, input at least data slice nii")
            exit(-1)
        # input z dimension
        niiData = niiData[:, :, np.newaxis]
    if shape.__len__() > 3:
        # flatten time dimension onto z
        niiData = np.reshape(niiData, [*shape[:2], -1])

    # pick noise data from .nii, cast to 1D
    noiseArray = niiData[mask_array].flatten()
    noiseArray = noiseArray[noiseArray > 0]

    # init distribution class object
    dist_ncChi = NcChi()
    # emc_fit -> updates channels and sigma of ncchi object
    dist_ncChi.fit_noise(noiseArray)

    if visualize:
        # build histogramm
        maxhist = int(noiseArray.max())
        hist, h_bins = np.histogram(noiseArray, bins=maxhist, range=(0, maxhist), density=True)
        h_bins = (h_bins[1] - h_bins[0]) / 2 + h_bins[:-1]
        hist[0] = 0
        chi_fit = dist_ncChi.pdf(h_bins, 0)
        # plot
        fig = plt.figure(figsize=(15, 7))
        gs = fig.add_gridspec(2, 2)
        ax = fig.add_subplot(gs[0])
        ax.imshow(niiData[:, :, 0], clim=(0, 3 / 4 * np.max(niiData)), cmap=cmap)
        ax.axis('off')
        ax.grid(False)

        ax = fig.add_subplot(gs[1])
        ax.imshow(mask_array[:, :, 0], clim=(0, 1), cmap=cmap)
        ax.axis('off')
        ax.grid(False)

        ax = fig.add_subplot(gs[2:])
        ax.plot(h_bins, hist, label="data")
        ax.plot(h_bins, chi_fit, label="chi_fit")

        ax.legend()
        plt.show()
    logModule.info(f"found distribution characteristics")
    dist_ncChi.get_stats()
    # reshape to original
    if shape.__len__() > 3:
        # if 4d take maximum along time dimension, assumed to be the last
        data_max = np.max(niiData.reshape(shape), axis=-1, keepdims=True)
        # collapse single axis
        if data_max.shape.__len__() > 4:
            data_max = data_max[:, :, :, :, 0]
        else:
            data_max = data_max[:, :, :, 0]
    else:
        # if 3d we take the total data maximum for snr mapping
        data_max = np.max(niiData, keepdims=True)
    snr_map = np.divide(data_max, dist_ncChi.mean(0))
    return dist_ncChi, snr_map


if __name__ == "__main__":
    path = Path("./").absolute().parent
    inputFolder = './data/test_subj_data'
    path = path.joinpath(inputFolder)
    files = natsorted([path.parent.joinpath(f) for f in list(path.iterdir()) if f.suffix == ".nii"])
    data = np.array([nib.load(f).get_fdata() for f in files])
    data = np.moveaxis(data, 0, -1)

    # init mask
    mask_img = np.zeros_like(data, dtype=bool)
    cornerIdx = int(mask_img.shape[0] / 8)
    mask_img[:cornerIdx, -cornerIdx:, :, :] = True
    mask_img[-cornerIdx:, -cornerIdx:, :, :] = True
    np.save("temp_mask_file.npy", mask_img)

    ncChi, snrData = extract_chi_noise_characteristics_from_nii(data, mask="temp_mask_file.npy")
    zero_mean = ncChi.mean(0)
