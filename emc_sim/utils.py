import json
import logging
logModule = logging.getLogger(__name__)
import numpy as np
import os
from pathlib import Path
import pandas as pd
import nibabel as nib
import pickle
import types


def create_folder_ifn_exist(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def normalize_array(data_array: np.ndarray, max_factor: float = 1.0,
                    normalization: str = "max", by_value: float = None) -> np.ndarray:
    norm = {
        "max": np.max(data_array, keepdims=True, axis=-1),
        "l2": np.linalg.norm(data_array, keepdims=True, axis=-1),
        "sum": np.sum(data_array, axis=-1, keepdims=True),
        "by_value": np.array([by_value])
    }
    if not type(norm.get(normalization)) == np.ndarray:
        raise ValueError(f"normalization type not recognized, got {normalization}\n"
                         f"choose one of the following: \n"
                         f"\tmax \n"
                         f"\tl2 \n"
                         f"\tsum \n"
                         f"\tby_value")
    data_norm = norm.get(normalization)
    data_array = np.divide(data_array, data_norm, where=data_norm != 0, out=np.zeros_like(data_array))
    return max_factor * data_array


def load_database(path_to_file, append_zero: bool = True, normalization: str = "l2") -> (pd.DataFrame, np.ndarray):
    # need standardized way of saving the database here
    path = Path(path_to_file).absolute()
    assert path.is_file()
    load_fn = {
        ".pkl": types.SimpleNamespace(function=pickle.load, context="rb"),
        ".json": types.SimpleNamespace(function=json.load, context="r")
    }
    loader = load_fn.get(path.suffix)
    # throws error if None -> None if key not in dict
    assert loader, f"database filetype not supported !\n" \
                   f"got: {path.suffix}; supported: {list(load_fn.keys())}"
    with open(path, loader.context) as FILE:
        sim_dict = loader.function(FILE)

    df = pd.DataFrame(sim_dict)
    len_b1s = len(df.b1.unique().astype(float))
    len_t1s = len(df.t1.unique().astype(float))
    len_ds = len(df.d.unique().astype(float))
    dim_expand_needed = len_ds * len_b1s * len_t1s
    if append_zero:
        for _ in range(dim_expand_needed):
            temp_row = df.iloc[0].copy()
            temp_row.emcSignal = np.zeros([len(temp_row.emcSignal)])
            temp_row.t2 = 0.0
            temp_row.b1 = 1.0
            temp_row.t1 = 1.5
            df.loc[len(df.index)] = temp_row
        df = df.reset_index(drop=True)

    sim_data_flat = np.array([*df.emcSignal.to_numpy()])
    sim_data_flat = normalize_array(sim_data_flat, normalization=normalization)
    return df, sim_data_flat


def niiDataLoader(path_to_nii_data: str, test_set: bool = False, normalize: str = "max") -> (
        np.ndarray, nib.nifti1.Nifti1Image):
    """
    Loads nii data into numpy array. and reshapes to 2d, normalizes
    :param normalize: kind of normalization (area, max)
    :param path_to_nii_data:
    :param test_set: assumes [x,y,z (,t] data and picks 10x10 subset of x and y
    :return: numpy array of nii data, nib img of data
    """
    path = Path(path_to_nii_data).absolute()
    if ".nii" in path.suffixes:
        # also works for .nii.gz -> path suffixes include all
        niiImg = nib.load(path)
        data = np.array(niiImg.get_fdata())
        if normalize == "max":
            data = normalize_array(data_array=data, normalization=normalize)
        if test_set:
            # want data from "middle" of image to not get 0 data for testing
            idx_half = [int(data.shape[k] / 2) for k in range(2)]
            data = data[idx_half[0]:idx_half[0] + 10, idx_half[1]:idx_half[1] + 10]
        return data, niiImg
    logModule.error(f"input file {path}: type not recognized or no .nii file")
    raise AttributeError(f"input file {path}: type not recognized or no .nii file")
