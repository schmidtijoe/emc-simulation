import json
import logging
import numpy as np
from pathlib import Path
import pandas as pd
import nibabel as nib
import pickle
import types
import typing
import tqdm

logModule = logging.getLogger(__name__)


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


def niiDataLoader(
        path_to_nii_data: typing.Union[str, Path],
        test_set: bool = False, normalize: str = "") -> (
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
        if normalize:
            data = normalize_array(data_array=data, normalization=normalize)
        if test_set:
            # want data from "middle" of image to not get 0 data for testing
            idx_half = [int(data.shape[k] / 2) for k in range(2)]
            data = data[idx_half[0]:idx_half[0] + 10, idx_half[1]:idx_half[1] + 10]
        return data, niiImg
    logModule.error(f"input file {path}: type not recognized or no .nii file")
    raise AttributeError(f"input file {path}: type not recognized or no .nii file")


def load_database(path_to_file: typing.Union[str, Path], append_zero: bool = True, normalization: str = "l2") -> (
        pd.DataFrame, np.ndarray):
    # need standardized way of saving the database: changes here need changes in save fn above
    if not isinstance(path_to_file, Path):
        path = Path(path_to_file).absolute()
    else:
        path = path_to_file.absolute()
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
    if append_zero:
        b1s = df.b1.unique().astype(float)
        t1s = df.t1.unique().astype(float)
        ds = df.d.unique().astype(float)
        for b1, t1, d in [(b1_val, t1_val, d_val) for b1_val in b1s for t1_val in t1s for d_val in ds]:
            # when normalizing 0 curves will be left unchanged. Data curves are unlikely 0
            temp_row = df.iloc[0].copy()
            temp_row.emcSignal = np.zeros([len(temp_row.emcSignal)])
            temp_row.t2 = 1e-3
            temp_row.b1 = b1
            temp_row.t1 = t1
            temp_row.d = d
            df.loc[len(df.index)] = temp_row
            # still append 0 curves that wont get scaled -> more useful for the pearson fitting
            temp_row = df.iloc[0].copy()
            temp_row.emcSignal = np.zeros([len(temp_row.emcSignal)])
            temp_row.t2 = 0.0
            temp_row.b1 = b1
            temp_row.t1 = t1
            temp_row.d = d
            df.loc[len(df.index)] = temp_row
        df = df.reset_index(drop=True)

    sim_data_flat = np.array([*df.emcSignal.to_numpy()])
    sim_data_flat = normalize_array(sim_data_flat, normalization=normalization)
    return df, sim_data_flat


def parse_dcm_to_pro(path_to_dcm: typing.Union[str, Path]):
    path = Path(path_to_dcm).absolute()
    suffixes = ['.dcm', '.ima']
    if set(suffixes).isdisjoint(path.suffixes):
        err = f"File {path} has unknown Type {path.suffix}; must be one of: {suffixes}"
        logModule.error(err)
        raise AttributeError(err)

    with open(path, "rb") as r_file:
        dcm = r_file.read().splitlines()

    # Find beginning and end of important info
    l_counter = -1
    start = -1
    end = -1
    lookout = False
    for line in dcm:
        l_counter += 1
        if line.find(b'<ParamString."Protocol0">') != -1:
            lookout = True
        if line.find(b'XProtocol') != -1 and lookout:
            start = l_counter
            logModule.info(f"found start: {start}")
            lookout = False
        if line.startswith(b'### ASCCONV END ###'):
            end = l_counter + 1
            logModule.info(f"found end: {end}")

    # some manipulations to transform to .pro
    data_lines = dcm[start:end]

    # define whats to be changed
    subs = types.SimpleNamespace(line_idx=-1, idx=-1, str='')
    subs_dict = {
        'Initialized by sequence': types.SimpleNamespace(line_idx=1, idx=-21, str=''),
        'ParamString."PatPosition': types.SimpleNamespace(line_idx=0, idx=0, str='<ParamString."PatPosition">  { }'),
        'ParamLong."SBCSOriginPositionZ': types.SimpleNamespace(line_idx=0, idx=0,
                                                                str='<ParamLong."SBCSOriginPositionZ">  { }'),
        'tSequenceFileName': types.SimpleNamespace(line_idx=0, idx=0,
                                                   str='tSequenceFileName	 = 	"Y:\n4\x86\prod\bin\se_mc"'),
        'tProtocolName': types.SimpleNamespace(line_idx=0, idx=0, str='tProtocolName	 = 	"Initialized by sequence"')
    }

    data = []
    bar = tqdm.trange(len(data_lines))
    for line_idx in bar:
        # pick line
        line = data_lines[line_idx].decode()
        # removing leasing/ trailing or double quotes ""
        if line.strip().startswith('"'):
            line = line.replace('"', '', 1)
        if line_idx == len(data_lines) - 1:
            line = line.replace(line[line.rfind('"')], '')
        line = line.replace('""', '"')

        # see if line contaings one of those
        for key in subs_dict.keys():
            idx = line.find(key)
            if idx != -1:
                bar.set_postfix_str(f"Found: {key} at {line_idx}")
                subs = subs_dict.get(key)
                subs.idx += idx
                subs.line_idx += line_idx

        if line_idx == subs.line_idx:
            bar.set_postfix_str(f"Changed: {line} to {subs.str}")
            line = line.replace(line[subs.idx:], subs.str)

        data.append(line.encode())

    # save
    path = path.with_suffix('.pro')
    logModule.info(f"Saving File: {path}")
    with open(path, "wb") as s_file:
        s_file.write(b'\n'.join(data_lines))


if __name__ == '__main__':
    PATH = "D:\\Daten\\01_Work\\11_owncloud\\ds_mese_cbs_js\\02_postmortem_scan_data\\01\\t2_semc_0p8_93slice\\dcm.dcm"
    parse_dcm_to_pro(PATH)
