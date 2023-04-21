import options
import plots
import denoize
import pathlib as plib
import logging
import numpy as np
import typing
import nibabel as nib


def load_data(path: typing.Union[str, plib.Path], test_debugging_flag: bool = False) -> (np.ndarray, nib.Nifti1Image):
    # cast to path
    if isinstance(path, str):
        path = plib.Path(path).absolute()
    # catch file not found
    if not path.is_file():
        err = f"given File not found: {path.__str__()}"
        logging.error(err)
        raise ValueError(err)
    if not '.nii' in path.suffixes:
        err =f"file no .nii file: {path.__str__()}"
        logging.error(err)
        raise ValueError(err)
    # loading nii image
    logging.info(f"Loading data from file {path.__str__()}")
    niiImg = nib.load(path)
    data = np.array(niiImg.get_fdata())
    # l2 normalize
    logging.info("Normalizing data (l2), assuming t in last dimension")
    norm = np.linalg.norm(data, axis=-1)
    data = np.divide(data, norm, where=norm>1e-9, out=np.zeros_like(data))

    if test_debugging_flag:
        # load only subset of data
        # want data from "middle" of image to not get 0 data for testing
        idx_half = [int(data.shape[k] / 2) for k in range(2)]
        data = data[idx_half[0]:idx_half[0] + 10, idx_half[1]:idx_half[1] + 10]

    return data, niiImg


def mode_denoize(
        fit_opts: options.FitOptions, data_to_fit: np.ndarray, data_img: nib.Nifti1Image) -> np.ndarray:
    # denoizing time series
    logging.info("Denoizing time series")
    if fit_opts.opts.DenoizeSave:
        save_plot_path = plib.Path(fit_opts.config.OutputPath).absolute().joinpath(
            f"{fit_opts.config.NameId}_denoising_efficiency.png"
        ).__str__()
    else:
        save_plot_path = ""

    denoize_algorithm = denoize.MajMinNcChiDenoizer(
        num_cp_runs=fit_opts.opts.DenoizeNumIterations,
        visualize=fit_opts.opts.Visualize,
        mp_headroom=fit_opts.opts.HeadroomMultiprocessing
    )
    denoize_algorithm.get_nc_stats(data=data_to_fit)
    denoize_algorithm.denoize_nii_data(data=data_to_fit)
    d_niiData = denoize_algorithm.denoize_nii_data(
        data=data_to_fit,
        save_plot=save_plot_path
    )

    if fit_opts.opts.DenoizeSave:
        logging.info("Writing denoized to .nii")
        if not fit_opts.config.NameId:
            name = plib.Path(fit_opts.config.NiiDataPath).absolute()
            for _ in name.suffixes:
                name = name.with_suffix("")
            name = name.stem
        else:
            name = fit_opts.config.NameId
        outPath = plib.Path(fit_opts.config.OutputPath).absolute().joinpath(f"d_{name}.nii")
        logging.info(f"Writing File: {outPath}")
        d_nii = nib.Nifti1Image(d_niiData, data_img.affine)
        nib.save(d_nii, outPath)
    return d_niiData


def mode_fit(
        fit_opts: options.FitOptions, data_to_fit: np.ndarray, data_img: nib.Nifti1Image) -> (np.ndarray, np.ndarray):
    pass


def mode_both(
        fit_opts: options.FitOptions, data_to_fit: np.ndarray, data_img: nib.Nifti1Image) -> (np.ndarray, np.ndarray):
    denoized_data = mode_denoize(fit_opts=fit_opts, data_to_fit=data_to_fit, data_img=data_img)
    mode_fit(fit_opts=fit_opts, data_to_fit=denoized_data, data_img=data_img)


def main(fit_opts: options.FitOptions):
    dataPath = plib.Path(fit_opts.config.NiiDataPath).absolute()
    niiData, niiImg = load_data(dataPath, test_debugging_flag=True)

    # set values to range 0 - 1000
    plotData = np.divide(1e3 * niiData, niiData.max())

    if fit_opts.opts.Visualize:
        # plot ortho
        plots.plot_ortho_view(plotData)

    modeOptions = {
        "Denoize": mode_denoize,
        "denoize": mode_denoize,
        "d": mode_denoize,
        "Fit": mode_fit,
        "fit": mode_fit,
        "f": mode_fit,
        "Both": mode_both,
        "both": mode_both,
        "df": mode_both
    }
    if modeOptions.get(fit_opts.opts.Mode) is None:
        err = f"fitting mode not provided choose one of {modeOptions}"
        logging.error(err)
        raise AttributeError(err)
    modeOptions.get(fit_opts.opts.Mode)(fit_opts, niiData, niiImg)


if __name__ == '__main__':
    parser, args = options.createCmdLineParser()
    opts = options.FitOptions.fromCmdLine(args)
    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    try:
        main(opts)
    except AttributeError or AssertionError or ValueError as err:
        logging.error(err)
        parser.print_usage()
