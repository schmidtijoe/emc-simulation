from emc_fit import options, plots, denoize, fit
import pathlib as plib
import logging
import numpy as np
import typing
import nibabel as nib
import emc_db


def load_data(path: typing.Union[str, plib.Path], test_debugging_flag: bool = False) -> (np.ndarray, nib.Nifti1Image):
    # cast to path
    if isinstance(path, str):
        path = plib.Path(path).absolute()
    # catch file not found
    if not path.is_file():
        err = f"given File not found: {path.__str__()}"
        logging.error(err)
        raise ValueError(err)
    if '.nii' not in path.suffixes:
        err = f"file no .nii file: {path.__str__()}"
        logging.error(err)
        raise ValueError(err)
    # loading nii image
    logging.info(f"Loading data from file {path.__str__()}")
    niiImg = nib.load(path)
    data = np.array(niiImg.get_fdata())

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
        max_num_runs=fit_opts.opts.DenoizeMaxNumIterations,
        visualize=fit_opts.opts.Visualize,
        mp_headroom=fit_opts.config.HeadroomMultiprocessing,
        single_iteration=True
    )
    d_niiData = np.zeros_like(data_to_fit)
    for num_iter in range(fit_opts.opts.DenoizeMaxNumIterations):
        denoize_algorithm.get_nc_stats(data=data_to_fit)
        if denoize_algorithm.check_low_noise(data_max=np.max(data_to_fit) / 5):
            break
        logging.info(f"denoize iteration: {num_iter + 1}")
        d_niiData = denoize_algorithm.denoize_nii_data(
            data=data_to_fit,
            save_plot=save_plot_path
        )
        data_to_fit = d_niiData

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
    db_path = plib.Path(fit_opts.config.DatabasePath).absolute()
    if not db_path.is_file():
        err = f"given database file {db_path.__str__()} does not exist. exiting..."
        logging.error(err)
        raise FileNotFoundError(err)
    logging.info(f"loading database: {db_path.__str__()}")
    db = emc_db.DB.load(db_path)
    logging.info("setting up fit")
    fit_algorithm = fit.EmcFit(nifti_data=data_to_fit, database=db,
                               mp_processing=fit_opts.config.Multiprocessing,
                               mp_headroom=fit_opts.config.HeadroomMultiprocessing)
    if fit_opts.opts.FitB1Weighting:
        # want to use B1 weighting in the fit
        b1_path = plib.Path(fit_opts.opts.FitB1WeightingInput).absolute()
        if b1_path.is_file():
            # we have a b1 input file
            logging.info(f"using B1 input file: {b1_path.__str__()}")
            # load file
            b1_map = nib.load(b1_path)
            fit_algorithm.set_b1_weight(b1_map=b1_map.get_fdata(), b1_lambda=fit_opts.opts.FitB1WeightingLambda)
        else:
            # use spherical prior
            logging.info(f"no b1 file specified, using spherical prior")
            fit_algorithm.set_b1_simple_prior(
                b1_lambda=fit_opts.opts.FitB1WeightingLambda,
                voxel_dims_mm=np.array(fit_opts.opts.FitB1PriorVoxelDims))
    # fitting
    t2_map, b1_map = fit_algorithm.get_maps()
    r2_map = np.divide(1, 1e-3 * t2_map, where=t2_map>1e-5, out = np.zeros_like(t2_map))

    # saving
    save_path = plib.Path(fit_opts.config.OutputPath).absolute()
    if save_path.suffixes:
        # catch if filename given
        save_path = save_path.parent
    # create folder ifn exist
    save_path.mkdir(parents=True, exist_ok=True)

    save_nii_file(t2_map, f"{fit_opts.config.NameId}_t2", save_path, data_img.affine)
    save_nii_file(r2_map, f"{fit_opts.config.NameId}_r2", save_path, data_img.affine)
    save_nii_file(b1_map, f"{fit_opts.config.NameId}_b1", save_path, data_img.affine)

    logging.info(f"finished")


def save_nii_file(data: np.ndarray, name: str, path: typing.Union[str, plib.Path], affine: nib.Nifti1Image.affine):
    img_save = path.joinpath(f"{name}_map").with_suffix(".nii")
    logging.info(f"write file: {img_save}")
    _img = nib.Nifti1Image(data, affine)
    nib.save(_img, img_save)


def mode_both(
        fit_opts: options.FitOptions, data_to_fit: np.ndarray, data_img: nib.Nifti1Image) -> (np.ndarray, np.ndarray):
    denoized_data = mode_denoize(fit_opts=fit_opts, data_to_fit=data_to_fit, data_img=data_img)
    mode_fit(fit_opts=fit_opts, data_to_fit=denoized_data, data_img=data_img)


def main(fit_opts: options.FitOptions):
    dataPath = plib.Path(fit_opts.config.NiiDataPath).absolute()
    niiData, niiImg = load_data(dataPath, test_debugging_flag=fit_opts.opts.TestingFlag)

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
    except Exception as e:
        logging.error(e)
        parser.print_usage()
