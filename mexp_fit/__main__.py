"""
small script for simple monoexponential fitting of data
_____________
13.10.2022
Jochen Schmidt
"""
import logging
import numpy as np
from mexp_fit import options, fit
import emc_fit


def main(config: options.Configuration):
    logging.info("Start processing")

    logging.info(f"loading in data - {config.NiiDataPath}")
    nii_data, nii_img = emc_fit.load_data(config.NiiDataPath)
    # map range 0 - 1000
    nii_data = np.divide(1e3 * nii_data, nii_data.max())
    if config.Visualize:
        emc_fit.plots.plot_ortho_view(nii_data)

    if config.Denoize:
        logging.info("Denoizing Data")
        denoizer = emc_fit.denoize.MajMinNcChiDenoizer(
            max_num_runs=config.DenoizeMaxNumIterations,
            use_mp=config.DenoizeUseMp, mp_headroom=config.DenoizeMpHeadroom,
            visualize=config.Visualize
        )
        for num_iter in range(config.DenoizeMaxNumIterations):
            denoizer.get_nc_stats(data=nii_data)
            if denoizer.check_low_noise(data_max=np.max(nii_data) / 5):
                break
            logging.info(f"denoize iteration: {num_iter + 1}")
            nii_data = denoizer.denoize_nii_data(
                data=nii_data
            )

    logging.info("Fitting data")
    t2_map, s_map = fit.fit_data(nii_data, te=config.ArrEchoTimes)
    r2_map = np.divide(1e3, t2_map, where=t2_map > 1e-8, out=np.zeros_like(t2_map))
    if config.Visualize:
        emc_fit.plots.plot_ortho_view(t2_map)

    output_path = config.OutputPath
    options.save_nii(output_path, t2_map, nii_img, name=f"{config.NameId}_t2")
    options.save_nii(output_path, r2_map, nii_img, name=f"{config.NameId}_r2")
    options.save_nii(output_path, s_map, nii_img, name=f"{config.NameId}_s0")
    if config.Denoize:
        options.save_nii(output_path, nii_data, nii_img, name=f"{config.NameId}_denoized")


if __name__ == '__main__':
    parser, args = options.createCmdLineParser()
    opts = options.Configuration.from_cmd_args(args)
    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)

    try:
        main(opts)
        logging.info("finished succesful")
    except AttributeError as ae:
        logging.error(ae)
        parser.print_usage()
