"""
Main script for optimization
"""
from emc_abo import abo
import logging


def main():
    abo_optimizer = abo.Optimizer(
        config_path="./emc_abo/config/emc_config.json",
        multiprocessing=True,
        mp_num_cpus=200
    )
    abo_optimizer.optimize()
    abo_optimizer.save("emc_abo/result/optim_fa.json")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    try:
        main()
    except Exception as e:
        logging.error(e)
