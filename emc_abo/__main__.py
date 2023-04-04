"""
Main script for optimization
"""
from emc_abo import abo, options
import logging


def main(conf: options.Config):
    abo_optimizer = abo.Optimizer(config=conf)
    logging.info(f"abo configuration: \n -- {conf}")
    abo_optimizer.optimize()
    abo_optimizer.save(conf.saveFile)
    abo_optimizer.plot()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    parser, args = options.createCmdLineParser()
    abo_config = options.Config.create_from_cmd(args)

    try:
        main(conf=abo_config)
    except Exception as e:
        logging.error(e)
        parser.print_usage()
