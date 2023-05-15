import simple_parsing as sp
import dataclasses as dc
import logging

logModule = logging.getLogger(__name__)


@dc.dataclass
class Config(sp.Serializable):
    configFile: str = sp.field(alias=["-c"], default="")
    emcSimulationConfiguration: str = sp.field(alias=["-ec"], default="./emc_abo/config/emc_config.json")
    saveFile: str = sp.field(alias=["-s"], default="./emc_abo/result/optim_fa.json")
    multiProcess: bool = sp.field(alias=["-mp"], default=True)
    mpHeadroom: int = sp.field(alias=["-mph"], default=8)
    mpNumCpus: int = sp.field(alias=["-mpn"], default=16)
    optimMaxIter: int = sp.field(alias=["-omi"], default=100)
    optimPopsize: int = sp.field(alias=["-omp"], default=15)
    optimCrossover: float = sp.field(alias=["-omc"], default=0.5)
    optimMutation: float = sp.field(alias=["-omm"], default=1.3)
    optimLambda: float = sp.field(alias=["-oml"], default=0.3)
    useYabox: bool = sp.field(alias=["-yb"], default=False)
    varyPhase: bool = sp.field(alias=["-p"], default=False)
    varySpoilGrad: bool = sp.field(alias=["-spg"], default=False)
    maxTime: int = sp.field(alias=["-t"], default=6*60*60)      # 6h maximum compute time

    @classmethod
    def create_from_cmd(cls, args: sp.ArgumentParser.parse_args):
        conf = Config.from_dict(args.config.to_dict())

        nonDefaultConfig = conf._checkNonDefaultVars()

        if args.config.configFile:
            conf = Config.load(args.config.configFile)
            # overwrite non default input args - eg. given by cmd line extra to config file
            for key, value in nonDefaultConfig.items():
                conf.__setattr__(key, value)

        return conf

    def _checkNonDefaultVars(self) -> dict:
        defConfig = Config()
        nonDefaultConfig = {}
        for key, value in vars(self).items():
            if self.__getattribute__(key) != defConfig.__getattribute__(key):
                nonDefaultConfig.__setitem__(key, value)

        return nonDefaultConfig


def createCmdLineParser() -> (sp.ArgumentParser, sp.ArgumentParser.parse_args):
    """
    Build the parser for arguments
    Parse the input arguments.
    :return: parser, parsed arguments
    """
    parser = sp.ArgumentParser(prog='emc_abo')
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()

    return parser, args
