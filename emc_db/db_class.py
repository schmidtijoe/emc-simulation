import logging
import pickle
import typing
import emc_sim.options as es_opts
import numpy as np
import pandas as pd
import pathlib as plib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mpc
plt.style.use('ggplot')

# debugging dev
import emc_sim.utils as emsu

logModule = logging.getLogger(__name__)


class DB:
    def __init__(self, pd_dataframe: pd.DataFrame = pd.DataFrame(),
                 config: es_opts.SimulationConfig = es_opts.SimulationConfig(), name: str = "db_"):
        self.pd_dataframe: pd.DataFrame = pd_dataframe
        self.config: es_opts.SimulationConfig = config
        self.np_array: np.ndarray = np.array([*pd_dataframe.emc_signal.to_numpy()])
        self.etl: int = self.np_array.shape[-1]
        self.name: str = name

        self.indexes: list = ["emc_signal", "t2", "t1", "b1", "d"]

        # normalize
        self.normalize()

    def get_indexes(self):
        return self.indexes

    def get_t2_b1_values(self) -> (np.ndarray, np.ndarray):
        return np.unique(self.pd_dataframe.t2), np.unique(self.pd_dataframe.b1)

    def plot(self, t2_range_ms: tuple = (10.0, 40.0), b1_range: tuple = (0.6, 1.2), save: str = ""):
        logModule.info("plotting")
        t2_range_s = 1e-3 * np.array(t2_range_ms)
        df_selection = self.pd_dataframe[self.pd_dataframe.t2.between(t2_range_s[0], t2_range_s[1], inclusive='both')]
        df_selection = df_selection[df_selection.b1.between(b1_range[0], b1_range[1], inclusive='both')]
        t2s = np.unique(df_selection.t2)
        b1s = np.unique(df_selection.b1)

        x_ax = np.arange(1, self.etl + 1)

        if b1s.shape[0] > 8:
            b1s = b1s[::2]
        if b1s.shape[0] > 4:
            b1s = b1s[:4]

        curves = np.zeros((t2s.shape[0], b1s.shape[0], self.etl))
        c_range = np.linspace(0.25, 1.0, t2s.shape[0])

        cmaps = [cm.get_cmap('Purples'), cm.get_cmap('Greens'), cm.get_cmap('Oranges'), cm.get_cmap('Reds')]
        colors = [cmaps[k](c_range) for k in range(b1s.shape[0])]

        for t2_idx in range(t2s.shape[0]):
            for b1_idx in range(b1s.shape[0]):
                curve = df_selection.emc_signal[
                    (df_selection.t2 == t2s[t2_idx]) & (df_selection.b1 == b1s[b1_idx])
                    ].to_numpy()[0]
                curves[t2_idx, b1_idx] = np.divide(curve, np.linalg.norm(curve))

        fig = plt.figure(figsize=(14, 6))
        wr = np.ones(len(cmaps) + 1)
        wr[0] = 20

        gs = fig.add_gridspec(1, 1 + len(cmaps), width_ratios=wr, wspace=0.1)
        ax = fig.add_subplot(gs[0])
        ax.set_xlabel(f"echo #")
        ax.set_ylabel(f"$l_2$ normalized intensity [A.U.]")
        ax.set_yticklabels([])

        for b in range(curves.shape[1]):
            ax.hlines(0.2 * (b + 1), 0, x_ax[-1], color=colors[b][-int(t2s.shape[0]/3)], linestyle='dotted')
            for a in range(curves.shape[0]):
                ax.plot(x_ax, 0.2 * (b + 1) + curves[a, b], color=colors[b][a])

        norm = mpc.Normalize(vmin=t2_range_ms[0], vmax=t2_range_ms[1])
        ticks = [[], None]
        titles = [f"$B_1$: {b1s[0]:.1f} \t", *[f"{b1s[k]:.1f}" for k in np.arange(1, b1s.shape[0])]]
        for k in range(len(cmaps)):
            cax = fig.add_subplot(gs[1 + k])
            cax.grid(False)
            if k < len(cmaps) - 1:
                cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmaps[k]), cax=cax, ticks=ticks[0])
            else:
                cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmaps[k]), cax=cax)
            cax.set_title(titles[k])
        cb.set_label(f"$T_2$ [ms]")
        if save:
            save_path = plib.Path(save).absolute()
            if not save_path.suffixes:
                save_path = save_path.joinpath(f"{self.name}_plot.png")
            if ".png" not in save_path.suffixes:
                logModule.info(f"plot saved as .png image. suffix adapted!")
                save_path = save_path.with_suffix(".png")
            save_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path, bbox_inches='tight', transparent=True)
        plt.show()

    def save(self, path: typing.Union[str, plib.Path]):
        path = plib.Path(path).absolute()
        if not path.suffixes:
            # given a path not a file
            path = path.joinpath(f"{self.name}_database_file.pkl")
        if ".pkl" not in path.suffixes:
            # given wrong filending
            logModule.info("filename saved as .pkl, adopting suffix.")
            path = path.with_suffix('.pkl')
        # mkdir ifn existent
        path.parent.mkdir(exist_ok=True, parents=True)

        logModule.info(f"writing file {path}")

        with open(path, "wb") as p_file:
            pickle.dump(self, p_file)

    @classmethod
    def load(cls, path: typing.Union[str, plib.Path]):
        path = plib.Path(path).absolute()
        if ".pkl" not in path.suffixes:
            # given wrong filending
            logModule.info("filename not .pkl, try adopting suffix.")
            path = path.with_suffix('.pkl')
        if not path.is_file():
            # given a path not a file
            err = f"{path.__str__()} not a file"
            logModule.error(err)
            raise ValueError(err)
        with open(path, "rb") as p_file:
            db = pickle.load(p_file)
        return db

    def normalize(self):
        arr = self.np_array
        norm = np.linalg.norm(arr, axis=-1, keepdims=True)
        self.np_array = np.divide(arr, norm, where=norm>1e-12, out=np.zeros_like(arr))

        for k in range(len(self.pd_dataframe)):
            self.pd_dataframe.at[k, "emc_signal"] = self.np_array[k]


if __name__ == '__main__':
    db_pd, _ = emsu.load_database(
        "D:\\Daten\\01_Work\\11_owncloud\\ds_mese_cbs_js\\01_invivo_scan_data\\emc\\database_0p6_fa180_esp12p3.pkl"
    )
    db_pd = db_pd.rename(columns={"emcSignal": "emc_signal"})

    database = DB(name="test_db", pd_dataframe=db_pd)

    database.plot(save="test/")
    database.save(path="test/")

    dl = DB.load("test/test_db_database_file.pkl")
    dl.plot()
