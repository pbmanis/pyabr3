import datetime
import pathlib
import pickle
import platform
from pathlib import Path
from string import ascii_letters
from typing import Union

import numpy as np
import pandas as pd
import pyqtgraph as pg
from matplotlib import pyplot as mpl
from pylibrary.plotting import styler as ST
from pylibrary.tools import cprint as CP

# import ephys.tools.get_configuration as GETCONFIG
import get_configuration as GETCONFIG
import plothelpers as mpl_PH
from src import abr_regexs as REX
from src import filter_util as filter_util
from src import parse_ages as PA
from src import read_calibration as read_calibration

use_matplotlib = True
from pylibrary.plotting import plothelpers as PH

# Check the operating system and set the appropriate path type
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

class AnalyzeABR:
    def __init__(self):
        self.caldata = None
        self.gain = 1e4
        self.FILT = filter_util.Utility()
        self.frequencies = []
        self.hide_treatment = False
        self.experiment = (
            None  # this is the experiment dict from the project configuration directory.
        )

        # 24414.0625

    def set_hide_treatment(self, hide_treatment: bool):
        self.hide_treatment = hide_treatment

    def get_experiment(
        self, config_file_name: Union[Path, str] = "experiments.cfg", exptname: str = None
    ):
        datasets, experiments = GETCONFIG.get_configuration(config_file_name)
        if exptname in datasets:
            self.experiment = experiments[exptname]
        else:
            raise ValueError(
                f"Experiment {exptname} not found in the configuration file with datasets={datasets!s}"
            )

    def read_abr_file(self, fn):
        with open(fn, "rb") as fh:
            d = pickle.load(fh)
            self.caldata = d["calibration"]
            # Trim the data array to remove the delay to the stimulus.
            # to be consistent with the old ABR4 program, we leave
            # the first 1 msec prior to the stimulus in the data array
            delay = float(d["protocol"]["stimuli"]["delay"])
            # print("read_abr_file, d = ", d.keys())
            # print(d['record_frequency'], "Hz, delay: ", delay, "msec")
            if d['record_frequency'] is None:
                d['record_frequency'] = 24414.0625  # default to 24.4 kHz
            i_delay = int((delay - 1e-3) * d["record_frequency"])
            d["data"] = d["data"][i_delay:]
            d["data"] = np.append(d["data"], np.zeros(i_delay))
        return d

    def show_calibration(self, fn):
        # d = self.read_abr_file(fn)
        # dbc = self.convert_attn_to_db(20.0, 32000)
        # print(dbc)
        read_calibration.plot_calibration(self.caldata)

    def show_calibration_history(self):
        fn = "abr_data/calibration_history"
        files = list(Path(fn).glob("frequency_MF1*.cal"))
        cal_files = []
        creation_dates = []
        for j, f in enumerate(files):
            cal_files.append(read_calibration.get_calibration_data(f))
            creation_dates.append(f.stat().st_ctime)
        app = pg.mkQApp("Calibration Data Plot")
        win = pg.GraphicsLayoutWidget(show=True, title="ABR Data Plot")
        win.resize(800, 600)
        win.setWindowTitle(f"Calibration History")
        symbols = ["o", "s", "t", "d", "+", "x"]
        win.setBackground("w")
        pl = win.addPlot(title=f"Calibration")
        # li = pg.LegendItem()
        # li.setColumnCount(2)

        # li.setParentItem(pl)
        pl.setLogMode(x=True, y=False)
        pl.addLegend()
        pl.legend.setOffset((0.1, 0))
        pl.legend.setColumnCount(2)
        for cd in sorted(creation_dates):
            i = creation_dates.index(cd)
            caldata = cal_files[i]
            freqs = caldata["freqs"]
            if "2022" in caldata["date"]:
                continue
            # pl.plot(freqs, caldata['maxdb'], pen='r', name="Max SPL (0 dB Attn)")
            pl.plot(
                freqs,
                caldata["db_cs"],
                pen=pg.mkPen(i, len(cal_files), width=2),
                symbolPen=pg.mkPen(i, len(cal_files)),
                symbol=symbols[i % len(symbols)],
                symbolSize=7,
                name=f"{caldata['date']:s}",
            )
            # li.addItem(pl.legend.items[-1][1], f"{caldata['date']:s}")
            # pl.plot(freqs, caldata['db_bp'], pen='g', name=f"Measured dB SPL, attn={caldata['calattn']:.1f}, bandpass")
            # pl.plot(freqs, caldata['db_nf'], pen='b', name="Noise Floor")
            # pl.setLogMode(x=True, y=False)
        pl.setLabel("bottom", "Frequency", units="Hz")
        pl.setLabel("left", "dB SPL")
        pl.showGrid(x=True, y=True)
        # text_label = pg.LabelItem(txt, size="8pt", color=(255, 255, 255))
        # text_label.setParentItem(pl)
        # text_label.anchor(itemPos=(0.5, 0.05), parentPos=(0.5, 0.05))

        pg.exec()

    def convert_attn_to_db(self, attn, fr):
        """convert_attn_to_db converts the attenuation value at a particular
        frquency to dB SPL, based on the calibration file data

        Parameters
        ----------
        attn : float
            attenuator setting (in dB)
        fr : float
            the frequency of the stimulus (in Hz)
        """
        if self.caldata is None:
            raise ValueError(
                f"Calibration data not loaded; must load from a data file or a calibration file"
            )

        dBSPL = 0.0
        if fr in self.caldata["freqs"]:  # matches a measurement frequency
            ifr = np.where(self.caldata["freqs"] == fr)
            dBSPL = self.caldata["maxdb"][ifr]
            # print("fixed")
        else:
            # interpolate linearly between the two closest frequencies
            # first, we MUST sort the caldata and frequencies so that the freqs are in ascending
            # order, otherwise numpy gives the wrong result.
            ifr = np.argsort(self.caldata["freqs"])
            freqs = self.caldata["freqs"][ifr]
            maxdb = self.caldata["maxdb"][ifr]
            # for i, frx in enumerate(freqs):
            #     print(f"{frx:8.1f}  {maxdb[i]:8.3f}")
            dBSPL = np.interp([fr], freqs, maxdb)
        #     print("interpolated", dBSPL)
        # print(f"dBSPL = {dBSPL:8.3f} for freq={float(fr):9.2f} with {float(attn):5.1f} dB attenuation")
        dBSPL_corrected = dBSPL[0] - attn
        return dBSPL_corrected

    def average_within_traces(
        self, fd, i, protocol, date, high_pass_filter: Union[float, None] = None, low_pass_filter: Union[float, None] = None,
    ):
        # _i_ is the index into the acquisition series. There is one file for each repetition of each condition.
        # the series might span a range of frequencies and intensities; these are
        # in the protocol:stimulus dictionary (dblist, freqlist)
        # we average response across traces for each intensity and frequency
        # this function specifically works when each trace has one stimulus condition (db, freq), and is
        # repeated nreps times.
        # The returned data is the average of the responses across the nreps for this (the "ith") stimulus condition
        stim_type = str(Path(fd).stem)
        # print("Stim type: ", stim_type)
        # print("avg within traces:: pyabr3 Protocol: ", protocol)
        if stim_type.lower().startswith("tone"):
            name = "tonepip"
        elif stim_type.lower().startswith("interleaved"):
            name = "interleaved_plateau"
        elif stim_type.lower().startswith("click"):
            name = "click"
        nreps = protocol["stimuli"]["nreps"]
        rec = protocol["recording"]
        interstim_interval = protocol["stimuli"]["interval"]
        missing_reps = []
        for n in range(nreps):  # loop over the repetitions for this specific stimulus
            fn = f"{date}_{name}_{i:03d}_{n+1:03d}.p"
            if not Path(fd, fn).is_file():
                missing_reps.append(n)
                continue
            d = self.read_abr_file(Path(fd, fn))
            sample_rate = d["record_frequency"]
            if n == 0:
                data = d["data"]
                tb = np.linspace(0, len(data) / sample_rate, len(data))
            else:
                data += d["data"]

        if len(missing_reps) > 0:
            CP.cprint("r", f"Missing {len(missing_reps)} reps for {name} {i}", "red")
        data = data / (nreps - len(missing_reps))
        if high_pass_filter is not None:
            data = self.FILT.SignalFilter_HPFButter(
                data, high_pass_filter, sample_rate, NPole=4, bidir=True
            )
        if low_pass_filter is not None:
            data = self.FILT.SignalFilter_LPFButter(
                data, low_pass_filter, sample_rate, NPole=4, bidir=True
            )
        # tile the traces.
        # first interpolate to 100 kHz
        # If you don't do this, the blocks will precess in time against
        # the stimulus, which is timed on a 500 kHz clock.
        # It is an issue because the TDT uses an odd frequency clock...

        trdur = len(data) / sample_rate
        newrate = 1e5
        tb100 = np.arange(0, trdur, 1.0 / newrate)

        one_response = int(interstim_interval * newrate)  # interstimulus interval
        arraylen = one_response * protocol["stimuli"]["nstim"]

        abr = np.interp(tb100, tb, data)
        sub_array = np.split(abr[:arraylen], protocol["stimuli"]["nstim"])
        sub_array = np.mean(sub_array, axis=0)
        tb100 = tb100[:one_response]
        return sub_array, tb100, newrate

    def average_across_traces(
        self, fd, i, protocol, date, high_pass_filter: Union[float, None] = None, low_pass_filter: Union[float, None] = None
    ):
        """average_across_traces for abrs with multiple stimuli in a trace.
        This function averages the responses across multiple traces.
        and returns a list broken down by the individual traces.

        Parameters
        ----------
        fd : _type_
            _description_
        i : _type_
            _description_
        protocol : _type_
            _description_
        date : _type_
            _description_
        stim_type : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        nreps = protocol["stimuli"]["nreps"]
        rec = protocol["recording"]

        dblist = protocol["stimuli"]["dblist"]
        frlist = protocol["stimuli"]["freqlist"]
        stim_type = str(Path(fd).stem)
        # print("Stim type: ", stim_type)
        if stim_type.lower().startswith("interleaved"):
            stim_type = "interleaved_plateau"
        missing_reps = []
        ndata = 0
        for n in range(nreps):
            fn = f"{date}_{stim_type}_{i:03d}_{n+1:03d}.p"
            if not Path(fd, fn).is_file():
                missing_reps.append(n)
                continue
            d = self.read_abr_file(Path(fd, fn))

            if ndata == 0:
                data = d["data"]
            else:
                data += d["data"]
            if ndata == 0:

                tb = np.arange(0, len(data) / d["record_frequency"], 1.0 / d["record_frequency"])
            ndata += 1
        # if len(missing_reps) > 0:
        #     CP.cprint("r", f"Missing {len(missing_reps)} reps for {fn!s}", "red")
        data = data / (nreps - len(missing_reps))
        if high_pass_filter is not None:
            # print("average across traces hpf: ", high_pass_filter, d["record_frequency"])
            data = self.FILT.SignalFilter_HPFButter(
                data, HPF=high_pass_filter, samplefreq=d["record_frequency"], NPole=4, bidir=True
            )
        if low_pass_filter is not None:
            data = self.FILT.SignalFilter_LPFButter(
                data, low_pass_filter, samplefreq=d["record_frequency"], NPole=4, bidir=True
            )
        # tile the traces.
        # first linspace to 100 kHz
        trdur = len(data) / d["record_frequency"]
        newrate = 1e5
        tb = np.linspace(0, trdur, len(data))
        # f, ax = mpl.subplots(2,1)
        # ax[0].plot(tb, data)

        tb100 = np.arange(0, trdur, 1.0 / newrate)
        # print('len tb100: ', len(tb100), np.mean(np.diff(tb100)))
        abr = np.interp(tb100, tb, data)
        # ax[0].plot(tb100, abr, '--r', linewidth=0.5)
        # mpl.show()
        # exit()

        one_response_100 = int(protocol["stimuli"]["stimulus_period"] * newrate)
        # print("avg across traces:: pyabr3 .p file protocol[stimuli]: ", protocol["stimuli"])
        if isinstance(protocol["stimuli"]["freqlist"], str):
            frlist = eval(protocol["stimuli"]["freqlist"])
        if isinstance(protocol["stimuli"]["dblist"], str):
            dblist = eval(protocol["stimuli"]["dblist"])
        arraylen = one_response_100 * protocol["stimuli"]["nstim"]
        if stim_type.lower() in ["click", "tonepip"]:
            nsplit = protocol["stimuli"]["nstim"]
        elif stim_type.lower() in ["interleaved_plateau"]:
            nsplit = int(len(frlist) * len(dblist))
        else:
            raise ValueError(f"Stimulus type {stim_type} not recognized")
        arraylen = one_response_100 * nsplit
        # print(len(frlist), len(dblist))
        # print("one response: ", one_response_100)
        # print("nsplit: ", nsplit)
        # print("arranlen/nsplit: ", float(arraylen) / nsplit)
        # print("len data: ", len(data), len(abr), nsplit * one_response_100)
        sub_array = np.split(abr[:arraylen], nsplit)
        # print(len(sub_array))
        abr = np.array(sub_array)
        tb = tb100[:one_response_100]

        # print("abr shape: ", abr.shape, "max time: ", np.max(tb))
        stim = np.meshgrid(frlist, dblist)
        return abr, tb, newrate, stim

    def show_stimuli(self, fn):
        d = self.read_abr_file(fn)
        stims = list(d["stimuli"].keys())
        wave = d["stimuli"][stims[0]]["sound"]
        pg.plot(wave)
        pg.exec()

    ###########################################################################

    def parse_metadata(self, metadata, stim_type, acquisition):
        """parse_metadata : use the metadata dict to generate a
        title string for the plot, and the stack order for the data.

        Parameters
        ----------
        metadata : _type_
            _description_
        filename : _type_
            _description_
        stim_type : _type_
            _description_
        acquisition : _type_
            _description_


        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        fn = metadata["filename"]
        if metadata["type"] == "ABR4":
            filename = str(Path(fn).name)
            if filename in ["Click", "Clicks", "Tone", "Tones"]:
                filename = str(Path(fn).parent.name)
            else:
                filename = str(Path(fn).parent)
        else:
            filename = str(Path(fn).parent)
        subject_id = metadata["subject_id"]
        age = PA.ISO8601_age(metadata["age"])
        sex = metadata["sex"]
        amplifier_gain = metadata["amplifier_gain"]
        strain = metadata["strain"]
        weight = metadata["weight"]
        genotype = metadata["genotype"]
        rec_freq = metadata["record_frequency"]
        hpf = metadata["highpass"]

        title_file_name = filename
        page_file_name = filename
        if self.hide_treatment:
            # hide the treatment values in the title just in case...
            if acquisition == "ABR4":
                tf_parts = list(Path(fn).parts)
                fnp = tf_parts[-2]
                fnsplit = fnp.split("_")
                fnsplit[-1] = "*"
                tf_parts[-2] = "_".join([x for x in fnsplit])
                title_file_name = tf_parts[-2]
                page_file_name = str(Path(*tf_parts))
            elif acquisition == "pyabr3":
                tf_parts = list(Path(fn).parts)
                fnp = tf_parts[-3]
                fnsplit = fnp.split("_")
                fnsplit[-1] = "*"
                tf_parts[-3] = "_".join([x for x in fnsplit])
                page_file_name = str(Path(*tf_parts[:-1]))
                title_file_name = tf_parts[-3]

        title = f"\n{title_file_name!s}\n"
        title += f"Stimulus: {stim_type}, Amplifier Gain: {amplifier_gain}, Fs: {rec_freq}, HPF: {hpf:.1f}, Acq: {acquisition:s}\n"
        title += f"Subject: {subject_id:s}, Age: {age:s} Sex: {sex:s}, Strain: {strain:s}, Weight: {weight:.2f}, Genotype: {genotype:s}"

        # determine the direction for stacking the plots.
        # stack_dir = "up"
        # if metadata["type"] == "ABR4":
        #     stack_dir = "up"
        # elif metadata["type"] == "pyabr3" and stim_type.lower().startswith("click"):
        #     stack_dir = "up"
        # elif metadata["type"] == "pyabr3" and (
        #     stim_type.lower().startswith("tone") or stim_type.lower().startswith("interleaved")
        # ):
        #     stack_dir = "up"
        # else:
        #     raise ValueError(
        #         f"Stimulus type {stim_type} not recognized togethger with data type {metadata['type']}"
        #     )
        return title, page_file_name  # , stack_dir

    def plot_abrs(
        self,
        abr_data: np.ndarray,
        tb: np.ndarray,
        waveana: object,
        scale: str = "V",
        ax_plot: Union[object, None] = None,
        acquisition: str = "pyabr3",
        V_stretch: float = 1.0,
        stim_type: str = "Click",
        dblist: Union[list, None] = None,
        frlist: Union[list, None] = None,
        maxdur: float = 10.0,
        thresholds: Union[list, None] = None,
        csv_filename: Union[str, None] = None,
        metadata: dict = {},
        use_matplotlib: bool = True,
        live_plot: bool = False,
        pdf: Union[object, None] = None,
    ):
        """plot_abrs : Make a plot of the ABR traces, either into a matplotlib figure or a pyqtgraph window.
        If ax_plot is None, then a new figure is created. If ax_plot is not None, then the plot is added to the
        current ax_plot axis.

        Parameters
        ----------
        abr_data : np.ndarray
            the abr data set to plot.
            Should be a 2-d array (db x time)
        tb : np.ndarray
            time base
        scale : str, optional
            scale representation of the data ["uV" or "V"], by default "V"
        ax_plot : Union[object, None], optional
            matplot axis to plot the data into, by default None
        acquisition : str, optional
            what kind of data is being plotted - from ABR4 or pyabr3, by default "pyabr3"
        V_stretch : float, optional
            Voltage stretch factor, by default 1.0
        stim_type : str, optional
            type of stimulus - click or tone, by default "click"
        dblist : Union[list, None], optional
            intensity values, by default None
        frlist : Union[list, None], optional
            frequency values, by default None
        maxdur : float, optional
            max trace duration, in msec, by default 14.0
        metadata : dict, optional
            a dictionary of metadata returned from reading the data file, by default {}
        use_matplotlib : bool, optional
            flag to force usage of matplotlib vs. pyqtgraph, by default True
        live_plot : bool, optional
            flag to allow a live plot (pyqtgraph), by default False
        pdf : Union[object, None], optional
            A pdf file object for the output, by default None

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        # print("metadata: ", metadata)
        sninfo = Path(metadata["filename"]).parent.parent.name
        subid = REX.re_subject.search(sninfo)

        amplifier_gain = metadata["amplifier_gain"]
        # if subid is not None:
        #     subj = subid.group("subject")
        #     if subj in ["N004", "N006", "N005", "N007"]:
        #         amplifier_gain = 1

        if scale == "uV":
            # amplifier gain has already been accountd for.
            added_gain = 1.0
        elif scale == "V":
            added_gain = 1e6  # convert to microvolts
        else:
            raise ValueError(f"Scale {scale} not recognized, must be 'V' or 'uV'")

        title, page_file_name = self.parse_metadata(metadata, stim_type, acquisition)
        if frlist is None or len(frlist) == 0 or stim_type == "click":
            frlist = [0]
            ncols = 1
            width = 5  # (1.0 / 3.0) * len(dblist) * ncols
            height = 1.0 * len(dblist)
            lmar = 0.15
        else:
            ncols = len(frlist)
            width = 2.0 * ncols
            height = 1.0 * len(dblist)
            lmar = 0.1

        stack_increment = self.experiment["ABR_settings"]["stack_increment"]
        if height > 10.0:
            height = 10.0 * (10.0 / height)
        if use_matplotlib:
            if ax_plot is None:  # no external plot to use
                P = mpl_PH.regular_grid(
                    cols=ncols,
                    rows=len(dblist),
                    order="rowsfirst",
                    figsize=(width, height),
                    verticalspacing=0.01,
                    horizontalspacing=0.03,
                    margins={
                        "leftmargin": lmar,
                        "rightmargin": 0.05,
                        "topmargin": 0.15,
                        "bottommargin": 0.10,
                    },
                )

                # check the data type to build the datasets


                print("title: ", title)
                P.figure_handle.text(0.5, 0.99, s=title, fontsize=11, ha="center", va="top")
                ax = P.axarr
            else:
                ax = ax_plot

            v_min = 0.0
            v_max = 0.0
            n = 0
            colors = [
                "xkcd:raspberry",
                "xkcd:red",
                "xkcd:orange",
                "xkcd:golden yellow",
                "xkcd:green",
                "xkcd:blue",
                "xkcd:purple",
                "xkcd:bright violet",
            ]
            click_colors = [
                "xkcd:azure",
                "xkcd:lightblue",
                "xkcd:purple",
                "xkcd:orange",
                "xkcd:red",
                "xkcd:green",
                "xkcd:golden yellow",
            ]
            n_click_colors = len(click_colors)
            refline_ax = []
            # print("dblist: ", dblist)
            df_abr_waves = pd.DataFrame()

            for j, fr in enumerate(range(ncols)):  # enumerate(abr_data.keys()):
                for i, db in enumerate(dblist):
                    db = float(db)
                    delta_y = 0.0
                    if ax_plot is not None:  # single axis
                        ax = ax_plot
                        delta_y = stack_increment * i
                    else:  # multiple axes
                        ax = P.axarr[len(dblist) - i - 1, j]
                        delta_y = 0
                    if not i % 2:
                        ax_plot.text(
                            -0.2, delta_y, f"{int(db):d}", fontsize=8, ha="right", va="center"
                        )
                    npts = abr_data.shape[-1]
                    n_disp_pts = int(
                        maxdur * 1e-3 * metadata["record_frequency"]
                    )  # maxdur is in msec.
                    if n_disp_pts < npts:
                        npts = n_disp_pts
                    # print("added_gain: ", added_gain)
                    if abr_data.ndim > 3:
                        abr_data = abr_data.squeeze(axis=0)
                    if stim_type == "Click":
                        plot_data = added_gain * abr_data[0, i, :] / amplifier_gain
                    else:
                        plot_data = added_gain * abr_data[i, j, :] / amplifier_gain
                    # print("thresholds in plot_abrs: ", thresholds)
                    if (
                        thresholds is not None
                        and isinstance(thresholds, list)
                        and len(thresholds) < j
                        and np.abs(db - thresholds[j]) < 1
                    ):
                        CP.cprint(
                            "c", f"** Found threshold, db: {db!s}, threshold: {thresholds[j]!s}"
                        )
                        ax.plot(
                            tb[:npts] * 1e3,
                            np.zeros_like(plot_data[:npts]) + delta_y,
                            color="xkcd:blue",
                            linewidth=4,
                            clip_on=False,
                            alpha=0.7,
                        )
                    else:
                        pass
                        # if thresholds is None:
                        #     CP.cprint("r", f"** Threshold value is NOne: db: {db!s}")

                        # else:
                        #     CP.cprint("r", f"** DID NOT find threshold, db: {db!s}, threshold: {thresholds[j]!s}")

                    if stim_type in ["Click"]:
                        ax.plot(
                            tb[:npts] * 1e3,
                            plot_data[:npts] + delta_y,
                            color=click_colors[i % n_click_colors],
                            linewidth=1,
                            clip_on=False,
                        )
                        if i == 0:
                            df_abr_waves["time"] = tb[:npts] * 1e3
                        df_abr_waves = pd.concat(
                            [df_abr_waves, pd.DataFrame({f"{dblist[i]:.1f}": plot_data[:npts]})],
                            axis=1,
                        )
                        # ax.plot(waveana.p1_latencies[i]*1e3, waveana.p1_amplitudes[i]+delta_y, 'ro', clip_on=False)
                        # ax.plot(waveana.n1_latencies[i]*1e3, waveana.n1_amplitudes[i]+delta_y, 'bo', clip_on=False)
                        marker_at = 1.0
                        ax.plot(
                            [marker_at, marker_at],
                            [delta_y - 1, delta_y + 1],
                            "k-",
                            clip_on=False,
                            linewidth=0.5,
                        )
                        if i == len(dblist) - 1:
                            ax.text(
                                marker_at,
                                delta_y + 1.5,
                                f"{marker_at:.1f} ms",
                                fontsize=8,
                                ha="center",
                                va="bottom",
                            )
                    else:
                        ax.plot(
                            (tb[:npts]) * 1e3,
                            plot_data[:npts] + delta_y,
                            color=colors[j],
                            clip_on=False,
                        )
                    ax.plot(
                        [tb[0] * 1e3, tb[npts] * 1e3],
                        [delta_y, delta_y],
                        linestyle="--",
                        linewidth=0.3,
                        color="grey",
                        zorder=0,
                    )
                    if ax not in refline_ax:
                        PH.referenceline(ax, reference=delta_y, linewidth=0.5)
                        refline_ax.append(ax)
                    # print(dir(ax))
                    ax.set_facecolor(
                        "#ffffff00"
                    )  # background will be transparent, allowing traces to extend into other axes
                    ax.set_xlim(0, maxdur)
                    # let there be an axis on one trace (at the bottom)

                    # if stack_dir == "up":
                    if i == len(dblist) - 1:
                        if ncols > 1:
                            ax.set_title(f"{frlist[j]} Hz")
                        else:
                            ax.set_title("Click")
                        PH.noaxes(ax)
                    elif i == 0:
                        PH.nice_plot(ax, direction="outward", ticklength=3)
                        ax.set_xlabel("Time (ms)")
                        ticks = np.arange(0, maxdur, 2)
                        ax.set_xticks(ticks, [f"{int(k):d}" for k in ticks])
                    else:
                        PH.noaxes(ax)

                    # elif stack_dir == "down":
                    #     if i == 0:
                    #         if ncols > 1:
                    #             ax.set_title(f"{frlist[j]} Hz")
                    #         else:
                    #             ax.set_title("Click")
                    #         PH.noaxes(ax)
                    #     elif i == len(dblist) - 1:
                    #         PH.nice_plot(ax, direction="outward", ticklength=3)
                    #         ticks = np.arange(0, maxdur, 2)
                    #         ax.set_xticks(ticks, [f"{int(k):d}" for k in ticks])
                    #         ax.set_xlabel("Time (ms)")
                    #     else:
                    #         PH.noaxes(ax)

                    if j == 0:
                        ax.set_ylabel(
                            f"dB SPL",
                            fontsize=8,
                            labelpad=18,
                            rotation=90,
                            ha="center",
                            va="center",
                        )
                        if i == 0:
                            muv = r"\u03BC"
                            PH.calbar(
                                ax,
                                calbar=[-2.5, stack_increment * len(dblist), 1, 1],
                                unitNames={"x": "ms", "y": f"uV"},
                                xyoffset=[0.5, 0.1],
                                fontsize=6,
                            )
                    # ax.set_xlim(0, np.max(tb[:npts]) * 1e3)
                    ax.set_xlim(0, maxdur)

                # ax.set_xticks([1, 3, 5, 7, 9], minor=True)

                n += 1
                if np.max(plot_data[:npts]) > v_max:
                    v_max = np.max(plot_data[:npts])
                if np.min(plot_data[:npts]) < v_min:
                    v_min = np.min(plot_data[:npts])

            # if metadata["type"] == "pyabr3" and stim_type.lower().startswith("click"):
            #     V_stretch = 10.0 * V_stretch
            if csv_filename is not None:
                df_abr_waves.to_csv(csv_filename, index=False)
                CP.cprint("g", f"Saved ABR waveform data to {csv_filename!s}", "green")
            amax = np.max([-v_min, v_max]) * V_stretch
            if amax < 0.5:
                amax = 0.5
            # print(P.axarr.shape, len(dblist), len(frlist))

            for isp in range(len(dblist)):
                if ax_plot is None:
                    for j in range(len(frlist)):
                        P.axarr[isp, j].set_ylim(-amax, amax)
                        P.axarr[isp, j].set_xlim([0, maxdur])
                    # PH.referenceline(ax, linewidth=0.5)
            if ax_plot is None:
                transform = P.figure_handle.transFigure
            else:
                transform = ax_plot.transAxes
            mpl.text(
                0.96, 0.01, s=datetime.datetime.now(), fontsize=6, ha="right", transform=transform
            )
            mpl.text(
                0.02,
                0.01,
                s=f"{page_file_name:s}",
                fontsize=5,
                ha="left",
                transform=transform,
            )

        # else:  # use pyqtgraph
        #     app = pg.mkQApp("ABR Data Plot")
        #     win = pg.GraphicsLayoutWidget(show=True, title="ABR Data Plot")
        #     win.resize(1200, 1000)
        #     win.setWindowTitle(f"File: {str(Path(fn).parent)}")
        #     win.setBackground("w")

        #     lw = pg.LayoutWidget(parent=win)
        #     lw.addLabel(text="Hi there", row=0, col=0, rowspan=1, colspan=len(frlist))
        #     lw.nextRow()

        #     plid = []
        #     if len(frlist) == 0:
        #         frlist = [1]
        #     col = 0

        #     print("stim_type (in pg plotting)", stim_type)
        #     if stim_type not in ["clicks", "click"]:  # this is for tones/interleaved, etc.
        #         ref_set = False
        #         v_min = 0
        #         v_max = 0
        #         for i, db in enumerate(dblist):
        #             row = i  # int(i/5)
        #             for j, fr in enumerate(frlist):
        #                 col = j
        #                 # if tb is None:
        #                 #     npts = len(abr_data[i, j])
        #                 #     tb = np.linspace(0, npts / rec_freq, npts)

        #                 pl = win.addPlot(
        #                     title=f"{dblist[-i-1]} dB, {fr} Hz", col=col, row=row
        #                 )  # i % 5)
        #                 if not ref_set:
        #                     ref_ax = pl
        #                     ref_set = True
        #                 plot_data = 1e6 * abr_data[len(dblist) - i - 1, j] / amplifier_gain
        #                 lpd = len(plot_data)
        #                 if stim_type in ["click", "tonepip"]:
        #                     pl.plot(
        #                         tb[:lpd] * 1e3,
        #                         plot_data,
        #                         pen=pg.mkPen(j, len(dblist), width=2),
        #                         clipToView=True,
        #                     )
        #                 else:
        #                     pl.plot(
        #                         tb[:lpd] * 1e3,
        #                         plot_data,
        #                         pen=pg.mkPen(j, len(dblist), width=2),
        #                         clipToView=True,
        #                     )
        #                 pl.plot(
        #                     tb[:lpd] * 1e3,
        #                     np.zeros_like(plot_data),
        #                     pen=pg.mkPen(
        #                         "grey", linetype=pg.QtCore.Qt.PenStyle.DashLine, width=0.33
        #                     ),
        #                     clipToView=True,
        #                 )
        #                 # pl.showGrid(x=True, y=True)
        #                 if j == 0:
        #                     pl.setLabel("left", "Amp", units="uV")
        #                 if i == len(dblist) - 1:
        #                     pl.setLabel("bottom", "Time", units="ms")
        #                 pl.setYRange(-3.0, 3.0)
        #                 pl.setXRange(0, 10)
        #                 if ref_set:
        #                     pl.setXLink(ref_ax)
        #                     pl.setYLink(ref_ax)
        #                 if np.max(plot_data) > v_max:
        #                     v_max = np.max(plot_data)
        #                 if np.min(plot_data) < v_min:
        #                     v_min = np.min(plot_data)

        #         ref_ax.setYRange(v_min * V_stretch, v_max * V_stretch)

        #     else:  # pyqtgraph
        #         v0 = 0
        #         v = []
        #         for i, db in enumerate(dblist):
        #             if i == 0:
        #                 pl = win.addPlot(title=f"{db} dB, {fr} Hz")  # i % 5)
        #             pl.plot(
        #                 tb * 1e3,
        #                 -v0 + abr_data[i, j] / amplifier_gain,
        #                 pen=pg.mkPen(pg.intColor(i, hues=len(dblist)), width=2),
        #                 clipToView=True,
        #             )
        #             v0 += 1e-6 * amplifier_gain
        #             v.append(v0)
        #             # pl.showGrid(x=True, y=True)
        #             pl.setLabel("left", "Amplitude", units="mV")
        #             pl.setLabel("bottom", "Time", units="s")
        #             label = pg.LabelItem(f"{db:.1f}", size="11pt", color="#99aadd")
        #             label.setParentItem(pl)
        #             label.anchor(itemPos=(0.05, -v0 * 180), parentPos=(0.1, 0))
        #             # pl.setYRange(-2e-6, 2e-6)
        #             plid.append(pl)
        #         for i, db in enumerate(dblist):
        #             label = pg.LabelItem(f"{db:.1f}", size="11pt", color="#99aadd")
        #             label.setParentItem(pl)
        #             label.anchor(itemPos=(0.05, -v[i] * 200), parentPos=(0.1, 0))
        #             # win.nextRow()
        #             for j, fr in enumerate(frlist):
        #                 ax.set_title(f"{self.convert_attn_to_db(db, fr)} dBSPL, {fr} Hz")
        #                 if i == 0:
        #                     ax.set_xlabel("Time (s)")
        #                 if j == 0:
        #                     ax.set_ylabel("Amplitude")
        #                 ax.set_ylim(-50, 50)
        #                 PH.noaxes(ax)
        #                 if i == 0 and j == 0:
        #                     PH.calbar(
        #                         ax,
        #                         calbar=[0, -20, 2, 10],
        #                         scale=[1.0, 1.0],
        #                         xyoffset=[0.05, 0.1],
        #                     )
        #                 PH.referenceline(ax, linewidth=0.5)
        #                 n += 1

        #     pg.exec()

        # # Enable antialiasing for prettier plots
        # pg.setConfigOptions(antialias=True)

    # def check_fsamp(self, d):
    #     if d["record_frequency"] is None:
    #         raise ValueError()
    #     d["record_frequency"] = 24414.0625  # 97656.25
    #     return d["record_frequency"]

    def read_and_average_abr_files(
        self,
        filename,
        amplifier_gain=1e4,
        scale: str = "V",
        high_pass_filter: Union[float, None] = None,
        low_pass_filter: Union[float, None] = None,
        maxdur: Union[float, None] = None,
        pdf: Union[object, None] = None,
    ):
        d = self.read_abr_file(filename)
        # print("     Read and average abrs")

        if maxdur is None:
            maxdur = 25.0
        stim_type = d["protocol"]["protocol"]["stimulustype"]
        fd = Path(filename).parent
        fns = Path(fd).glob("*.p")
        # break the filenames into parts.
        # The first part is the date,
        # the second part is the stimulus type,
        # the third part is the index into the stimulus array
        # the fourth part is the repetition number for a given stimulus
        protocol = d["protocol"]
        rec = protocol["recording"]
        # print("rec: ", rec)
        dblist = protocol["stimuli"]["dblist"]
        frlist = protocol["stimuli"]["freqlist"]
        if isinstance(dblist, str):
            dblist = eval(dblist)
        if isinstance(frlist, str):
            frlist = eval(frlist)

        subject_id = d["subject_data"]["Subject ID"]
        if d["subject_data"]["Age"] != "":
            age = PA.ISO8601_age(d["subject_data"]["Age"])
        else:
            age = 0
        sex = d["subject_data"]["Sex"]
        strain = d["subject_data"]["Strain"]
        if d["subject_data"]["Weight"] != "":
            weight = float(d["subject_data"]["Weight"])
        else:
            weight = 0.0
        genotype = d["subject_data"]["Genotype"]

        file_parts = Path(filename).stem.split("_")
        # print(file_parts)
        date = file_parts[0]
        stim_type = file_parts[1]
        # print("stim type(before...): ", stim_type)
        if len(frlist) == 0:
            frlist = [1]

        if stim_type in ["Click", "click", "Tone"]:
            n = 0

            for i, db in enumerate(dblist):
                for j, fr in enumerate(frlist):
                    print("HPF, LPF: ", high_pass_filter, low_pass_filter)
                    x, tb, sample_rate = self.average_within_traces(
                        fd,
                        n,
                        protocol,
                        date,
                        high_pass_filter=high_pass_filter,
                        low_pass_filter=low_pass_filter,
                    )
                    if i == 0 and j == 0:
                        abr_data = np.zeros((len(dblist), len(frlist), len(x)))
                    abr_data[i, j] = x
                    n += 1
            # print("raaabrf:: sample rate: ", sample_rate, "max time: ", np.max(tb))

        # print("read and average py3abr: stim type:: ", stim_type)
        if stim_type in ["interleaved"]:  # a form of Tones...
            n = 0
            abr_data, tb, sample_rate, stim = self.average_across_traces(
                fd, n, protocol, date, high_pass_filter=high_pass_filter
            )
            # print(len(frlist), len(dblist))
            abr_data = abr_data.reshape(len(dblist), len(frlist), -1)
            # print("calculated new tb")

        # print("pyabr3_data shape: ", abr_data.shape)
        # print("pyabr3 db list: ", dblist)
        if dblist[0] > dblist[-1]:  # need to reverse data order and dblist order
            dblist = dblist[::-1]
            abr_data = np.flip(abr_data, axis=0)
        # print("sample_rate: ", sample_rate, 1.0 / np.mean(np.diff(tb)))
        amp_gain = 1e4

        metadata = {
            "type": "pyabr3",
            "filename": filename,
            "subject_id": subject_id,
            "age": age,
            "sex": sex,
            "stim_type": stim_type,
            "stimuli": {"dblist": dblist, "freqlist": frlist},
            "amplifier_gain": amp_gain,
            "strain": strain,
            "weight": weight,
            "genotype": genotype,
            "record_frequency": sample_rate,
            "highpass": high_pass_filter,
        }

        if pdf is not None:
            self.plot_abrs(
                abr_data=abr_data,
                tb=tb,
                waveana=waveana,
                stim_type=stim_type,
                scale=scale,
                dblist=dblist,
                frlist=frlist,
                metadata=metadata,
                maxdur=maxdur,
                highpass=300.0,
                use_matplotlib=True,
                pdf=pdf,
            )
        # print("max time base rad and average: ", np.max(tb))
        return abr_data, tb, metadata

