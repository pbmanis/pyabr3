import datetime
import re
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pyqtgraph as pg
import scipy
from matplotlib import pyplot as plt
from pylibrary.tools import cprint as CP
from pyqtgraph.Qt import QtCore, QtGui

import parse_ages
from src import filter_util as filter_util

from src import read_abr as RA

re_click = re.compile(r"[\d]{8}-[\d]{4}-[pn]{1}.txt$", re.IGNORECASE)
re_spl = re.compile(r"[\d]{8}-[\d]{4}-SPL.txt$", re.IGNORECASE)
re_khz = re.compile(r"[\d]{8}-[\d]{4}-kHz.txt$", re.IGNORECASE)
re_tone_p = re.compile(r"([\d]{8}-[\d]{4})-[p]{1}-([\d]{3,5}.[\d]{3}).txt$", re.IGNORECASE)
re_tone_n = re.compile(r"([\d]{8}-[\d]{4})-[n]{1}-([\d]{3,5}.[\d]{3}).txt$", re.IGNORECASE)
re_tone_pn = re.compile(r"([\d]{8}-[\d]{4})-[pn]{1}-([\d]{3,5}.[\d]{3}).txt$", re.IGNORECASE)

""" Note: The SPL files are sometimes missing. 
Here we substitute in a "standard" spl array of 20-90 dB SPL in 5 dB steps for clicks,
and 20-90 dB SPL in 10 dB steps for tones, IF the file is not found.

ABR4 has a 1 ms delay from recording onset to start of stimulus.


"""


class find_datasets:
    pass


class READ_ABR4:
    def __init__(self):
        self.sample_freq = None  # default for matlab ABR4 program (interpolzted)
        self.amplifier_gain = 1e4  # default for ABR4 recording (set externally on Grass P511J)
        self.FILT = filter_util.Utility()
        self.invert = False
        self.frequencies = []

    def read_dataset(
        self,
        subject: str,
        datapath: Union[Path, str],  # path to the data (.txt files are in this directory)
        datatype: str = "Click",
        sample_frequency: float = 100000.0,
        subdir: str = "Tone",  # or "Clicks"
        run: str = "20220518-1624",
        highpass: Union[float, None] = None,
        lowpass: Union[float, None] = None,
        lineterm="\r",
        fold: bool = False,  # fold flag - for when there are 2 responses in a waveform.
    ):
        """
        Read a dataset, combining the positive and negative recordings,
        which are stored in separate files on disk. The waveforms are averaged
        which helps to minimize the CAP contribution.
        The waveforms are then bandpass filtered to remove the low frequency
        "rumble" and excess high-frequency noise.

        Parameters
        ----------
        run: base run name (str)
        lineterm: str
            line terminator used for this file set

        Returns
        -------
        waveform
            Waveform, as a nxm array, where n is the number of intensities,
            and m is the length of each waveform
        timebase

        """
        metadata = None
        # handle missing files.

        if datatype not in ["Click", "Tone"]:
            raise ValueError(f"Unknown datatype: {datatype:s}")
        print("READ_ABR4.read_dataset filters: ", highpass, lowpass)
        if datatype == "Click":
            # find click runs for this subject:
            click_runs = self.find_click_files([datapath], subject="", subdir="")
            # print("click runs: ", click_runs)
            if len(click_runs) == 0:
                return None, None
            for run in click_runs:
                waves, tb = self.get_clicks(
                    datapath,
                    subject,
                    subdir,
                    click_runs[run],
                    sample_freq=sample_frequency,
                    highpass=highpass,
                    lowpass=lowpass,
                    fold=fold,
                )
                self.sample_freq = 1.0 / np.mean(np.diff(tb))
                print("self sample freq; ", self.sample_freq)

        elif datatype == "Tone":
            self.sample_freq = sample_frequency  # tone data is sampled at 100 kHz
            tone_runs_avg, tb = self.find_tone_files(datapath, "", "", highpass=highpass, fold=fold)
            if tone_runs_avg is None:
                return None, None
            # now, re-organize the data into a 3d array for the main analysis program
            # The frequency columns must be sorted and aligne
            self.frlist = list(tone_runs_avg.keys())
            frequencies = list(set([f[0] for f in self.frlist]))  # from the keys.
            waves = np.zeros((len(self.dblist), len(frequencies), len(tb)))
            empty_trace = np.zeros(len(tb))
            n = 0
            # print("frequencies, dblist: ", frequencies, self.dblist)
            try:
                for ifr, freq in enumerate(sorted(frequencies)):
                    for idb, db in enumerate(self.dblist):
                        if len(tone_runs_avg[(freq, db)]) == 0:
                            waves[idb, ifr, :] = empty_trace
                        else:
                            waves[idb, ifr, :] = tone_runs_avg[(freq, db)][0].squeeze()
                        n += 1
            except:
                raise ValueError(
                    f"Error in reading tone data: {n:d}, {datapath!s}, {subject!s}, {freq}, {db}"
                )
                # print("readdataset wave shape: ", waves.shape)

        if metadata is None:
            metadata = {
                "type": "ABR4",
                "filename": str(self.filename),
                "subject_id": "no id",
                "subject": subject,
                "age": 0,
                "sex": "ND",
                "stim_type": datatype,
                "stimuli": {"dblist": self.dblist, "freqlist": sorted(self.frequencies)},
                "amplifier_gain": 1.0,  # already accounted for in the dataR.amplifier_gain,
                "scale": "V",
                "V_stretch": 0.5,
                "strain": "ND",
                "weight": 0.0,
                "genotype": "ND",
                "record_frequency": self.sample_freq,
                "highpass": highpass,
                "lowpass": lowpass,
            }
        else:

            metadata = {
                "type": "ABR4",
                "filename": str(self.filename),
                "subject_id": "no id",
                "subject": subject,
                "age": 0,
                "sex": "ND",
                "amplifier_gain": 1.0,  # already accounted for in the data: amplifier_gain,
                "stim_type": datatype,
                "stimuli": {"dblist": self.dblist, "freqlist": sorted(self.frequencies)},
                "scale": "V",
                "V_stretch": 0.5,
                "strain": "ND",
                "weight": 0.0,
                "genotype": "ND",
                "record_frequency": self.sample_freq,
                "highpass": highpass,
                "lowpass": lowpass,
            }
        # before returning, sort the waveforms in ascending order of level.
        dbs = metadata["stimuli"]["dblist"]
        print("read dataset wave shape: ", waves.shape)
        print("original dbs: ", dbs)
        if dbs[-1] < dbs[0]:
            print("would need to sort data)")
            waves = np.flip(waves, axis=0)
            metadata["stimuli"]["dblist"] = dbs[::-1]
            exit()
        return waves, tb, metadata

    def find_click_files(self, datapath, subject, subdir):
        print(datapath, subject, subdir)
        for dpath in datapath:
            directory = Path(dpath, subject, subdir)
            if not directory.is_dir():
                directory = directory.parent

            print("Data found: ", str(directory))
            datafiles = list(directory.rglob(f"*.txt"))
            click_runs = {}
            self.filename = directory
            for df in datafiles:
                m = re.match(re_click, df.name)
                if m is not None:
                    click_runs[df.name[:13]] = {
                        "p": f"{df.name[:14]}p.txt",
                        "n": f"{df.name[:14]}n.txt",
                        "SPL": f"{df.name[:14]}SPL.txt",
                    }
                    df_datetime = f"{df.name[:14]}SPL.txt"
                    # print(df_datetime, re.match(re_spl, df_datetime))
            # print("Found runs: ", click_runs)
            return click_runs

    def find_tone_files(
        self, datapath, subject, subdir, highpass: Union[float, None] = None, fold: bool = False
    ):
        """find_tone_files find the tone files in this directory
        The result is a dictionary whose keys are the frequency and intensity pairs,
        and whose values are a list of the csv (txt) files for the negative and positive
        polarity data, and the SPL and kHz files. The SPL and kHz files are not used in this
        Parameters
        ----------
        datapath : _type_
            _description_
        subject : _type_
            _description_
        subdir : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        directory = Path(datapath, subject, subdir)
        if not directory.is_dir():
            print(f"Directory: {str(directory):s} was not found")
            exit()

        print("Directory for data found: ", str(directory))
        datafiles = list(directory.rglob(f"*.txt"))
        self.filename = directory
        # first find all of the runs with at least one of an n or p files associated with tone stimuli.
        run_times = []
        for df in datafiles:
            runtime = df.name[:13]
            if runtime not in run_times:
                run_times.append(runtime)

        # get the data into two dictionaries, one for each polarity.
        # the keys for each dictionary are tuples of (db, freq)
        tone_runs_n = {}  # negative polirity
        tone_runs_p = {}  # positive polirity
        # the traces will be appened as a list of each db, freq pairing
        # This allows us to average traaces when multpile runs were done for the same
        # stimulus condition.
        print("tone run_times: ", run_times)
        frequencies = []
        spls = []
        # first generate ALL of the frequencis/spl pairs in the runs for this subject
        for run_time in run_times:
            run_files = [datafile for datafile in datafiles if datafile.name.startswith(run_time)]
            spl_file = Path(run_files[0].parent, f"{run_files[0].name[:14]}SPL.txt")
            if not spl_file.is_file():
                spl_list = np.linspace(20, 90, endpoint=True, num=8)
            else:
                spl_list = pd.read_csv(spl_file, header=None, usecols=[0]).values.squeeze()
                if len(spl_list) == 0:
                    spl_list = pd.read_csv(spl_file, header=None).values.squeeze()
            khz_file = Path(run_files[0].parent, f"{run_files[0].name[:14]}kHz.txt")
            if not khz_file.is_file():
                continue  # probably a click group, so skip the freqs.
            # old
            khz_list_1 = pd.read_csv(
                khz_file, sep=r"[ \t\r\n]+", header=None, engine="python"
            ).values.squeeze()
            # sometimes in rows without a return
            # khz_list_2 = pd.read_csv(khz_file, sep=r"[ \t]+", header=None, engine="python").values.squeeze()
            # # print(type(khz_list_1), type(khz_list_2), khz_list_1.shape, khz_list_1.ndim, khz_list_2.ndim, khz_list_1.shape, khz_list_2.shape)
            if khz_list_1.ndim == 0:
                khz_list_1 = [khz_list_1]
            # elif khz_list_1.ndim == 1:
            #     khz_list = khz_list_1
            # else:
            #     khz_list = [khz_list_2]
            # print("\nKhz file: ", khz_file.name)
            # print("    Khz list_1: ", khz_list_1)
            # # print("    Khz list_2: ", khz_list_2)
            khz_list = [float(k) for k in khz_list_1 if not np.isnan(k)]
            # print("    Khz list: ", khz_list)
            for khz in khz_list:
                if int(float(khz)) not in frequencies:
                    frequencies.append(int(float(khz)))
                for spl in spl_list:
                    run_key = (int(float(khz)), int((float(spl))))
                    if run_key not in tone_runs_n.keys():
                        tone_runs_n[run_key] = []
                    if run_key not in tone_runs_p.keys():
                        tone_runs_p[run_key] = []
        if len(datafiles) == 0:
            return None, None

        # now read the data into the dictionaries
        # and assign the data to the appropriate key in the dictionary

        for datafile in datafiles:
            match_tonefile = re.match(re_tone_pn, datafile.name)
            if match_tonefile is None:
                continue  # not a tone file
            filename = match_tonefile.group()
            # get polarity group
            if "-p-" in filename:
                target = "p"
            elif "-n-" in filename:
                target = "n"
            else:
                raise ValueError(f"Unknown target: {filename}")

            # get the frequency for this data file.
            # The data structure is such that EACH file holds data from
            # only one frequency (but all spls at that frequency)
            frequency = match_tonefile.group(2)
            # read the data
            data = pd.io.parsers.read_csv(
                datafile,
                sep=r"[\t ]+",
                lineterminator=r"[\r\n]+",  # lineterm,
                skip_blank_lines=True,
                header=None,
                names=spl_list,
                engine="python",
            )
            data = data.values

            if np.isnan(data[-1, :]).any():
                data[-1, :] = data[-2, :]

            data = np.array(data).T
            for ispl, spl in enumerate(spl_list):
                spl_asint = int(float(spl))
                if spl_asint not in spls:
                    spls.append(spl_asint)
                key = (int(float(frequency)), spl_asint)
                if highpass is not None:
                    data[ispl] = self.FILT.SignalFilter_HPFButter(
                        data[ispl],
                        HPF=highpass,
                        samplefreq=self.sample_freq,
                        NPole=4,
                        bidir=True,
                    )
                if key not in tone_runs_p.keys():
                    tone_runs_p[key] = []
                if key not in tone_runs_n.keys():
                    tone_runs_n[key] = []
                    # raise ValueError(f"Key {key} not found in tone_runs_p for datafile: {str(datafile):s}")
                if target == "p":
                    tone_runs_p[key].append(data[ispl])
                elif target == "n":
                    tone_runs_n[key].append(data[ispl])
                else:
                    raise ValueError(f"Unknown target: {target}")
        # exit()
        self.tone_runs_p = tone_runs_p
        self.tone_runs_n = tone_runs_n
        self.frequencies = frequencies
        if frequencies is None or len(frequencies) == 0:
            return None, None
        self.dblist = spls
        avg = self.abr4_average()
        tb = np.linspace(0, data.shape[1] * (1.0 / self.sample_freq), data.shape[1])
        return avg, tb

    def abr4_average(self):
        frequencies = sorted(self.frequencies)
        # print("frequencies: ", frequencies)
        spls = sorted(self.dblist)

        tone_runs_avg = {}
        for ifr, frdb in enumerate(frequencies):
            for idb, spl in enumerate(spls[-1::-1]):
                key = (frdb, spl)
                if len(self.tone_runs_p[key]) > 0:  # data in the keys
                    if len(self.tone_runs_p[key]) > 1:
                        self.tone_runs_p[key] = np.array(self.tone_runs_p[key])
                        self.tone_runs_p[key] = np.mean(self.tone_runs_p[key], axis=0)

                    if len(self.tone_runs_n[key]) > 1:
                        self.tone_runs_n[key] = np.array(self.tone_runs_n[key])
                        self.tone_runs_n[key] = np.mean(self.tone_runs_n[key], axis=0)

                d_avg = np.mean([self.tone_runs_p[key], self.tone_runs_n[key]], axis=0)
                tone_runs_avg[key] = d_avg
        self.tone_runs_avg = tone_runs_avg
        return tone_runs_avg

    def abr4_plot(self, waves: np.ndarray, tb: np.ndarray, frequencies, spls):
        """abr4_plot Plot the tone_runs (could be averaged, or p or n runs)
        This is mostly for testing to be sure the data is getting where it needs to be

        Returns
        -------
        _type_
            _description_
        """
        from matplotlib import pyplot as mpl
        from pylibrary.plotting import plothelpers as PH

        frequencies = sorted(frequencies)
        spls = sorted(spls)
        P = PH.regular_grid(
            len(spls),
            len(frequencies),
            order="rowsfirst",
            figsize=(12, 8),
            verticalspacing=0.03,
            horizontalspacing=0.01,
            margins={"leftmargin": 0.1, "rightmargin": 0.1, "topmargin": 0.1, "bottommargin": 0.1},
        )

        v_min = 0
        v_max = 0

        for ifr, frdb in enumerate(frequencies):
            for idb, spl in enumerate(spls):
                PH.noaxes(P.axarr[idb, ifr])
                P.axarr[idb, ifr].plot(  # convert to microvolts
                    t, np.array(waves[len(spls) - idb - 1, ifr]) * 1e6, linewidth=0.5, color="k"
                )
                PH.referenceline(P.axarr[idb, ifr], 0.0, color="grey", linestyle="--")
                vax = P.axarr[idb, ifr].get_ylim()
                v_min = min(v_min, vax[0])
                v_max = max(v_max, vax[1])
                P.axarr[idb, ifr].set_title(f"{frdb:d} kHz, {spls[len(spls)-idb-1]:d} dB SPL")
        v_scale = np.max([-v_min, v_max])
        for ifr, frdb in enumerate(frequencies):
            for idb, spl in enumerate(spls[-1::-1]):
                P.axarr[idb, ifr].set_ylim(-v_scale, v_scale)
        mpl.text(
            0.96,
            0.01,
            s=datetime.datetime.now(),
            fontsize=6,
            ha="right",
            transform=P.figure_handle.transFigure,
        )

        mpl.show()

    def get_clicks(
        self,
        datapath,
        subject,
        subdir,
        run,
        sample_freq: float = None,
        highpass: Union[float, None] = None,
        lowpass: Union[float, None] = None,
        fold: bool = False,
    ):
        # do a quick check to see if there are subdirectories for tones and clicks:
        # print("subdir: ", subdir)
        if datapath.name == "Click":
            print("run (spl file for click data): ", run["SPL"])
        # if subdir == "":
        #     test_dir = Path(datapath, subject, "Clicks")
        #     if test_dir.is_dir():
        #         subdir = "Clicks"
        #     test_dir = Path(datapath, subject, "Click")
        #     if test_dir.is_dir():
        #         subdir = "Click"

        spl_file = Path(datapath, run["SPL"])
        # print("spl_file: ", spl_file, spl_file.is_file(), subdir)
        if not spl_file.is_file():
            subdir = ""
            # spl_file = Path(datapath, subject, subdir, run["SPL"])
            splf_ile = Path(datapath, run["SPL"])
            if not spl_file.is_file():
                CP.cprint("r", f"    ABR_Reader.read_dataset: Did not find SPL file: {spl_file!s}")
                raise FileNotFoundError(f"Did not find SPL file: {spl_file!s}")

        pos_file = Path(datapath, run["p"])
        neg_file = Path(datapath, run["n"])
        # pos_file = Path(datapath, subject, subdir, run["p"])
        # neg_file = Path(datapath, subject, subdir, run["n"])
        if not pos_file.is_file():
            CP.cprint("r", f"    ABR_Reader.read_dataset: Did not find pos file: {pos_file!s}")
            return None, None
        if not neg_file.is_file():
            CP.cprint("r", f"    ABR_Reader.read_dataset: Did not find neg file: {neg_file!s}")
            return None, None
        CP.cprint("c", f"    ABR_Reader.read_dataset: Reading from: {pos_file!s} and {neg_file!s}")

        if not spl_file.is_file():  # missing spl file, substitute standard values
            spllist = np.linspace(20, 90, endpoint=True, num=15)
        else:
            spllist = pd.read_csv(spl_file, header=None).values.squeeze()

        self.dblist = spllist
        self.frlist = [0]
        posf = pd.io.parsers.read_csv(
            pos_file,
            sep=r"[\t ]+",
            lineterminator=r"[\r\n]+",  # lineterm,
            skip_blank_lines=True,
            header=None,
            names=spllist,
            engine="python",
        )
        negf = pd.io.parsers.read_csv(
            neg_file,
            sep=r"[\t ]+",
            lineterminator=r"[\r\n]+",
            skip_blank_lines=True,
            header=None,
            names=spllist,
            engine="python",
        )

        # fix Nan at end of array in posf data.
        # print(posf.columns)
        # print(posf[spllist[0]].values)
        for pc in posf.columns:
            pvals = posf[pc].values
            if np.isnan(pvals[-1]):
                pvals = pvals[-2]
                posf[pc] = pvals
        npoints = len(posf[spllist[0]])
        # print("Posf values: ", posf[spllist[0]].values)
        # exit()
        # print(f"Number of points: {npoints:d}")
        self.sample_freq = sample_freq
        tb = np.linspace(0, npoints * (1.0 / self.sample_freq), npoints)

        npoints = tb.shape[0]
        n2 = int(npoints / 2)
        #  waves are [#db, #fr, wave]
        if fold:
            waves = np.zeros((len(posf.columns), len(self.frlist), n2))
            tb = tb[:n2]
        else:
            waves = np.zeros((len(posf.columns), len(self.frlist), npoints))

        # app = pg.mkQApp("summarize abr4 output")
        # win = pg.GraphicsLayoutWidget(show=True, title="ABR Data Plot")
        # win.resize(800, 600)
        # win.setWindowTitle(f"awwww")
        # symbols = ["o", "s", "t", "d", "+", "x"]
        # win.setBackground("w")
        # pl = win.addPlot(title=f"abr")

        if fold:
            for j, fr in enumerate(self.frlist):
                for i1, cn in enumerate(posf.columns):
                    i = len(posf.columns) - i1 - 1
                    waves[i, j] = (
                        negf[cn].values[:n2]
                        + negf[cn].values[n2:]
                        + posf[cn].values[:n2]
                        + posf[cn].values[n2:]
                    ) / 4.0
        else:
            for j, fr in enumerate(self.frlist):
                for i1, cn in enumerate(posf.columns):
                    i = len(posf.columns) - i1 - 1
                    waves[i, j] = posf[cn].values + negf[cn].values
        if highpass is not None:
            # print("get clicks: higpass, samplefreq: ", highpass, self.sample_freq)
            for i in range(waves.shape[0]):
                for j in range(waves.shape[1]):
                    waves[i, j] = self.FILT.SignalFilter_HPFButter(
                        waves[i, j], HPF=highpass, samplefreq=self.sample_freq, NPole=4, bidir=True
                    )
        if lowpass is not None:
            # print("get clicks: lowpass, samplefreq: ", lowpass, self.sample_freq)
            for i in range(waves.shape[0]):
                for j in range(waves.shape[1]):
                    waves[i, j] = self.FILT.SignalFilter_LPFButter(
                        waves[i, j], LPF=lowpass, samplefreq=self.sample_freq, NPole=4, bidir=True
                    )

        if self.invert:
            waves = -waves

        return waves, tb

    def plot_dataset(
        self,
        AR: object,
        datatype: str,
        subject: str,
        topdir: Union[str, Path],
        subdir: Union[str, Path],
        highpass: Union[float, None],
        maxdur: float,
        hide_treatment: bool = False,
        metadata: Union[dict, None] = None,
        pdf=None,
    ):
        """plot_dataset
        Plot the dataset for the given subject and stimulus type

        Parameters
        ----------
        AR : read_abr instance
            _description_
        datatype : str
            "click" or "tone"
            _description_
        subject : str
            subject name (file)
        topdir : Path, str
            path to the data
        subdir : str,
            subdirectory for the data
        highpass : Union[None, float]
            High pass filter, Hz
        maxdur : float
            plot duration in milliseconds
        pdf : object, optional
            a pdf object for pdfpages by default None
        """
        w, t, metadata = self.read_dataset(
            subject=subject,
            datapath=topdir,
            subdir=subdir,
            datatype=datatype,
            highpass=highpass,
        )
        if w is None:  # no data for this stimulus type
            return
        # average = R.abr4_average()
        # w, t = R.read_dataset(fn, datatype="tones", subject ="")
        # R.plot_waveforms(stim_type="click", waveform=w, tb=t, fn=fn)
        # print("w, t, db, fr: ", w.shape, t.shape, len(R.dblist), len(R.frlist))
        # print("db: ", R.dblist)
        # print(fn)
        # print("R.record freq: ", R.sample_freq)

        # R.abr4_plot(waves=w, tb=t, frequencies = R.frequencies, spls=R.dblist)
        # exit()
        AR.plot_abrs(
            acquisition="ABR4",
            abr_data=w,
            tb=t,
            stim_type=datatype,
            scale="V",
            V_stretch=0.5,
            dblist=self.dblist,
            frlist=sorted(self.frequencies),
            metadata=metadata,
            maxdur=maxdur,
            use_matplotlib=True,
            live_plot=False,
            pdf=pdf,
        )


def make_excel(subject_list):
    # filenames will be of form: "CBA_F_N000_p30_NT"
    df = pd.DataFrame(columns=["Subject", "Age", "sex", "treatment"])
    for subject in subject_list:
        subj_data = subject.name.split("_")
        subj_name = f"{subj_data[0]:s}_{subj_data[2]:s}"
        subj_age = subj_data[3]
        subj_age = parse_ages.ISO8601_age(subj_age)
        subj_treat = subj_data[4]
        subj_sex = subj_data[1]
        df_subject = pd.Series(
            {"Subject": subj_name, "Age": subj_age, "treatment": subj_treat, "sex": subj_sex}
        )
        df = pd.concat([df, df_subject.to_frame().T], ignore_index=True)
    df.to_excel("CBA_ABR4_subjects.xlsx")


if __name__ == "__main__":

    from matplotlib.backends.backend_pdf import PdfPages

    AR = RA.AnalyzeABR()
    ABR4 = READ_ABR4()
    # Load the data
    # fn = "/Volumes/Pegasus_004/ManisLab_Data3/abr_data/Reggie_E/B2S_Math1cre_M_10-8_P36_WT/"
    # fn = Path(fn)

    datatype = "tone"
    if datatype == "tone":
        subdir = "Tones"
    elif datatype == "click":
        subdir = "Click"
    else:
        raise ValueError(f"Unknown datatype: {datatype:s}")
    subdir = ""
    subject = "CBA_M_N001_P21_NT"
    highpass = 300

    maxdur = 14.0
    topdir = "/Volumes/Pegasus_002/ManisLab_Data3/abr_data/Reggie_CBA_Age"  # CBA_18_Months"
    # fn = Path(topdir)
    # if not fn.is_dir():
    #     raise ValueError(f"File: {str(fn):s} not found")
    #     exit()

    dirpath = Path(topdir)
    if not dirpath.is_dir():
        raise FileNotFoundError(f"Directory: {str(dirpath):s} not found")
    subject_list = list(dirpath.glob("CBA*"))

    make_excel(subject_list)

    with PdfPages("ABR4_dataset_clicks.pdf") as pdf:

        for subject in subject_list:
            print("Subject: ", subject)

            ABR4.plot_dataset(
                AR,
                datatype=datatype,
                subject=subject,
                topdir=topdir,
                subdir=subdir,
                highpass=highpass,
                maxdur=maxdur,
                pdf=pdf,
            )
