import datetime
import pathlib
import pickle
import platform
import re
from pathlib import Path
from string import ascii_letters
from typing import Union

import numpy as np
import pandas as pd
import pyqtgraph as pg
import scipy.stats
import seaborn as sns
from matplotlib import pyplot as mpl
from pylibrary.plotting import styler as ST
from pylibrary.tools import cprint as CP

# import ephys.tools.get_configuration as GETCONFIG
import get_configuration as GETCONFIG
import plothelpers as mpl_PH
import src.analyze_abr
import src.analyzer
from src import filter_util as filter_util
from src import fit_thresholds
from src import parse_ages as PA
from src import read_abr4 as read_abr4
from src import read_calibration as read_calibration
import src.abr_regexs as REX


use_matplotlib = True
from matplotlib.backends.backend_pdf import PdfPages
from pylibrary.plotting import plothelpers as PH

# Check the operating system and set the appropriate path type
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

WaveAnalyzer = src.analyzer.Analyzer()
AnalyzeABR = src.analyze_abr.AnalyzeABR


def plot_click_stack(
    AR, ABR4, filename, directory_name: str = None, ax: Union[object, None] = None
):
    # AR = AnalyzeABR()
    # ABR4 = read_abr4.READ_ABR4()
    maxdur = AR.experiment["ABR_settings"]["maxdur"]
    HPF = AR.experiment["ABR_settings"]["HPF"]
    LPF = AR.experiment["ABR_settings"]["LPF"]
    stack_increment = AR.experiment["ABR_settings"]["stack_increment"]

    pdf = None
    if ax is None and pdf is None:
        raise ValueError("Must provide either an axis or a pdf file")

    print("Click file: ", filename)
    pfiles = list(Path(filename).glob("*_click_*.p"))
    if len(pfiles) > 0:
        AR.read_and_average_abr_files(
            filename=str(pfiles[0]), high_pass_filter=HPF, maxdur=maxdur, pdf=pdf
        )
    else:
        f = Path(filename)
        ABR4.plot_dataset(
            AR,
            datatype="click",
            subject=f.parent,
            topdir=directory_name,
            subdir=f.name,
            highpass=HPF,
            maxdur=maxdur,
            pdf=pdf,
        )


def do_directory(
    AR: AnalyzeABR,
    ABR4: object,
    directory_name: Union[Path, str],
    output_file: Union[Path, str],
    subject_prefix: str = "CBA",
    hide_treatment: bool = False,
):

    AR.set_hide_treatment(hide_treatment)

    subjs = Path(directory_name).glob(f"{subject_prefix}*")
    with PdfPages(output_file) as pdf:
        for subj in subjs:
            if not subj.is_dir() or subj.name.startswith("."):
                continue
            fns = subj.glob("*")
            for f in fns:
                if fname.startswith("Click"):
                    plot_click_stack(AR, ABR4, f, directory_name, ax=None)
                elif fname.startswith("Tone"):
                    print("Tone file: ", f)
                    ABR4.plot_dataset(
                        AR,
                        datatype="tone",
                        subject=f.parent,
                        topdir=directory_name,
                        subdir=f.name,
                        pdf=pdf,
                    )
                elif fname.startswith("Interleaved"):
                    print("Interleaved file: ", f)
                    files = list(Path(f).glob("*.p"))
                    print(f, "\n     # interleaved files: ", len(files))

                    AR.read_and_average_abr_files(
                        filename=str(files[0]),
                        pdf=pdf,
                    )
                else:
                    raise ValueError(f"File {f} for data file {fname:s} not recognized")


def get_treatment(subject, re=None):
    """get_treatment split off the treatment information from the subject name/file

    Parameters
    ----------
    subject : str
        directory name for data for this subject

    Returns
    -------
    The last underscore separated part of the subject name, which is the treatment
        treatment
    """
    m = REX.re_subject.match(subject["name"])
    subparts = subject["name"].split("_")
    return subparts[-1]


def get_age(subject):
    """get_age split off the treatment information from the subject name/file
        age_categories:
            Preweaning: [7, 20]
            Pubescent: [21, 49]
            Young Adult: [50, 179]
            Mature Adult: [180, 364]
            Old Adult: [365, 1200]

    Parameters
    ----------
    subject : str
        directory name for data for this subject
    position: int
        position of the age in the subject name
    Returns
    -------
    The last underscore separated part of the subject name, which is the treatment
        treatment
    """
    subject_name = subject["name"]

    m = REX.re_subject.match(subject_name)
    if m is None:
        raise ValueError(f"Cannot match subject information from {subject_name}")
    age = m.group("age")
    i_age = PA.age_as_int(PA.ISO8601_age(age))
    if i_age <= 20:
        age_str = "Preweaning"
    elif 21 <= i_age <= 49:
        age_str = "Pubescent"
    elif 50 < i_age <= 179:
        age_str = "Young Adult"
    elif 180 < i_age <= 364:
        age_str = "Mature Adult"
    elif i_age > 365:
        age_str = "Old Adult"
    else:
        print("age not found: ", i_age, age)
        raise ValueError(f"Age {age} not found in {subject_name}")
    return age_str


def get_categories(subjects, categorize="treatment"):
    """get_categories Find all of the categories in all of the subjects

    Parameters
    ----------
    subjects : list
        list of subject directories

    Returns
    -------
    list
        list of unique categories across all subjects
    """
    categories = []
    for subj in subjects:
        print("subj: ", subj)
        # if subj.name.startswith("."):
        # continue
        if categorize == "treatment":
            category = get_treatment(subj)
        elif categorize == "age":
            category = get_age(subj)
        # print("treatment: ", treatment)
        if category not in categories:
            categories.append(category)

    return categories


def get_analyzed_click_data(
    filename,
    AR,
    ABR4,
    subj,
    HPF: Union[float, None] = None,
    LPF: Union[float, None] = None,
    maxdur: float = 10.0,
    scale: float = 1.0,
    invert: bool = False,
):
    pfiles = list(Path(filename).glob("*_click_*.p"))
    # f, ax = mpl.subplots(1, 1)  # for a quick view of the data
    if len(pfiles) > 0:
        waves, tb, metadata = AR.read_and_average_abr_files(
            filename=str(pfiles[0]), high_pass_filter=HPF, low_pass_filter=LPF,
            maxdur=maxdur, pdf=None
        )
        sym = "D"
        # print("metadata: ", metadata, pfiles[0])

        waves *= scale
        if invert:
            waves = -waves
            print("INVERTED!!!")
        else:
            print("NOT INVERTED!!!")
        print("get analyzed click data:: metadata: ", metadata)
        print("gacd: wave shape: ", waves.shape)
        print("gacd: Max time: ", np.max(tb), "sample rate: ", metadata["record_frequency"])

    else:
        # sample frequency for ABR4 depends on when the data was collected.
        # so we set it this way:
        subj_id = REX.re_subject.match(subj["name"]).group("subject")
        # print(subj_id)
        if subj_id[0] == "R":  # Ruili data, 100 kHz
            sample_frequency = 100000.0
        elif subj_id[0] == "N":  # Reggie data, 50 kHz
            sample_frequency = 50000.0
        else:
            sample_frequency = 50000.0
            # print("Sample frequency cannot be determined with the information provided")
            # raise ValueError(f"Subject ID {subj_id} not recognized")

        waves, tb, metadata = ABR4.read_dataset(
            subject=subj,
            datapath=subj["filename"],
            subdir=filename.name,
            datatype="Click",
            sample_frequency=sample_frequency,
            highpass=HPF,
            lowpass=LPF,
        )
        waves *= scale
        if invert:
            waves = -waves
            print("INVERTED!!!")
        else:
            print("NOT INVERTED!!!")
        # print("Read waves, shape= ", waves.shape)
        # print("metadata: ", metadata)
    dbs = metadata["stimuli"]["dblist"]
    # for i, db in enumerate(dbs):
    #     ax.plot(tb, waves[i, 0, :]/metadata["amplifier_gain"])
    # mpl.show()

    if waves is None:  # no data for this stimulus type
        return None
    # print(np.max(waves))

    WaveAnalyzer.analyze(timebase=tb, waves=waves[:, 0, :])
    dbs = metadata["stimuli"]["dblist"]
    # print(dir(WaveAnalyzer))
    # f, ax = mpl.subplots(1,1)
    # for i in range(WaveAnalyzer.waves.shape[0]):
    #     ax.plot(tb, WaveAnalyzer.waves[i, :]/metadata["amplifier_gain"],
    #     )

    # mpl.show()
    # exit()
    print("  get analyzed click data: dbs: ", dbs)
    return WaveAnalyzer, dbs, metadata


def get_analyzed_tone_data(
    filename,
    AR,
    ABR4,
    subj,
    HPF: Union[float, None] = 100.0,
    LPF: Union[float, None] = None,
    maxdur: float = 10.0,
    scale: float = 1.0,
    invert: bool = False,
):
    pfiles = list(Path(filename).glob("*_interleaved_plateau_*.p"))
    # print("pfiles: ", len(pfiles))
    if len(pfiles) > 0:
        waves, tb, metadata = AR.read_and_average_abr_files(
            filename=str(pfiles[0]), high_pass_filter=HPF, 
            low_pass_filter=LPF, maxdur=maxdur, pdf=None
        )
        sym = "D"
        # print("Read waves, shape= ", waves.shape)
        # print("metadata: ", metadata)

    else:
        # sample frequency depends on when the data was collected.
        # so we set it this way:
        subj_id = REX.re_subject.match(subj["name"]).group("subject")

        waves, tb, metadata = ABR4.read_dataset(
            subject=subj["name"],
            datapath=subj["filename"],
            subdir=filename.name,
            datatype="Tone",
            highpass=HPF,
            lowpass=LPF,
        )
        # print("Read waves, shape= ", waves.shape)
        # print("metadata: ", metadata)
    if invert:
        waves = -waves
        print("INVERTED!!!")
    # if passing to ABRA, here we should combine the waveforms into the CSV file format that they want..
    # print("subject : ", subj)
    export_for_abra_tones(subj, waves, metadata)
    # print("metadata: ", metadata)
    return (
        WaveAnalyzer,
        metadata["stimuli"]["dblist"],
        metadata,
        metadata["stimuli"]["dblist"],
        metadata["stimuli"]["freqlist"],
    )
    #############################

    if waves is None:  # no data for this stimulus type
        return None
    # print(waves.shape)
    # print("metadata: ", metadata)
    # f, ax = mpl.subplots(1,1)
    subject_threshold = []
    subject_frequencies = []
    # print(metadata['stimuli'])
    if "frlist" in metadata["stimuli"]:
        freq_list = metadata["stimuli"]["frlist"]
    elif "freqlist" in metadata["stimuli"]:
        freq_list = metadata["stimuli"]["freqlist"]
    else:
        raise ValueError("No frequency list in metadata", metadata["stimuli"])

    c = mpl.colormaps["rainbow"](np.linspace(0, 1, waves.shape[1]))

    for ifr in range(waves.shape[1]):
        WaveAnalyzer.analyze(timebase=tb, waves=waves[:, ifr, :])
        # WaveAnalyzer.ppio[np.isnan(WaveAnalyzer.ppio)] = 0
        # print("waves analyzed: ", WaveAnalyzer.ppio)
        # print("baseline: ", WaveAnalyzer.rms_baseline)
        valid_dbs = range(
            waves.shape[0]
        )  # use this for ppio: #np.where(np.isfinite(WaveAnalyzer.ppio))[0]
        dbx = np.array(metadata["stimuli"]["dblist"])[valid_dbs]
        spec_power = WaveAnalyzer.specpower(
            fr=[750, 1300], win=[1e-3, 8e-3], spls=metadata["stimuli"]["dblist"]
        )
        spec_noise = WaveAnalyzer.specpower(
            fr=[750, 1300],
            win=[1e-3, 8e-3],
            spls=metadata["stimuli"]["dblist"],
            get_reference=True,
            i_reference=np.argmin(metadata["stimuli"]["dblist"]),
        )  # use the lowest stimulus level to get the reference spectral baseline
        # print("valid)", valid)
        # ax.plot(metadata['stimuli']['dblist'], WaveAnalyzer.ppio, color=c[ifr], marker='o', linewidth=0, markersize=3, label=f"{freq_list[ifr]:.1f} Hz")
        # print("Waveana rms baseline: ", WaveAnalyzer.rms_baseline)
        threshold_value, threshold_index, fit = fit_thresholds.fit_thresholds(
            dbx,
            WaveAnalyzer.ppio[valid_dbs],
            # WaveAnalyzer.psdwindow[valid_dbs],
            # WaveAnalyzer.psdwindow[0],
            # WaveAnalyzer.reference_psd,
            WaveAnalyzer.rms_baseline,
            threshold_factor=AR.experiment["ABR_settings"]["tone_threshold_factor"],
        )
        frequency = float(freq_list[ifr])
        # mpl.plot(fit[0], fit[1], '-', color=c[ifr])
        threshold_value = round(threshold_value / 2) * 2  # round to nearest 2 dB
        subject_threshold.append(float(threshold_value))
        subject_frequencies.append(float(frequency))
    print("thresholds: ", subject_threshold, subject_frequencies)
    # ax.plot()
    # mpl.show()
    dbs = metadata["stimuli"]["dblist"]
    # ABR4.dblist
    print("  got analyzed tone data: dbs: ", dbs)
    return WaveAnalyzer, dbs, metadata, subject_threshold, subject_frequencies


def clean_subject_list(subjs):
    """clean_subject_list Remove cruft from the subject directory list.
    Things like hidden files, or unrecongized subdirectories, are deleted
    from the list.

    Parameters
    ----------
    subjs : list
        list of subject directoryies, Path objects
    """
    for subj in subjs:
        if not subj.is_dir():
            subjs.remove(subj)
        elif (
            subj.name.startswith(".")
            or subj.name.startswith(".DS_Store")
            or subj.name.endswith("Organization")
        ):
            subjs.remove(subj)
    return subjs


def _plot_io_data(
    waveana: object,
    dbs: list,
    V2uV: float,
    color: str,
    ax: object,
    symdata: list,
    threshold: float = None,
    threshold_index: int = None,
    dataset_name: str = None,
    add_label: bool = False,
) -> (np.ndarray, np.ndarray):

    # CP.cprint("g", f"len(ppio), dbs: {len(waveana.ppio):d}, {len(dbs):d}")
    if len(waveana.ppio) != len(dbs):
        return  # skip

    if add_label:
        label = dataset_name[1]
    else:
        label = None
    if dataset_name[1] == "Old Adult":
        mec = "#33333380"
        melw = 0.5
    else:
        mec = color
        melw = 0.5
    linew = 0.33
    dbs = np.array(dbs)
    iwaves = np.argwhere((20.0 <= dbs) & (dbs <= 90.0))
    # v = np.array([float(w[1]) for w in waveana.p1n1_amplitudes])[iwaves]
    ax.plot(
        dbs[iwaves],
        waveana.ppio[iwaves] * V2uV,
        color=mec,
        linestyle="-",  # f"{color:s}{sym:s}-",
        linewidth=linew,
        # marker=symdata[0],
        # markersize=symdata[1],
        # markerfacecolor=color,
        # markeredgecolor=mec,
        # markeredgewidth=melw,
        alpha=0.33,
        label=label,
    )

    # if threshold is not None:
    #     ax.plot(
    #         dbs[threshold_index],
    #         waveana.ppio[threshold_index] * V2uV,
    #         color=color,
    #         marker="o",
    #         markersize=4,
    #         markerfacecolor="w",
    #     )
    return dbs[iwaves], waveana.ppio[iwaves] * V2uV


def plot_click_stacks(
    AR: object,
    example_subjects: list,
    filename: Union[str, Path],
    metadata: dict = None,
    dbs: list = None,
    thresholds: list = None,
    waveana: object = None,
    axlist: list = None,
):

    if example_subjects is None:
        return
    if len(axlist) != len(example_subjects):
        raise ValueError("Number of axes must match number of example subjects")
    subject = filename.parts[-2]
    if subject in example_subjects:
        fnindex = example_subjects.index(subject)
        waveana.waves = np.expand_dims(waveana.waves, 0)
        AR.plot_abrs(
            abr_data=waveana.waves,
            tb=waveana.timebase,
            scale="V",
            waveana=waveana,
            acquisition="pyabr3",
            V_stretch=1.0,
            metadata=metadata,
            stim_type="Click",
            dblist=dbs,
            frlist=None,
            thresholds=thresholds,
            maxdur=AR.experiment["ABR_settings"]["maxdur"],
            ax_plot=axlist[fnindex],
            csv_filename=f"{AR.experiment['ABR_settings']['csv_filename']:s}_{subject:s}.csv",
        )
    else:
        # print("Subject: ", subject, "not in examples: ", example_subjects)
        return


def remap_xlabels(ax):
    remapper = {
        "Preweaning": "PW",
        "Pubescent": "PB",
        "Young Adult": "YA",
        "Mature Adult": "MA",
        "Old Adult": "OA",
    }
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i, label in enumerate(labels):
        if label in remapper:
            labels[i] = remapper[label]
    ax.set_xticks(range(len(labels)))  # need to do this to make sure ticks and labels are in sync
    ax.set_xticklabels(labels)


def set_gain_and_scale(subj, AR):
    # Set minimum latency and gain values from the configuratoin file
    if "ABR_parameters" not in AR.experiment.keys():
        raise ValueError("'ABR_parameters' missing from configuration file")

    scd = AR.experiment["ABR_parameters"]  # get the parameters dictionary
    if "default" not in scd.keys():
        raise ValueError("'default' values missing from ABR_parameters")

    scale = scd["default"]["scale"]
    invert = scd["default"]["invert"]
    min_lat = scd["default"]["minimum_latency"]
    # print("ABR keys: ", list(AR.experiment["ABR_parameters"].keys()))

    # if the full name is in the parameter list, use it instead of any defaults
    if subj["name"] in list(AR.experiment["ABR_parameters"].keys()):
        # CP.cprint("g", f"\nsubject name: {subj['name']!s} is in experiment ABR_parameters list")
        scale = scd[subj["name"]]["scale"]
        invert = scd[subj["name"]]["invert"]
        min_lat = scd[subj["name"]]["minimum_latency"]
        fit_index = scd[subj["name"]]["fit_index"]
        print("Set Specific parameters for subject: ", subj["name"])
        print("Scale: ", scale, "Invert: ", invert, "Minimum Latency: ", min_lat)
    else:
        # CP.cprint(
        #     "r",
        #     f"\nsubject name: {subj['name']!s} is NOT experiment ABR_parameters list - checking abbreviated versions",
        # )
        smatch = REX.re_subject.match(subj["name"])
        CP.cprint("m", f"SMATCH: {smatch}, {subj['name']:s}")

        if smatch["subject"] is not None:
            sname = smatch["subject"]
            if sname.startswith("N0"):
                scale = scd["N0"]["scale"]
                invert = scd["N0"]["invert"]
                min_lat = scd["N0"]["minimum_latency"]
                fit_index = scd["N0"]["fit_index"]
            elif sname.startswith("T0"):
                scale = scd["T0"]["scale"]
                invert = scd["T0"]["invert"]
                min_lat = scd["T0"]["minimum_latency"]
                fit_index = scd["T0"]["fit_index"]
            else:
                print("Using defaults for subject: ", sname)
                scale = 1
                invert = scd["default"]["invert"]
                min_lat = 0.0
                fit_index = 0
        else:
            raise ValueError(
                f"Subject name {subj['name']:s} not in configuration ABR_parameters dictionary"
            )
    return scale, invert, min_lat, fit_index


def check_file(
    AR: object, filename: Path, subj: dict, requested_stimulus_type: str, donefiles: list
) -> bool:

    stim_type = None

    fname = filename.name
    filematch = REX.re_splfile.match(fname)  # old ABR4 file type for click
    if filematch is not None:
        if filematch.group(1) in donefiles:
            return donefiles, None, False
        else:
            # print(f"    ABR4 File {fname:s} now being processed...")
            donefiles.append(filematch.group(1))
    else:
        # might be a new pyabr3 click file?
        if fname.endswith(".p"):
            if fname[:15] in donefiles:
                if fname.find("interleaved_plateau") >= 0:
                    stim_type = "Interleaved_plateau"
                else:
                    return donefiles, stim_type, False
            else:
                # print(f"    pyabr3 File {fname:s} has not been processed, continuing")
                donefiles.append(fname[:15])
    scale = 1

    CP.cprint("c", f"Testing for subject name: {subj['name']:s}")

    if (
        AR.experiment["ABR_subject_excludes"] is not None
        and subj["name"] in AR.experiment["ABR_subject_excludes"]
    ):
        # print("Exclusion files: ", AR.experiment["ABR_subject_excludes"].keys())
        CP.cprint("r", f"Excluding subject file: {subj['name']: s}")
        return donefiles, stim_type, False

    scale, invert, min_lat, fit_index = set_gain_and_scale(subj, AR)

    fname = filename.name.lower()
    # determine the stimulus type.
    print(
        "subject data type: ",
        subj["filename"].name,
        " and requested type: ",
        requested_stimulus_type,
    )
    if subj["filename"].name.lower() != requested_stimulus_type.lower():
        print(
            "failed to find matching file... ",
            subj["filename"].name,
            " requested: ",
            requested_stimulus_type,
        )
        return donefiles, stim_type, False
    # print(f"requested stimulus type: <{requested_stimulus_type}>, filename.name:  <{subj["filename"].name}>")
    # print("subj in tones?: ", subj["filename"].name in ["Tones", "Tone"])
    match requested_stimulus_type:
        case "Click":
            if (
                subj["filename"].name == "Click"
                or fname.endswith("-n.txt")
                or fname.endswith("-p.txt")
            ):
                stim_type = "Click"
        case "Tone" | "Tones":
            if subj["filename"].name in ["Tones", "Tone"]:
                #     or (
                #     subj["filename"].name == "Interleaved_plateau"
                # ):
                stim_type = "Tones"
        case "Interleaved_plateau" | "interleaved_plateau":
            if subj["filename"].name.lower() == "Interleaved_plateau".lower():
                stim_type = "Interleaved_plateau"

        case _:
            print(
                "Requested stimulus type not recognized",
                requested_stimulus_type,
                "sub filename: ",
                subj["filename"].name,
            )
            raise ValueError
    CP.cprint("g", f"    ***** checkfile: stim_type: {stim_type!s}, {fname:s}")
    if stim_type is None:
        raise ValueError(f"Stimulus type {stim_type!s} not recognized for subject {subj['name']!s}")
    return donefiles, stim_type, True


def export_for_abra_clicks(subj: dict, dbs: list, waveana: object, outputpath: Union[Path, str] = 'abra'):

    nsamps = 244
    new_freq = 24414.0625
    newx = np.linspace(0, 10.0, num=nsamps)
    t_delay = 0.0010  # ms
    oldx = np.arange(-t_delay, 0.010, 1.0 / waveana.sample_freq) * 1e3
    nold = len(oldx)
    ldb = []
    delay_n = int(0.0015 * new_freq)  # 2 ms
    # f, ax = mpl.subplots(1, 1, figsize=(12, 8))
    for i, db in enumerate(dbs):
        wave_interpolated = np.interp(newx, oldx, waveana.waves[i, :nold])
        row = {
            "Freq(Hz)": 100.0,
            "Level(dB)": float(dbs[i]),
            "Samp. Per.": float((1.0 / new_freq) * 1e6),
            "No. Samps.": nsamps,
        }
        # print("interpol: ", wave_interpolated.shape)
        for n in range(nsamps):
            row.update({f"{n:d}": wave_interpolated[n] * 1e6})
            # ax.plot(wave_interpolated)
        ldb.append(row)
        # print("Waveana waves: ", waveana.waves.shape, "sample freq: ", waveana.sample_freq)
    # mpl.show()
    df = pd.DataFrame.from_dict(ldb)
    abrap = Path("abra")
    if not abrap.exists():
        abrap.mkdir()
    outpath = Path(outputpath, 'Clicks', f"{subj['name']:s}_click_data.csv")
    df.to_csv(outpath, index=False)
    print("wrote to: ", outpath)


def export_for_abra_tones(subj: dict, waves: np.ndarray, metadata: dict, outputpath: Union[Path, str] = 'abra'):

    print("Exporting for ABRA tones: ", subj["subject"])
    nsamps = 244
    new_freq = 24414.0625
    newx = np.linspace(0, 10.0, num=nsamps)
    t_delay = 0.0010  # ms
    original_freq = metadata["record_frequency"]
    dbs = metadata["stimuli"]["dblist"]
    freqs = metadata["stimuli"]["freqlist"]
    oldx = np.arange(-t_delay, 0.010, 1.0 / original_freq) * 1e3
    nold = len(oldx)
    ldb = []
    delay_n = int(0.0015 * new_freq)  # 2 ms
    checkplot = False
    if checkplot:
        fig, ax = mpl.subplots(waves.shape[0], waves.shape[1], figsize=(12, 8))
    for i, db in enumerate(dbs):
        for j, fr in enumerate(freqs):
            wave_interpolated = (
                np.interp(newx, oldx, waves[i, j, :nold]) * 1e6 / metadata["amplifier_gain"]
            )
            row = {
                "Sub. ID": subj["subject"],
                "Freq(Hz)": float(fr),
                "Level(dB)": float(db),
                "Samp. Per.": float((1.0 / new_freq) * 1e6),
                "No. Samps.": nsamps,
            }
            for n in range(nsamps):
                row.update({f"{n:d}": wave_interpolated[n]})
            ldb.append(row)
            if checkplot:
                ax[i, j].plot(newx, wave_interpolated)  # , label=f"{fr:.1f} Hz, {db:.1f} dB")
                ax[i, j].set_ylim(-5, 5)
        # print("Waveana waves: ", waveana.waves.shape, "sample freq: ", waveana.sample_freq)
    if checkplot:
        mpl.show()
    df = pd.DataFrame.from_dict(ldb)
    abrap = Path("abra/Tones")
    if not abrap.exists():
        abrap.mkdir()
    outpath = Path(outputpath, 'Tones', f"{subj['name']:s}_tone_data.csv")
    df.to_csv(outpath, index=False)
    print("Wrote to: ", outpath)
    # exit()


def export_for_abra(
    subj: dict,
    requested_stimulus_type: str,
    AR: object,
    ABR4: object,
    outputpath: Union[Path, str] = 'abra'
) -> dict:
    print("\n\nCALLING EXPORT FOR ABRA")
    donefiles = []
    waveana = None  # in case the analysis fails or the dataset was excluded
    # print("requested file type: ", requested_stimulus_type)
    if requested_stimulus_type in ["Click", "Tones", "Tone", "Interleaved_plateau"]:
        fns = list(subj["filename"].glob("*"))
    else:
        raise ValueError("Requested stimulus type not recognized")

    fns = [f for f in fns if not f.name.startswith(".")]
    for filename in sorted(fns):
        # check to see if we have done this file already (esp. ABR4 files)
        fname = filename.name
        if fname.startswith("."):  # skip hidden files
            continue
        # print("Donefiles: ", donefiles, fname[:13], filename)
        processed = False
        for donefile in donefiles:
            # print("donefile: ", donefile, "fname: ", fname[:13])
            if donefile.startswith(fname[:13]):
                # print(f"Skipping file {fname:s} as it has already been processed")
                processed = True
                break

        # print("processed: ", processed)
        if processed:
            continue
        donefiles, stim_type, ok = check_file(
            AR=AR,
            filename=filename,
            subj=subj,
            requested_stimulus_type=requested_stimulus_type,
            donefiles=donefiles,
        )
        if not ok:
            print("NOT OK::::")
            continue
        # print("do one sub req ::  stim type: ", requested_stimulus_type)
        if requested_stimulus_type == "Click":
            filename = Path(filename).parent
            scale, invert, min_lat, fit_index = set_gain_and_scale(subj, AR)
            CP.cprint("g", f"    ***** do one sub: filename: {filename!s}")
            waveana, dbs, metadata = get_analyzed_click_data(
                filename,
                AR,
                ABR4,
                subj,
                HPF=AR.experiment["ABR_settings"]["HPF"],
                LPF=AR.experiment["ABR_settings"]["LPF"],
                maxdur=AR.experiment["ABR_settings"]["maxdur"],
                scale=scale,
                invert=invert,
            )
            CP.cprint("g", f"    ***** do one subj: dbs: {dbs!s}")
            #  # generate csv file for abra (Manor lab) analysis
            #  print(waveana.waves.shape)
            # print(1./waveana.sample_freq)

            export_for_abra_clicks(subj=subj, dbs=dbs, waveana=waveana, outputpath=outputpath)
        elif requested_stimulus_type in ["Tone", "Tones", "Interleaved_plateau"]:
            filename = Path(filename).parent
            scale, invert, min_lat, fit_index = set_gain_and_scale(subj, AR)
            CP.cprint("g", f"    ***** do one sub: filename: {filename!s}")
            waveana, dbs, metadata, subject_threshold, subject_frequencies = get_analyzed_tone_data(
                filename,
                AR,
                ABR4,
                subj=subj,
                HPF=AR.experiment["ABR_settings"]["HPF"],
                LPF=AR.experiment["ABR_settings"]["LPF"],
                maxdur=AR.experiment["ABR_settings"]["maxdur"],
                scale=scale,
                invert=invert,
            )
            # CP.cprint("g", f"    ***** do one subj: dbs: {dbs!s}")


def do_one_subject(
    subj: dict,
    treat: str,
    requested_stimulus_type: str,
    AR: object,
    ABR4: object,
    V2uV: float,
    example_subjects: list = None,
    example_axes: list = None,
    test_plots: bool = False,
):
    """do_one_subject : Compute the mean ppio, threshold and IO curve for one subject.

    Raises
    ------
    ValueError
        _description_
    """
    subject_threshold = []
    subject_threshold_unadjusted = []
    subject_ppio = []
    donefiles = []
    dbs = []
    threshold_index = None
    waveana = None  # in case the analysis fails or the dataset was excluded
    # print("requested file type: ", requested_stimulus_type)
    if requested_stimulus_type in ["Click", "Tone", "Interleaved_plateau"]:
        fns = list(subj["filename"].glob("*"))
    else:
        raise ValueError("Requested stimulus type not recognized")
    print("fns: ", fns)
    stim_type = None
    fns = [f for f in fns if not f.name.startswith(".")]
    for filename in sorted(fns):
        # check to see if we have done this file already (esp. ABR4 files)
        fname = filename.name
        print(fname)
        if fname.startswith("."):  # skip hidden files
            continue
        donefiles, stim_type, ok = check_file(
            AR=AR,
            filename=filename,
            subj=subj,
            requested_stimulus_type=requested_stimulus_type,
            donefiles=donefiles,
        )
        if not ok:
            print("Hmmm. no ok")
            continue

        CP.cprint("g", f"    ***** do one subj: stim_type: {stim_type!s}, {fname:s}")
        if stim_type is None:
            print("Stim type is None???")
            return subject_threshold, None, None, None, None, None
        print("do one sub req ::  stim type: ", requested_stimulus_type)
        if requested_stimulus_type == "Click":
            scale, invert, min_lat, fit_index = set_gain_and_scale(subj, AR)
            filename = Path(filename).parent
            CP.cprint("g", f"    ***** do one sub: filename: {filename!s}")
            waveana, dbs, metadata = get_analyzed_click_data(
                filename,
                AR,
                ABR4,
                subj,
                HPF=AR.experiment["ABR_settings"]["HPF"],
                LPF=AR.experiment["ABR_settings"]["LPF"],
                maxdur=AR.experiment["ABR_settings"]["maxdur"],
                scale=scale,
                invert=invert,
            )
            CP.cprint("g", f"    ***** do one subj: dbs: {dbs!s}")
            #  # generate csv file for abra (Manor lab) analysis
            #  print(waveana.waves.shape)
            # print(1./waveana.sample_freq)

            # interpolate to 244 points
            # nsamps = 244
            # new_freq = 24414.0625
            # print(waveana.sample_freq, "new freq: ", new_freq)
            # newx = np.linspace(0, 10.0, num=nsamps)
            # t_delay = 0.0010  # ms
            # oldx = np.arange(-t_delay, 0.010, 1.0 / waveana.sample_freq) * 1e3
            # nold = len(oldx)
            # print(
            #     "oldx: ", oldx.shape, "newx: ", newx.shape, "waveana waves: ", waveana.waves.shape
            # )
            # print("oldx: ", oldx.min(), oldx.max(), newx.min(), newx.max())
            # ldb = []
            # delay_n = int(0.0015 * new_freq)  # 2 ms
            # for i, db in enumerate(dbs):
            #     wave_interpolated = np.interp(newx, oldx, waveana.waves[i, :nold])
            #     row = {
            #         "Freq(Hz)": 100.0,
            #         "Level(dB)": float(dbs[i]),
            #         "Samp. Per.": float((1.0 / new_freq) * 1e6),
            #         "No. Samps.": nsamps,
            #     }
            #     print("interpol: ", wave_interpolated.shape)
            #     for n in range(nsamps):
            #         row.update({f"{n:d}": wave_interpolated[n] * 1e6})
            #     ldb.append(row)
            #     # print("Waveana waves: ", waveana.waves.shape, "sample freq: ", waveana.sample_freq)
            # df = pd.DataFrame.from_dict(ldb)
            # print(df.head())
            # abrap = Path("abra")
            # if not abrap.exists():
            #     abrap.mkdir()
            # df.to_csv(f"abra/{subj['name']:s}_click_data.csv", index=False)

            # exit()

            waveana.get_triphasic(min_lat=min_lat, dev=3)

            waveana.rms_baseline = waveana.rms_baseline  # * 1./metadata["amplifier_gain"]
            # adjust the measures to follow a line of latency when the response gets small.
            waveana.adjust_triphasic(dbs, threshold_index=fit_index, window=0.0005)
            waveana.ppio = waveana.ppio * 1.0 / metadata["amplifier_gain"]

            metadata["stimuli"]["dblist"] = dbs

            # here we check to see if our current file is one of the example
            # subjects, and plot the traces if it is.
            plot_click_stacks(
                AR,
                example_subjects=example_subjects,
                filename=filename,
                waveana=waveana,
                metadata=metadata,
                dbs=dbs,
                axlist=example_axes,
            )

            if np.max(waveana.ppio) > 10:
                CP.cprint(
                    "r",
                    f"     {filename!s} has a max ppio of {np.max(waveana.ppio) :.2f} uV, which is too high",
                )
                raise ValueError
            above_thr = np.where(waveana.ppio > 3.0 * np.mean(waveana.rms_baseline))[0]

            threshold_index = None

            if subj["datatype"] == "pybar3":
                ref_db = (
                    10.0  # use first trace set at the lowest stimulus intensity (below threshold)
                )
            else:
                ref_db = None

            # thr_value, threshold_index, rms = waveana.thresholds(
            #     timebase=waveana.timebase,
            #     waves=waveana.waves,
            #     spls=dbs,
            #     response_window=[1.0e-3, 8e-3],
            #     baseline_window=[0, 0.8e-3],
            #     ref_db = ref_db,
            # )
            # fit with Hill function and get the threshold
            # from the baseline.
            # Note, we fit to the top of the dataset, in case there is
            # some non-monotonicity in the data.
            imax_index = np.nanargmax(waveana.p1n1_amplitudes)
            imax_amp = list(range(imax_index))

            threshold_value, threshold_index, fit = fit_thresholds.fit_thresholds(
                x=np.array(dbs)[imax_amp],
                y=np.array(waveana.p1n1_amplitudes)[imax_amp],
                baseline=waveana.rms_baseline,
                threshold_factor=AR.experiment["ABR_settings"]["click_threshold_factor"],
            )
            print("threshold value: ", threshold_value)
            if threshold_value is not None and not np.isnan(threshold_value):
                db_steps = np.abs(np.mean(np.diff(dbs)))
                threshold_value_adj = round(threshold_value / 2.5) * 2.5  # round to nearest 2.5 dB
                threshold_value_unadj = dbs[(np.abs(dbs - threshold_value)).argmin()]
                subject_threshold.append(threshold_value_adj)
                subject_threshold_unadjusted.append(threshold_value_unadj)
                CP.cprint("m", f"    *****  do one subject: threshold: {threshold_value:.2f} dB")
                subject_ppio.append(float(np.max(waveana.ppio) * V2uV))
            else:
                CP.cprint(
                    "r", f"    *****  do one subject: {subj['name']:s} has no measurable threshold"
                )
                subject_threshold.append(np.nan)
                subject_threshold_unadjusted.append(np.nan)
                subject_ppio.append(np.nan)
            if test_plots:
                f, ax = mpl.subplots(1, 2, figsize=[8, 5])
                # print(filename, subj["filename"].name)
                plot_click_stacks(
                    AR,
                    example_subjects=[subj["filename"].parent.name],
                    filename=filename,
                    waveana=waveana,
                    metadata=metadata,
                    dbs=dbs,
                    thresholds=subject_threshold_unadjusted,
                    axlist=[ax[0]],
                )

                stacki = AR.experiment["ABR_settings"]["stack_increment"]
                dy = stacki * np.array(range(len(dbs)))
                ax[0].plot(
                    waveana.fitline_p1_lat * 1e3,
                    dy + waveana.p1_amplitudes * 1e6,
                    "-o",
                    color="r",
                    linewidth=0.3,
                )
                ax[0].plot(
                    waveana.fitline_n1_lat * 1e3,
                    dy + waveana.n1_amplitudes * 1e6,
                    "-o",
                    color="b",
                    linewidth=0.3,
                )

                ax[1].plot(dbs, waveana.p1n1_amplitudes, "o-", color="k")
                ax[1].plot(fit[0], fit[1], "-", color="r")

                ax[0].set_title(f"{subj['filename'].name:s}")
                mpl.show()
        print("checking for interleaved plateau")
        if requested_stimulus_type in ["Tone", "Interleaved_plateau"]:
            print("found interleaved plateau")
            filename = Path(filename).parent
            CP.cprint("g", f"    ***** do one sub: Tones, filename: {filename!s}")

            waveana, dbs, metadata, thresholds, frequencies = get_analyzed_tone_data(
                filename,
                AR,
                ABR4,
                subj,
                AR.experiment["ABR_settings"]["HPF"],
                AR.experiment["ABR_settings"]["maxdur"],
                scale=scale,
            )
            # print("thr, freq: ", thresholds, frequencies)
            subject_threshold = [thresholds, frequencies]
            waveana.get_triphasic(min_lat=min_lat, dev=3)

            waveana.rms_baseline = waveana.rms_baseline  # * 1./metadata["amplifier_gain"]
            # adjust the measures to follow a line of latency when the response gets small.
            waveana.adjust_triphasic(dbs, threshold_index=fit_index, window=0.0005)
            waveana.ppio = waveana.ppio * 1.0 / metadata["amplifier_gain"]
            if not np.isnan(waveana.p1n1_amplitudes).all():
                imax_index = np.nanargmax(waveana.p1n1_amplitudes)
                imax_amp = list(range(imax_index))
                # print("imax_index: ", imax_index, "imax_amp: ", imax_amp, len(waveana.p1n1_amplitudes))
                # print(waveana.p1n1_amplitudes)
                # print("rms baseline: ", waveana.rms_baseline)
                threshold_value, threshold_index, fit = fit_thresholds.fit_thresholds(
                    x=np.array(dbs)[imax_amp],
                    y=np.array(waveana.p1n1_amplitudes)[imax_amp],
                    baseline=waveana.rms_baseline,
                    threshold_factor=AR.experiment["ABR_settings"]["click_threshold_factor"],
                )

                if threshold_value is not None:
                    db_steps = np.abs(np.mean(np.diff(dbs)))
                    threshold_value_adj = (
                        round(threshold_value / 2.5) * 2.5
                    )  # round to nearest 2.5 dB
                    threshold_value_unadj = dbs[(np.abs(dbs - threshold_value)).argmin()]
                    subject_threshold.append(threshold_value_adj)
                    subject_threshold_unadjusted.append(threshold_value_unadj)
                    CP.cprint(
                        "m", f"    *****  do one subject: threshold: {threshold_value:.2f} dB"
                    )
                else:
                    CP.cprint("r", f"    *****  do one subject: {subj['name']:s} has no threshold")
                    subject_threshold.append(np.nan)
                    subject_threshold_unadjusted.append(np.nan)
            else:
                CP.cprint("r", f"    *****  do one subject: {subj['name']:s} has no threshold")
                subject_threshold.append(np.nan)
                subject_threshold_unadjusted.append(np.nan)
            subject_ppio.append(float(np.max(waveana.ppio) * V2uV))

            if test_plots:
                f, ax = mpl.subplots(1, 2, figsize=[8, 5])
                # print(filename, subj["filename"].name)
                plot_click_stacks(
                    AR,
                    example_subjects=[subj["filename"].parent.name],
                    filename=filename,
                    waveana=waveana,
                    metadata=metadata,
                    dbs=dbs,
                    thresholds=subject_threshold_unadjusted,
                    axlist=[ax[0]],
                )

                stacki = AR.experiment["ABR_settings"]["stack_increment"]
                dy = stacki * np.array(range(len(dbs)))
                ax[0].plot(
                    waveana.fitline_p1_lat * 1e3,
                    dy + waveana.p1_amplitudes * 1e6,
                    "-o",
                    color="r",
                    linewidth=0.3,
                )
                ax[0].plot(
                    waveana.fitline_n1_lat * 1e3,
                    dy + waveana.n1_amplitudes * 1e6,
                    "-o",
                    color="b",
                    linewidth=0.3,
                )

                ax[1].plot(dbs, waveana.p1n1_amplitudes, "o-", color="k")
                ax[1].plot(fit[0], fit[1], "-", color="r")

                ax[0].set_title(f"{subj['filename'].name:s}")
                mpl.show()

        else:
            print("stimulus type not recognized", requested_stimulus_type)
            exit()

    if len(subject_threshold) == 0:
        subject_threshold = [np.nan]
    if waveana is None:
        CP.cprint("y", f"    *****  do one subject: {subj['name']:s} has no data (waveana is NONE)")
        return subject_threshold, None, None, None, None, None
    CP.cprint(
        "g",
        f"\n   *****  do one subject ({fns[0]!s}): # of thresholds measured:  {len(subject_threshold):d}, thrs: {subject_threshold!s}\n",
    )
    return subject_threshold, subject_ppio, dbs, threshold_index, waveana, stim_type


def stuffs(filename, sub, directory_name, HPF, maxdur):
    fname = filename.name
    if fname.startswith("Tone"):

        print("Tone file: ", filename)
        pfiles = list(Path(filename).glob("*_Interleaved_plateau_*.p"))
        if len(pfiles) > 0:
            waves, tb, metadata = AR.read_and_average_abr_files(
                filename=str(pfiles[0]), high_pass_filter=HPF, maxdur=maxdur, pdf=None
            )
            sym = "D"
        else:
            w, t = ABR4.read_dataset(
                subject=subj,
                datapath=directory_name,
                subdir=filename.name,
                datatype="tone",
                highpass=HPF,
            )
        if w is None:  # no data for this stimulus type
            return
    elif fname.lower().startswith("interleaved"):

        print("Interleaved file: ", filename)
        files = list(Path(filename).glob("*.p"))
        print(filename, "\n     # interleaved files: ", len(files))
        HPF = 300.0

        AR.read_and_average_abr_files(
            filename=str(files[0]),
            high_pass_filter=HPF,
            maxdur=maxdur,
            pdf=None,
        )


def compute_tone_thresholds(
    AR,
    ABR4,
    categorize: str = "treatment",
    requested_stimulus_type: str = "Tone",
    subjs: Union[str, Path, None] = None,
    thr_mapax=None,
    example_axes: list = None,
    example_subjects: list = None,
    categories_done: list = None,
    symdata: dict = None,
    test_plots: bool = False,
):
    overridecolor = False
    if categorize in ["treatment"]:
        categories = get_categories(subjs)
    elif categorize in ["age"]:
        categories = get_categories(subjs, categorize="age")
    else:
        raise ValueError(f"Category type {categorize} not recognized")
    # sort categories by the table in the configuration file
    ord_cat = []
    for cat in AR.experiment["plot_order"]["age_category"]:
        if cat in categories:
            ord_cat.append(cat)
    categories = ord_cat
    colors = ["r", "g", "b", "c", "m", "y", "k", "skyblue", "orange", "purple", "brown"]

    plot_colors = AR.experiment["plot_colors"]

    V2uV = 1e6  # factor to multiply data to get microvolts.
    treat = "NT"  # for no treatment
    baseline_dbs = []

    all_thr_freq = {}
    subjs_done = []
    print("requested stimulus type: ", requested_stimulus_type)
    for isubj, subj in enumerate(subjs):  # get data for each subject
        # print("checking sub: ", subj["name"], subj["datatype"])
        if subj["name"] in AR.experiment["ABR_subject_excludes"]:
            continue

            # if subj["name"] not in ["CBA_F_N023_p175_NT", "CBA_F_N024_p175_NT", "CBA_F_N000_p18_NT", "CBA_F_N002_p27_NT", "CBA_F_N015_p573_NT"]:
        #     continue
        # if isubj > 5:
        #     continue

        if subj["name"] in ["CBA_M_N011_p99_NT"]:
            continue

        if subj["name"] in subjs_done:
            continue
        subjs_done.append(subj["name"])
        CP.cprint("m", f"\nSubject: {subj['name']!s}")

        if categorize == "treatment":
            treat = get_treatment(subj)
        elif categorize == "age":
            treat = get_age(subj)

        # f str(subj).endswith("Organization") or str(subj).startswith(".") or str(subj).startswith("Store") or treat.startswith("Store") or treat.startswith("."):
        #     continue
        # print(subj)
        subject_threshold, subject_ppio, dbs, threshold_index, waveana, stim_type = do_one_subject(
            subj=subj,
            treat=treat,
            requested_stimulus_type=requested_stimulus_type,
            AR=AR,
            ABR4=ABR4,
            V2uV=V2uV,
            example_subjects=example_subjects,
            example_axes=example_axes,
            test_plots=test_plots,
        )
        if treat not in all_thr_freq.keys():
            all_thr_freq[treat] = [subject_threshold]  # treat was "subj_category"
        else:
            all_thr_freq[treat].append(subject_threshold)

    # each treatment group in the dict consists of a list (subjects) of lists ([thresholds], [frequencies])
    # first we get all the frequencies in the list of lists
    omit_freqs = [3000.0, 24000.0, 240000.0]
    allfr = []

    for treat in all_thr_freq.keys():
        for i, subn in enumerate(all_thr_freq[treat]):
            if len(subn) == 0:
                continue
            allfr.extend(subn[1])
    allfr = sorted(list(set(allfr)))
    allfr = [f for f in allfr if f not in omit_freqs]
    # next, for each treament group, we create a list of thresholds for each frequency
    # if a subject does not have a threshold for a frequency, we insert a NaN
    thrmap = {}
    for treat in all_thr_freq.keys():
        if treat not in thrmap.keys():
            thrmap[treat] = []
        for i, subn in enumerate(all_thr_freq[treat]):
            # print("subn: ", subn)
            if len(subn) == 0:
                thrmap[treat].append([np.nan for f in allfr if f not in omit_freqs])
            else:
                thrmap[treat].append(
                    [
                        subn[0][subn[1].index(f)] if f in subn[1] else np.nan
                        for f in allfr
                        if f not in omit_freqs
                    ]
                )
    all_tone_map_data = {"frequencies": allfr, "thresholds": thrmap, "plot_colors": plot_colors}
    with open("all_tone_map_data.pkl", "wb") as f:
        pickle.dump(all_tone_map_data, f)

    # plot_tone_map_data(thr_mapax, all_tone_map_data)

    return categories_done, all_tone_map_data


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


def plot_tone_map_data(thr_mapax, all_tone_map_data):
    thrmap = all_tone_map_data["thresholds"]
    allfr = all_tone_map_data["frequencies"]
    plot_colors = all_tone_map_data["plot_colors"]
    print("thrmap: ", thrmap)
    print("allfr: ", allfr)
    f, thr_mapax = mpl.subplots(1, 1)
    t_done = []
    fr_offset = 0.02
    # define frequencies that where ABRs were not consistently measured
    freq_to_remove = [1000.0, 2000.0, 3000.0, 6000.0, 12000.0, 40000.0]
    # for each condition (treatment, age, etc)
    for itr, treat in enumerate(thrmap.keys()):
        print("treat: ", treat)
        if treat == "Young Adult":
            continue
        if treat in t_done:
            t_label = ""
        else:
            t_label = treat
            t_done.append(treat)

        dtr = np.array(thrmap[subj_category])
        frs = np.array(fr_offset * (itr - len(thrmap.keys()) / 2) + np.log10(allfr))
        # stack the arrays together. col 0 is raw freq, col2 is log freq, the rest are the subject threshols
        # measured at this frequency
        data = np.vstack((allfr, frs))
        for dx in dtr:
            data = np.vstack((data, dx))
        data = data.T
        nmeas = data.shape[0] - 2
        rem_fr = []
        for ifr, fr in enumerate(allfr):
            if fr in freq_to_remove:
                rem_fr.append(ifr)
        datao = data.copy()
        data = np.delete(data, rem_fr, axis=0)
        data = np.where(data == 0.0, np.nan, data)
        data = np.where(data > 90.0, np.nan, data)
        topmarker = np.where(datao >= 90.0, datao, np.nan)
        print(topmarker)
        print(len(np.where(topmarker[:, 2:] == 100.0)[0]))
        # for d in data:
        thr_mapax.errorbar(
            data[:, 1],
            np.nanmean(data[:, 2:], axis=1),
            yerr=scipy.stats.sem(data[:, 2:], axis=1, nan_policy="omit"),
            marker="o",
            markersize=4,
            color=plot_colors[subj_category],
            capsize=4,
            linestyle="-",
            linewidth=1.5,
        )
        thr_mapax.plot(
            data[:, 1],
            data[:, 2:],
            color=plot_colors[subj_category],
            marker="o",
            linewidth=0.0,
            markersize=2.5,
            label=t_label,
        )
        thr_mapax.plot(
            datao[:, 1],
            topmarker[:, 2:] + np.random.uniform(-2, 2, size=topmarker[:, 2:].shape),
            color=plot_colors[subj_category],
            marker="^",
            linewidth=0.0,
            markersize=3.5,
        )

    thr_mapax.plot([np.log10(3000.0), np.log10(64000)], [90, 90], linewidth=0.35, color="grey")
    thr_mapax.set_xlabel("Frequency (kHz)")
    thr_mapax.set_ylabel("Threshold (dB SPL)")
    x_vals = [np.log10(4000), np.log10(8000), np.log10(16000), np.log10(32000), np.log10(48000)]
    x_labs = ["4", "8", "16", "32", "48"]
    thr_mapax.set_xticks(x_vals)
    thr_mapax.set_xticklabels(x_labs)

    legend_without_duplicate_labels(thr_mapax)
    leg = thr_mapax.get_legend()
    leg.set_draggable(state=1)
    thr_mapax.set_ylim([0, 110])
    thr_mapax.set_xlim([np.log10(3000.0), np.log10(64000)])
    # thr_mapax.set_xscale("log")
    mpl.show()


def do_tone_map_analysis(
    AR: AnalyzeABR,
    ABR4: object,
    subject_data: dict = None,
    output_file: Union[Path, str] = None,
    subject_prefix: str = "CBA",
    categorize: str = "treatment",
    requested_stimulus_type: str = "Tone",
    experiment: Union[dict, None] = None,
    example_subjects: list = None,
    test_plots: bool = False,
):

    # base directory
    subjs = []
    # combine subjects from the directories
    # for directory_name in directory_names.keys():
    #     d_subjs = list(Path(directory_name).glob(f"{subject_prefix:s}*"))
    #     subjs.extend([ds for ds in d_subjs if ds.name.startswith(subject_prefix)])
    print("subject data keys: ", subject_data.keys())
    if requested_stimulus_type == "Tones":  # include interleaved_plateau as well
        subjs = subject_data[requested_stimulus_type]
        subjs.extend(subject_data["Interleaved_plateau"])
    # print("subjs with stim type", subjs)
    categories_done = []
    if output_file is not None:
        STYLE = ST.styler("JNeurophys", figuresize="full", height_factor=0.6)
        with PdfPages(output_file) as pdf:
            P = make_tone_figure(subjs, example_subjects, STYLE)

            if example_subjects is None:
                thr_mapax = P.axarr[1][0]

                example_axes = None
            else:
                example_axes = [P.axarr[i][0] for i in range(len(example_subjects))]
                nex = len(example_subjects)
                thr_mapax = P.axarr[1][0]

            categories_done, all_tone_map_data = compute_tone_thresholds(
                AR,
                ABR4,
                requested_stimulus_type=requested_stimulus_type,
                categorize=categorize,
                subjs=subjs,
                thr_mapax=thr_mapax,
                example_subjects=example_subjects,
                example_axes=example_axes,
                categories_done=categories_done,
                test_plots=test_plots,
            )
            print("all tone map data: ", all_tone_map_data)
            plot_tone_map_data(thr_mapax, all_tone_map_data)


def make_tone_figure(subjs, example_subjects, STYLE):
    row1_bottom = 0.1
    vspc = 0.08
    hspc = 0.06
    ncols = 1
    if example_subjects is not None:
        ncols += len(example_subjects)

    up_lets = ascii_letters.upper()
    ppio_labels = [up_lets[i] for i in range(ncols)]
    sizer = {
        "A": {"pos": [0.05, 0.175, 0.1, 0.90], "labelpos": (-0.15, 1.05)},
        "B": {"pos": [0.25, 0.6, 0.1, 0.90], "labelpos": (-0.15, 1.05)},
        # "C": {"pos": [0.49, 0.20, 0.12, 0.83], "labelpos": (-0.15, 1.05)},
        # "D": {"pos": [0.76, 0.22, 0.59, 0.36], "labelpos": (-0.15, 1.05)},
        # "E": {"pos": [0.76, 0.22, 0.12, 0.36], "labelpos": (-0.15, 1.05)},
    }

    P = PH.arbitrary_grid(
        sizer=sizer,
        order="rowsfirst",
        figsize=STYLE.Figure["figsize"],
        font="Arial",
        fontweight=STYLE.get_fontweights(),
        fontsize=STYLE.get_fontsizes(),
    )

    # PH.show_figure_grid(P, STYLE.Figure["figsize"][0], STYLE.Figure["figsize"][1])
    return P


def compute_click_io_analysis(
    AR,
    ABR4,
    categorize: str = "treatment",
    requested_stimulus_type: str = "Click",
    subjs: Union[str, Path, None] = None,
    axio=None,
    axthr=None,
    axppio=None,
    example_axes: list = None,
    example_subjects: list = None,
    categories_done: list = None,
    symdata: dict = None,
    test_plots: bool = False,
):
    overridecolor = False
    if categorize in ["treatment"]:
        categories = get_categories(subjs)
    elif categorize in ["age_category"]:
        categories = get_categories(subjs, categorize="age")
    else:
        raise ValueError(f"Category type {categorize} not recognized")
    CP.cprint("y", f"   compute_click_io_analysis::categories: {categories}")
    # sort categories by the table in the configuration file
    ord_cat = []
    for cat in AR.experiment["plot_order"]["age_category"]:
        if cat in categories:
            ord_cat.append(cat)
    categories = ord_cat
    colors = ["r", "g", "b", "c", "m", "y", "k", "skyblue", "orange", "purple", "brown"]

    plot_colors = AR.experiment["plot_colors"]

    color_dict = {}
    # print(categories)
    waves_by_treatment = {}
    baseline = {}
    thresholds_by_treatment = {}  # {"treat1": [thr1, thr2, thr3], "treat2": [thr1, thr2, thr3]}
    amplitudes_by_treatment = {}  # {"treat1": [amp1, amp2, amp3], "treat2": [amp1, amp2, amp3]}
    dbs_by_treatment = {}

    baseline_dbs = []
    line_plot_colors = {k: v for k, v in plot_colors["line_plot_colors"].items()}
    df_abr = None  # pd.DataFrame({"Subject", "treatment", "threshold", "amplitude"])
    df_io = pd.DataFrame()  # accumulate IO data for all subjects
    print("\n    Click IO for subjects: ", subjs)
    for subj in subjs:  # get data for each subject
        if AR.experiment["ABR_subject_excludes"] is not None:
            if subj["name"] in AR.experiment["ABR_subject_excludes"]:
                continue
        print("    Subject Name: ", subj["name"])
        # CP.cprint("m", f"\nSubject: {subj!s}")

        if categorize == "treatment":
            subj_category = get_treatment(subj)
        elif categorize == "age_category":
            subj_category = get_age(subj)
        V2uV = 1e6  # factor to multiply data to get microvolts.

        # f str(subj).endswith("Organization") or str(subj).startswith(".") or str(subj).startswith("Store") or treat.startswith("Store") or treat.startswith("."):
        #     continue
        # print(subj)
        subject_threshold, subject_ppio, dbs, threshold_index, waveana, stim_type = do_one_subject(
            subj=subj,
            treat=subj_category,
            requested_stimulus_type=requested_stimulus_type,
            AR=AR,
            ABR4=ABR4,
            V2uV=V2uV,
            example_subjects=example_subjects,
            example_axes=example_axes,
            test_plots=test_plots,
        )
        ddict = {
            "Subject": subj["name"],
            categorize: subj_category,
            "sex": REX.re_subject.match(subj["name"]).group("sex"),
            "threshold": np.mean(subject_threshold),
            "amplitude": float(np.max(subject_ppio)),
        }
        if df_abr is None:
            df_abr = pd.DataFrame(ddict, index=[0])
        else:
            df_abr = pd.concat([df_abr, pd.DataFrame(ddict, index=[0])], ignore_index=True)
        if waveana is None or stim_type is None:
            continue
        CP.cprint(
            "r", f"Subject {subj['name']:s} thr: {subject_threshold!s}  stim_type: {stim_type!s}"
        )
        # if subject_threshold is None or stim_type is None:
        #     continue
        dataset_name = (stim_type, subj_category)
        if subj_category not in thresholds_by_treatment:
            thresholds_by_treatment[subj_category] = []
        thresholds_by_treatment[subj_category].append(float(np.nanmean(subject_threshold)))
        if subj_category not in amplitudes_by_treatment:
            amplitudes_by_treatment[subj_category] = []
        amplitudes_by_treatment[subj_category].append(float(np.nanmean(subject_ppio)))
        if subj_category not in dbs_by_treatment:
            dbs_by_treatment[subj_category] = []
        dbs_by_treatment[subj_category].extend(dbs)
        # print("PPIO data: ", waveana.ppio)
        if np.all(np.isnan(waveana.ppio)):
            print("all nan? : ", waveana.ppio)
            raise ValueError
        else:
            if subj_category not in waves_by_treatment:
                waves_by_treatment[subj_category] = [np.array([dbs, waveana.ppio])]
            else:
                waves_by_treatment[subj_category].append(np.array([dbs, waveana.ppio]))
        CP.cprint(
            "c",
            f"   dataset name: {dataset_name!s}  categories_done: {categories_done!s}, group: {subj_category:s}, {axio!s}",
        )

        # plot the individual io functions.
        if axio is not None:
            # if subj["datatype"] == "pyabr3":
            #     symdata = ["d", 4]
            V2uV = 1e6
            if subj["name"][6] == "R":  # Ruili data set
                V2uV = 1e6  # different scale.
            if subj_category not in line_plot_colors.keys():
                print(
                    "subj category NOT in line plot colors: ",
                    subj_category,
                    line_plot_colors.keys(),
                )
                exit()

            if dataset_name not in categories_done:
                color = line_plot_colors[subj_category]
                colors.pop(0)
                color_dict[dataset_name] = color

                categories_done.append(dataset_name)
                if overridecolor:
                    color = "k"
                dbs, ppio = _plot_io_data(
                    waveana=waveana,
                    dbs=dbs,
                    # threshold_index=threshold_index,
                    V2uV=V2uV,
                    color=color,
                    ax=axio,
                    symdata=symdata,
                    dataset_name=dataset_name,
                    add_label=True,
                )
            # else:
            #     color = line_plot_colors[subj_category]  # color_dict[dataset_name]
            #     if overridecolor:
            #         color = "k"
            #     dbs, ppio = _plot_io_data(
            #         waveana=waveana,
            #         dbs=dbs,
            #         # threshold_index=threshold_index,
            #         V2uV=V2uV,
            #         color=color,
            #         ax=axio,
            #         symdata=symdata,
            #         dataset_name=dataset_name,
            #         add_label=False,
            #     )

            if requested_stimulus_type not in baseline.keys():
                baseline[stim_type] = waveana.rms_baseline
                baseline_dbs.extend(dbs)
            else:
                if baseline[stim_type].ndim == 1:
                    baseline[stim_type] = baseline[stim_type][np.newaxis, :]
                n_base = baseline[stim_type].shape[1]
                n_wave = waveana.rms_baseline.shape[0]
                if n_wave == n_base:
                    baseline[stim_type] = np.vstack((baseline[stim_type], waveana.rms_baseline))
                    baseline_dbs = np.append(baseline_dbs, dbs)
            baseline_dbs = np.sort(np.unique(baseline_dbs))  # sorted(list(set(baseline_dbs)))

        dbs = dbs.squeeze()
        ppio = ppio.squeeze()
        subj_b = "_".join(subj["name"].split("_")[:3])
        subj_x = subj_b + "_x"
        subj_y = subj_b + "_y"
        df_ppio = pd.DataFrame({subj_x: dbs, subj_y: ppio})
        df_io = pd.concat([df_io, pd.DataFrame(df_ppio)], axis=1, ignore_index=False)
        mpl.show()
        exit()
    return

    # this snext section saves the data to csv files,
    # but also some other stuff we don't really need...
    df_io.to_csv("io_data.csv", index=False)
    df_abr.to_csv("abr_data.csv", index=False)  # saves in long form for R
    categories_done = [v[1] for v in categories_done]
    #  Now plot the baseline and the mean for each stimulus type
    # cnames = list(set(["_".join(s.split("_")[:3])for s in df_io.columns]))
    # f, ax = mpl.subplots(1, 1, figsize=(8, 5))
    # for i, cname in enumerate(cnames):
    #     ax.plot(df_io[cname + "_x"], df_io[cname + "_y"], "o-", color=colors[i % len(colors)], label=cname)
    # ax.set_xlabel("SPL (dB)")
    # ax.set_ylabel("PPIO (uV)")
    # ax.set_title("Click IO functions")
    # mpl.show()
    # exit()
    plot_colors = AR.experiment["plot_colors"]
    line_plot_colors = {
        k: p for k, p in plot_colors["line_plot_colors"].items() if k in categories_done
    }
    print("Category: ", categorize)
    plot_order = AR.experiment["plot_order"][categorize]
    plot_order = [p for p in plot_order if p in categories_done]
    bkc = [v for k, v in plot_colors["bar_background_colors"].items() if k in categories_done]
    bec = [v for k, v in plot_colors["bar_edge_colors"].items() if k in categories_done]
    symc = [v for k, v in plot_colors["symbol_colors"].items() if k in categories_done]
    # exit()
    # print(bkc)
    # print(bec)
    # exit()

    if stim_type is None:
        print("stimulus type is None for ", subj)
        return categories_done
    else:
        print("stimulus type: for subj", subj["name"], stim_type)

    if axio is not None and stim_type in baseline.keys():
        # get baseline data and plot a grey bar
        bl_mean = np.mean(baseline[stim_type], axis=0) * V2uV
        bl_std = np.std(baseline[stim_type], axis=0) * V2uV

        axio.fill_between(baseline_dbs, bl_mean - bl_std, bl_mean + bl_std, color="grey", alpha=0.7)
        # For each category
        for k, treat in enumerate(categories):
            print("getting mean for category: ", treat)
            dataset_name = (stim_type, treat)
            # get all the SPLs for this dataset.
            all_dbs = []
            if treat not in waves_by_treatment.keys():
                continue
            for i in range(len(waves_by_treatment[subj_category])):
                all_dbs.extend(waves_by_treatment[subj_category][i][0])
            # and just the unique ones, in order
            all_dbs = [float(d) for d in sorted(list(set(all_dbs)))]
            # build a np array for all the response measures
            wvs = np.nan * np.ones((len(waves_by_treatment[subj_category]), len(all_dbs)))
            for i in range(len(waves_by_treatment[subj_category])):
                for j, db in enumerate(waves_by_treatment[subj_category][i][0]):
                    k = all_dbs.index(db)
                    wvs[i, k] = waves_by_treatment[subj_category][i][1][j]
            waves_by_treatment[subj_category] = wvs
            all_dbs = np.array(all_dbs)

            # limit the plot to a common range  (this could be smarter - we could count nonnan observations
            # at each level and only plot where we have sufficient data, e.g., N = 3 or larger?
            valid_dbs = np.argwhere((all_dbs >= 20.0) & (all_dbs <= 90.0))
            valid_dbs = [int(v) for v in valid_dbs]
            if waves_by_treatment[subj_category].ndim > 1:
                d_mean = np.nanmean(waves_by_treatment[subj_category], axis=0) * V2uV
                d_std = np.nanstd(waves_by_treatment[subj_category] * V2uV, axis=0)
            else:
                d_mean = waves_by_treatment[subj_category] * V2uV
                d_std = np.zeros_like(d_mean)
            color = line_plot_colors[subj_category]
            n_mean = len(all_dbs)
            axio.plot(
                all_dbs[valid_dbs],
                d_mean[valid_dbs],
                color=color,
                linestyle="-",
                linewidth=1.5,
                alpha=0.75,
            )
            if np.max(d_std) > 0:
                axio.fill_between(
                    all_dbs[valid_dbs],
                    d_mean[valid_dbs] - d_std[valid_dbs],
                    d_mean[valid_dbs] + d_std[valid_dbs],
                    color=color,
                    alpha=0.1,
                    edgecolor=color,
                    linewidth=1,
                )
        axio.set_ylim(0, 12)
        PH.do_talbotTicks(axio, axes="x", density=[1, 2], insideMargin=0.05)
        PH.do_talbotTicks(
            axio,
            axes="y",
            density=[0.5, 1.5],
            tickPlacesAdd={"x": 0, "y": 0},
            floatAdd={"x": 0, "y": 0},
            insideMargin=0.05,
        )
        PH.nice_plot(axio, direction="outward", ticklength=3)
        axio.set_yticks(np.arange(0, 12.0, 1), minor=True)
        axio.tick_params(axis="y", which="minor", length=1.5, width=0.5, color="k")
        axio.set_xticks(np.arange(20, 100.0, 10), minor=True)
        axio.tick_params(axis="x", which="minor", length=1.5, width=0.5, color="k")

    if axthr is not None:
        df = pd.DataFrame.from_dict(thresholds_by_treatment, orient="index")
        df = df.transpose()

        bplot = sns.barplot(
            df,
            ax=axthr,
            width=0.65,
            palette=bkc,
            order=plot_order,
            alpha=0.7,
            linewidth=1,
            capsize=0.18,
            edgecolor="k",
            errorbar=("sd"),
            err_kws={"linewidth": 0.75, "color": "k"},
        )
        # need to set edge color separately
        for i, patch in enumerate(bplot.patches):
            clr = patch.get_facecolor()
            patch.set_edgecolor("w")
            patch.set_linewidth(0.0)

        sns.swarmplot(
            data=df,
            ax=axthr,
            palette=plot_colors["bar_background_colors"],
            order=plot_order,
            size=3.0,
            linewidth=0.5,
        )
        axthr.set_ylabel("Threshold (dB SPL)")
        axthr.set_xlabel("Treatment")
        remap_xlabels(axthr)
        axthr.set_ylim(0, 100)
        PH.nice_plot(axthr, direction="outward", ticklength=3)
        PH.do_talbotTicks(axthr, axes="x", density=[1, 2], insideMargin=0.05)
        PH.do_talbotTicks(axthr, axes="y", density=[0.5, 1], insideMargin=0.05)
        df.to_csv("thresholds.csv")

    if axppio is not None:
        df = pd.DataFrame.from_dict(amplitudes_by_treatment, orient="index")
        df = df.transpose()

        bplot = sns.barplot(
            df,
            ax=axppio,
            width=0.65,
            palette=bkc,
            order=plot_order,
            alpha=0.7,
            linewidth=1,
            edgecolor="k",
            errorbar=("sd"),
            capsize=0.18,
            err_kws={"linewidth": 0.75, "color": "k"},
        )
        # need to set edge color separately
        for i, patch in enumerate(bplot.patches):
            clr = patch.get_facecolor()
            patch.set_edgecolor("w")
            patch.set_linewidth(0.0)
        print("palette: ", symc)
        print("plot order: ", plot_order)
        sns.swarmplot(data=df, ax=axppio, palette=symc, order=plot_order, size=3.0, linewidth=0.5)

        axppio.set_ylabel("Amplitude (uV)")
        axppio.set_xlabel("Treatment")
        remap_xlabels(axppio)
        axppio.set_ylim(0, 12)
        PH.nice_plot(axppio, direction="outward", ticklength=3)
        PH.do_talbotTicks(
            axppio,
            axes="y",
            density=[0.5, 1.5],
            tickPlacesAdd={"x": 0, "y": 0},
            floatAdd={"x": 0, "y": 0},
            insideMargin=0.05,
        )
        PH.nice_plot(axppio, direction="outward", ticklength=3)
        axppio.set_yticks(np.arange(0, 12.0, 1), minor=True)
        axppio.tick_params(axis="y", which="minor", length=1.5, width=0.5, color="k")
        df.to_csv("amplitudes.csv")
    return categories_done
    # break
    # print("ax click2, 2, example: ", ax_click1, ax_click2, example_subjects)
    # if ax_click1 is not None and example_subjects is not None:
    #     plot_click_stack(filename=filename, directory_name=directory_name, ax=ax_click1)

    # if ax_click2 is not None and example_subjects is not None:
    #     plot_click_stack(filename=filename, directory_name=directory_name, ax=ax_click2)


def make_click_figure(subjs, example_subjects, STYLE):
    row1_bottom = 0.1
    vspc = 0.08
    hspc = 0.06
    ncols = 3
    if example_subjects is not None:
        ncols += len(example_subjects)

    up_lets = ascii_letters.upper()
    ppio_labels = [up_lets[i] for i in range(ncols)]
    sizer = {
        "A": {"pos": [0.05, 0.175, 0.1, 0.90], "labelpos": (-0.05, 1.05)},
        "B": {"pos": [0.25, 0.175, 0.1, 0.90], "labelpos": (-0.05, 1.05)},
        "C": {"pos": [0.49, 0.20, 0.12, 0.83], "labelpos": (-0.05, 1.05)},
        "D": {"pos": [0.76, 0.22, 0.59, 0.36], "labelpos": (-0.05, 1.05)},
        "E": {"pos": [0.76, 0.22, 0.12, 0.36], "labelpos": (-0.05, 1.05)},
    }

    P = PH.arbitrary_grid(
        sizer=sizer,
        order="rowsfirst",
        figsize=STYLE.Figure["figsize"],
        font="Arial",
        fontweight=STYLE.get_fontweights(),
        fontsize=STYLE.get_fontsizes(),
        labelsize=12,
    )
    # P = PH.Plotter(
    #     (5,1),
    #     axmap = axmap,
    #     order="rowsfirst",
    #     figsize=STYLE.Figure["figsize"],
    #     # horizontalspacing=hspc,
    #     # verticalspacing=vspc,
    #     margins={
    #         "leftmargin": 0.07,
    #         "rightmargin": 0.07,
    #         "topmargin": 0.05,
    #         "bottommargin": row1_bottom,
    #     },
    #     labelposition=(-0.15, 1.05),
    #     # panel_labels=ppio_labels,
    #     font="Arial",
    #     fontweight=STYLE.get_fontweights(),
    #     fontsize=STYLE.get_fontsizes(),
    # )

    # PH.show_figure_grid(P, STYLE.Figure["figsize"][0], STYLE.Figure["figsize"][1])
    return P


def do_click_io_analysis(
    AR: AnalyzeABR,
    ABR4: object,
    subject_data: dict = None,
    output_file: Union[Path, str] = None,
    subject_prefix: str = "CBA",
    categorize: str = "treatment",
    requested_stimulus_type: str = "Click",
    example_subjects: list = None,
    test_plots: bool = False,
):

    # base directory
    subjs = []
    # combine subjects from the directories
    # for directory_name in directory_names.keys():
    #     d_subjs = list(Path(directory_name).glob(f"{subject_prefix:s}*"))
    #     subjs.extend([ds for ds in d_subjs if ds.name.startswith(subject_prefix)])
    subjs = subject_data[requested_stimulus_type]
    categories_done = []
    if output_file is not None:
        STYLE = ST.styler("JNeurophys", figuresize="full", height_factor=0.6)
        with PdfPages(output_file) as pdf:
            CP.cprint("m", f"Making Output file: {output_file!s}")
            P = make_click_figure(subjs, example_subjects, STYLE)

            if example_subjects is None:
                ioax = P.axarr[0][0]
                thrax = P.axarr[1][0]
                ppioax = P.axarr[2][0]
                example_axes = None
            else:
                example_axes = [P.axarr[i][0] for i in range(len(example_subjects))]
                nex = len(example_subjects)
                ioax = P.axarr[nex][0]
                thrax = P.axarr[nex + 1][0]
                ppioax = P.axarr[nex + 2][0]

            print("\n    Stimulus_type: ", requested_stimulus_type)

            categories_done = compute_click_io_analysis(
                AR,
                ABR4,
                requested_stimulus_type=requested_stimulus_type,
                categorize=categorize,
                subjs=subjs,
                axio=ioax,
                axthr=thrax,
                axppio=ppioax,
                example_subjects=example_subjects,
                example_axes=example_axes,
                categories_done=categories_done,
                test_plots=test_plots,
            )

            ioax.set_xlabel("dB SPL")
            ioax.set_clip_on(False)
            ioax.set_ylabel(f"{requested_stimulus_type:s} P1-N1 Peak to Peak Amplitude (uV)")
            leg = ioax.legend(
                prop={"size": 4, "weight": "normal", "family": "Arial"},
                fancybox=False,
                shadow=False,
                facecolor="none",
                loc="upper left",
                draggable=True,
            )
            leg.set_zorder(1000)
            ioax.set_title(
                f"Peak to Peak Amplitude for all subjects\n{subject_prefix:s}",
                va="top",
            )
            ioax.text(
                0.96,
                0.01,
                s=datetime.datetime.now(),
                fontsize=6,
                ha="right",
                transform=P.figure_handle.transFigure,
            )
            mpl.show()

            pdf.savefig(P.figure_handle)
            CP.cprint("y", f"\nSaved to: {output_file}")
            P.figure_handle.clear()
    else:
        for directory_name in directory_names:
            CP.cprint("m", f"directory name: {directory_name}")

            categories_done = compute_io_analysis(
                AR,
                ABR4,
                stimulus_type=stimulus_type,
                categorize=categorize,
                subjs=subjs,
                directory_name=directory_name,
                axio=ioax,
                axthr=thrax,
                axppio=ppioax,
                example_subjects=example_subjects,
                example_axes=example_axes,
                symdata=directory_names[directory_name],
                categories_done=categories_done,
            )
    return categories_done


def test_age_re():
    import re

    re_age = re.compile(r"[_ ]{1}[p]{0,1}[\d]{1,3}[d]{0,1}[_ ]{1}", re.IGNORECASE)
    examples = [
        "CBA_M_N017_p572_NT",
        "CBA_M_N004_p29_NT",
    ]
    for s in examples:
        m = re_age.search(s)
        if m is not None:
            print(m.group())
            age = m.group().strip("_ ")
            parsed = PA.age_as_int(PA.ISO8601_age(age))
            print(parsed)
            print()
        else:
            print("No match")


def find_files_for_subject(
    dataset: Union[str, Path], subj_data: dict = None, stim_types: list = None
):
    """find_files_for_subject : Find the files for a given subject
    Return the list of files for the requested stimulus type, and the
    directory that holds those files.
    If there is not match, we return None
    """

    filenames = list(
        Path(dataset).glob("*")
    )  # find all of the files and directories in this subject directory
    filenames = [fn for fn in filenames if not fn.name.startswith(".")]  # clean out .DS_Store, etc.
    # print("Top level fns: ", filenames)
    # these are possible directories in the subject directory
    # the data may be at the top level (common for older ABR4 datasets)
    # or they may be in these subdirectories. We will find out by looking at the
    # files in the subject directory.

    # dirs = ["click", "tone", "interleaved_plateau"]
    # if requested_stimulus_type not in dirs:
    #     raise ValueError(f"Requested stimulus type {requested_stimulus_type} not recognized")
    if subj_data is None:
        subj_data = {"Click": [], "Tones": [], "Interleaved_plateau": []}
    for filename in filenames:
        # print("filename: ", filename)
        if filename.name in [
            "Click",
            "Tones",
            "interleaved_plateau",
            "Interleaved_plateau",
            "Interleaved_plateau_High",
        ]:
            short_name = Path(filename.parent).name
        else:
            short_name = filename.name

        m_subject = REX.re_subject.match(short_name)
        if m_subject is not None:
            subject = m_subject["subject"]
        else:
            subject = None
        # abr4 clicks in the main directory
        m_click = REX.re_click_file.match(filename.name)
        if m_click is not None and filename.is_file():
            dset = {
                "datetime": m_click["datetime"],
                "subject": subject,
                "name": filename.parent.name,
                "filename": filename,
                "datatype": "ABR4",
            }
            if dset not in subj_data["Click"]:
                subj_data["Click"].append(dset)

        # abr4 tones in the main directory
        m_tone = REX.re_tone_file.match(filename.name)
        if m_tone is not None and filename.is_file():
            dset = {
                "datetime": m_tone["datetime"],
                "subject": subject,
                "name": filename.parent.name,
                "filename": filename,
                "datatype": "ABR4",
            }
            if dset not in subj_data["Tone"]:
                subj_data["Tone"].append(dset)

        # subdirectory with Click name - either ABR4 or pyabr3
        if filename.name == "Click" and filename.is_dir():
            # first check for ABR4 files
            click_files = filename.glob("*.txt")
            for cf in click_files:
                m_click = REX.re_click_file.match(cf.name)
                if m_click is None:
                    continue
                dset = {
                    "datetime": m_click["datetime"],
                    "subject": subject,
                    "name": filename.parent.name,
                    "filename": filename,
                    "datatype": "ABR4",
                }
                if dset not in subj_data["Click"]:
                    subj_data["Click"].append(dset)

            # next check for pyabr3 files
            click_files = filename.glob("*.p")
            for i, cf in enumerate(click_files):
                m_click = REX.re_pyabr3_click_file.match(cf.name)
                if m_click is None:
                    continue
                dset = {
                    "datetime": m_click["datetime"],
                    "subject": subject,
                    "name": filename.parent.name,
                    "filename": filename,
                    "datatype": "pyabr3",
                }
                if dset not in subj_data["Click"]:
                    subj_data["Click"].append(dset)

        # Tones in a Tone subdirectory
        if filename.name == "Tones" and filename.is_dir():
            tone_files = filename.glob("*.txt")
            for tone_file in tone_files:
                m_tone = REX.re_tone_file.match(tone_file.name)
                if m_tone is None:
                    continue
                dset = {
                    "datetime": m_tone["datetime"],
                    "subject": subject,
                    "name": filename.parent.name,
                    "filename": filename,
                    "datatype": "ABR4",
                }
                if dset not in subj_data["Tones"]:
                    subj_data["Tones"].append(dset)

        # next check for pyabr3 files
        if (
            filename.name.startswith(("Interleaved_plateau", "interleaved_plateau"))
            and filename.is_dir()
        ):
            tone_files = filename.glob("*.p")
            # print("tone files: ", filename, list(tone_files))
            for i, tone_file in enumerate(tone_files):
                m_tone = REX.re_pyabr3_interleaved_tone_file.match(tone_file.name)
                # print(tone_file.name, m_tone["datetime"], m_tone["serial"])
                if m_tone is None:
                    continue
                dset = {
                    "datetime": m_tone["datetime"],
                    "subject": subject,
                    "name": filename.parent.name,
                    "filename": filename,
                    "datatype": "pyabr3",
                }
                if dset not in subj_data["Interleaved_plateau"]:
                    subj_data["Interleaved_plateau"].append(dset)

    return subj_data


def get_datasets(directory_names, filter: str = "CBA"):
    subdata = None
    for directory in list(directory_names.keys()):
        if not Path(directory).is_dir():
            raise ValueError("Directory does not exist: ", directory)
        subs = [
            sdir
            for sdir in Path(directory).glob("*")
            if sdir.is_dir()
            and not sdir.name.startswith(".")
            and not sdir.name.startswith("Old Organization")
            and not sdir.name.startswith("NG_")
        ]

        for sub in subs:
            print("subdir name: ", sub.name, filter)
            if not sub.name.startswith(filter):
                continue
            subdata = find_files_for_subject(
                dataset=sub, subj_data=subdata, stim_types=["Click", "Interleaved_plateau", "Tone"]
            )
    # for s in subdata.keys():
    #     for d in subdata[s]:
    #         print(s, d)
    return subdata


if __name__ == "__main__":

    AR = AnalyzeABR()
    ABR4 = read_abr4.READ_ABR4()

    # do_io_analysis(
    #     directory_name="/Volumes/Pegasus_002/ManisLab_Data3/abr_data/Reggie_NIHL",
    #     output_file="NIHL_VGAT-EYFP_IO_ABRs_combined.pdf",
    #     subject_prefix="VGAT-EYFP",
    #     categorize="treatment",
    #     # hide_treatment=True,
    # )
    #  do_click_io_analysis:
    # AR: AnalyzeABR,
    #     ABR4: object,
    #     subject_data: dict = None,
    #     output_file: Union[Path, str] = None,
    #     subject_prefix: str = "CBA",
    #     categorize: str = "treatment",
    #     requested_stimulus_type: str = "Click",
    #     experiment: Union[dict, None] = None,
    #     example_subjects: list = None,
    #     test_plots: bool
    # do_click_io_analysis(
    #     AR=AR,
    #     directory_name="/Volumes/Pegasus_004/ManisLab_Data3/abr_data/Reggie_NIHL",
    #     subject_prefix="Glyt2EGFP",
    #     output_file="NIHL_Glyt2EGFP_ABRs_IOFunctions_combined.pdf",
    #     # output_file="NIHL_GlyT2_ABRs_IOFunctions_combined.pdf",
    #     categorize="treatment",
    #       )
    # exit()
    config_file_name = "/Users/pbmanis/Desktop/Python/RE_CBA/config/experiments.cfg"
    expt = "GlyT2_NIHL"  # or "CBA"

    if expt == "CBA":
        AR.get_experiment(config_file_name, "CBA_Age")
        directory_names = {  # values are symbol, symbol size, and relative gain factor
            "/Volumes/Pegasus_002/ManisLab_Data3/abr_data/Reggie_CBA_Age": ["o", 3.0, 1.0],
            # "/Volumes/Pegasus_002/ManisLab_Data3/abr_data/Ruilis ABRs": ["x", 3.0, 10.],
        }

        subdata = get_datasets(directory_names)
        # select subjects for tuning analysis parameters in the configuration file.
        # subdata = get_datasets(directory_names)
        tests = False
        if tests:
            test_subjs = ["N004", "N005", "N006", "N007"]
            newsub = {"Click": []}
            # print(subdata.keys())
            for sub in subdata:
                # print(sub)
                for d in subdata[sub]:
                    # print(d)
                    if d["subject"] in test_subjs:
                        newsub["Click"].append(d)

            subdata = newsub
            # for ns in subdata["Click"]:
            # print(ns)
        # exit()
        test_plots = False

        do_click_io_analysis(
            AR=AR,
            ABR4=ABR4,
            subject_data=subdata,
            subject_prefix="CBA_",
            output_file="CBA_Age_ABRs_Clicks_combined.pdf",
            categorize="age_category",
            requested_stimulus_type="Click",
            # example_subjects=["CBA_F_N002_p27_NT", "CBA_M_N017_p572_NT"],
            test_plots=test_plots,
        )
        # do_tone_map_analysis(
        #     AR=AR,
        #     ABR4=ABR4,
        #     subject_data=subdata,
        #     subject_prefix="CBA_",
        #     output_file="CBA_Age_ABRs_Tones_combined.pdf",
        #     categorize="age",
        #     requested_stimulus_type="Tone",
        #     example_subjects=["CBA_F_N002_p27_NT", "CBA_M_N017_p572_NT"],
        #     test_plots = test_plots
        # )

        # with open("all_tone_map_data.pkl", "rb") as f:
        #     all_tone_map_data = pickle.load(f)
        # f, ax = mpl.subplots(1, 1)
        # plot_tone_map_data(ax, all_tone_map_data=all_tone_map_data)

    if expt == "GlyT2_NIHL":
        AR.get_experiment(config_file_name, "GlyT2_NIHL")
        directory_names = {  # values are symbol, symbol size, and relative gain factor
            "/Volumes/Pegasus_004/ManisLab_Data3/abr_data/Reggie_NIHL": ["o", 3.0, 1.0],
        }

        outputpath = Path(AR.experiment["ABR_settings"].get("outputpath", "abra"))
        subdata = get_datasets(directory_names, filter="Glyt2EGFP")
        print("Subdata: ", subdata.keys())
        # exit()
        # select subjects for tuning analysis parameters in the configuration file.
        tests = False
        stim_type = "Click"
        # stim_type = "Interleaved_plateau"
        # stim_type="Click"
        if tests:
            test_subjs = ["WJ8"]  # , "XT9", "XT11", "N007"]
            newsub = {stim_type: []}
            # print(subdata.keys())
            for sub in subdata:
                # print(sub)
                for d in subdata[sub]:
                    # print("*****", d)
                    if d["subject"] in test_subjs:
                        newsub[stim_type].append(d)
                        print("Adding subject: ", d["subject"], " to new subdata")

            subdata = newsub
        print("subdata stim type: ", subdata[stim_type])
        test_plots = False
        # exit()
        # uncomment this to write the click csv files for the ABRA program.
        for subj in subdata[stim_type]:
            print("Subject: ", subj)
            print("output path: ", outputpath)
            export_for_abra(AR=AR, ABR4=ABR4, subj=subj, requested_stimulus_type=stim_type,
                            outputpath=outputpath)
        exit()

        # stim="Click"
        # print("\nSubdata IP: ", subdata[stim])
        # print("\nSubdata Click: ", subdata["Click"])
        # print("\nSubdata Tone: ", subdata["Tone"])
        # for n, subj in enumerate(subdata[stim]):
        #     print("Subject: ", subj["subject"])
        #     export_for_abra(AR=AR, ABR4=ABR4, subj=subj, requested_stimulus_type=stim)
        # do_click_io_analysis(
        #     AR=AR,
        #     ABR4=ABR4,
        #     subject_data=subdata,
        #     subject_prefix="GlyT2",
        #     output_file="GlyT2_NIHL_ABRs_Clicks_combined.pdf",
        #     categorize="treatment",
        #     requested_stimulus_type="Click",
        #     example_subjects=None,  # ["CBA_F_N002_p27_NT", "CBA_M_N017_p572_NT"],
        #     test_plots=test_plots,
        # )
        # # do_tone_map_analysis(
        #     AR=AR,
        #     ABR4=ABR4,
        #     subject_data=subdata,
        #     subject_prefix="CBA_",
        #     output_file="CBA_Age_ABRs_Tones_combined.pdf",
        #     categorize="age",
        #     requested_stimulus_type="Tone",
        #     example_subjects=["CBA_F_N002_p27_NT", "CBA_M_N017_p572_NT"],
        #     test_plots = test_plots
        # )

        # with open("all_tone_map_data.pkl", "rb") as f:
        #     all_tone_map_data = pickle.load(f)
        # f, ax = mpl.subplots(1, 1)
        # plot_tone_map_data(ax, all_tone_map_data=all_tone_map_data)

    # mpl.show()
