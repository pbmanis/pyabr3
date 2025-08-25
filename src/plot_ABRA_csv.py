from pathlib import Path
import pandas as pd
import matplotlib.pyplot as mpl
import numpy as np
import seaborn as sns
from typing import Union
import pylibrary.plotting.plothelpers as PH
from sklearn.cluster import KMeans
from kneed import KneeLocator

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

import analyze_abr
AR = analyze_abr.AnalyzeABR()

mpl.rcParams["text.latex.preamble"] = r"\DeclareUnicodeCharacter{03BC}{\ensuremath{\mu}}"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 10

# palette = sns.color_palette("tab10")[:3]
""" Takes output from the ABRA program (github.com/manorlab)
"""


def assign_treatment(df):
    df["treatment"] = {}

    def apply_treatment(row):
        if "Filename" in row.keys():
            rf = row.Filename
        elif "File Name" in row.keys():
            rf = row["File Name"]
        if "Sham" in rf:
            return "Sham"
        elif "NE2wks106" in rf:
            return "NE2wks106"
        elif "NE2wks115" in rf:
            return "NE2wks115"
        else:
            return "Unknown"

    df["treatment"] = df.apply(apply_treatment, axis=1)
    return df


def assign_coding(df, coding: Union[list, None] = None):
    """
    Assign a coding column to the DataFrame based on the provided coding list.
    If coding is None, no coding column is added.
    """
    if coding is not None:
        df["coding"] = df["subject"].apply(lambda x: x in coding)
    else:
        df["coding"] = False
    return df


def assign_subject(df):
    df["subject"] = {}

    def apply_subject(row):
        if "Filename" in row.keys():
            rf = row.Filename
        elif "File Name" in row.keys():
            rf = row["File Name"]
        subject = rf.split("_")[2]
        return subject

    df["subject"] = df.apply(apply_subject, axis=1)
    return df


def assign_cross(df, cross_key="NF107"):
    df["cross"] = {}

    def apply_cross(row):
        if "Filename" in row.keys():
            rf = row.Filename
        elif "File Name" in row.keys():
            rf = row["File Name"]
        if rf.find(cross_key) > 0:
            return cross_key
        else:
            return "C57Bl/6"

    df["cross"] = df.apply(apply_cross, axis=1)
    return df


def split_groups(
    split: bool = True,
    group_col="treatment",
    split_col="cross",
    cross_order=["C57Bl/6", "NF107"],
    treat_order=["Sham", "NE2wks106", "NE2wks115"],
):
    """
    Split the groups based on the treatment and cross columns.
    If split is True, use the split_col for hue, otherwise use group_col.
    """
    if split:
        hue = split_col
        hue_order = cross_order
        dodge = True
    else:
        hue = group_col
        hue_order = treat_order
        dodge = False
    return hue, hue_order, dodge


def select_subjects(
    df: pd.DataFrame, coding: pd.DataFrame, selection: Union[str, list, None] = None
):
    assert isinstance(coding, pd.DataFrame)
    if "File Name" in df.columns:
        all_subject_files = list(df["File Name"].unique())
    else:  # abr analysis is not consistent in naming conventions
        all_subject_files = list(df["Filename"].unique())
    all_subjects = [f.split("_")[2] for f in all_subject_files]
    # print("all_subjects: ", all_subjects)
    # print("select subjects: ", selection)
    # print("all subjects in dataframe: ", len(all_subjects), all_subjects)
    coding = coding[coding["Subject"].notna()]
    coded_subjects = coding["Subject"].tolist()
    df["experiment"] = {}
    # print("all coded: ", len(coded_subjects), coded_subjects)
    match selection:
        case "ephys":
            subjects = list(set(all_subjects).intersection(set(coded_subjects)))
            df = df[df["subject"].isin(subjects)]
            df["experiment"] = "ephys"
        case "anatomy":
            # subjects = list(set(all_subjects).difference(set(coded_subjects)))
            df = df[~df["subject"].isin(coded_subjects)]
            subjects = df["subject"].unique().tolist()
            df["experiment"] = "anatomy"
        case x if isinstance(x, list):
            df = df[df["subject"].isin(selection)]
            subjects = selection
        case None | "None" | "all" | "All":
            subjects = all_subjects
            for subject in subjects:
                if subject in coded_subjects:
                    df.loc[df["subject"] == subject, "experiment"] = "ephys"
                else:
                    df.loc[df["subject"] == subject, "experiment"] = "anatomy"
        case _:
            raise ValueError(
                "select_subjects: subjects must be 'ephys', 'anatomy', 'all' or a list of subject IDs, got: {subjects!r}"
            )

    # print("selection: ", selection)
    # print("selected subjects: ", subjects)
    # exit()
    return df, subjects


def plot_thresholds(
    filename: Union[str, Path],
    coding: pd.DataFrame,
    ax=None,
    palette=None,
    treat_order=None,
    # split: bool = False,
    # split_col="cross",
    # cross_order=["C57Bl/6", "NF107"],
    # plottype: str = "bar",
    **kwargs,
):
    plottype = kwargs.get("plottype", "bar")
    cross_order = kwargs.get("cross_order", ["C57Bl/6", "NF107"])
    split_col = kwargs.get("split_col", "cross")
    treat_order = kwargs.get("treat_order", ["Sham", "NE2wks106", "NE2wks115"])
    assert plottype in ["bar", "box"]
    selection = kwargs.get("selection", None)
    # Define the path to the CSV file
    fn = filename
    df = pd.read_csv(fn)
    df = assign_treatment(df)
    df = assign_subject(df)
    if split_col == "cross":
        df = assign_cross(df, "NF107")
    hue, hue_order, dodge = split_groups(
        split=split,
        group_col="treatment",
        split_col=split_col,
        cross_order=cross_order,
        treat_order=treat_order,
    )
    print("hue, hue_order, dodge: ", hue, hue_order, dodge)
    df, subjects = select_subjects(df, coding=coding, selection=selection)

    if ax is None:
        f, ax = mpl.subplots(1, 1)
    match plottype:
        case "bar":
            kwds = {"errorbar": ("sd", 1), "saturation": 0.45}
            fn = sns.barplot
        case "box":
            kwds = {"saturation": 0.6}
            fn = sns.boxplot
        case _:
            raise ValueError(f"Unknown plot type: {plottype}")
    fn(
        x="treatment",
        y="Threshold",
        data=df,
        order=treat_order,
        hue_order=hue_order,
        hue=hue,
        palette=palette,
        ax=ax,
        **kwds,
    )
    if coding is not None:
        print("row names: ", df.columns)
        print(df["Filename"].unique())
        df["marker"] = df.apply(lambda row: "^" if row["subject"] in coding else "o", axis=1)
    else:
        df["marker"] = "o"  # assign to ALL subjects
    print("df['treatment'].unique(): ", df["treatment"].unique())

    df = df[df["treatment"] != "Unknown"]
    print("df['treatment'].unique(): ", df["treatment"].unique())
    print("hue order: ", hue_order)
    if hue_order is None:
        hue = "treatment"
        hue_order = df["treatment"].unique()
    else:
        hue_order = [x for x in hue_order if x in df["treatment"].unique()]
    if len(hue_order) == 0:
        hue_order = df["treatment"].unique()
    print("hue_order: ", hue_order)
    print("selection: ", selection)
    if selection == 'all':
        markerdict = {"ephys": "o", "anatomy": "^"}
    else:
        markerdict = "o"
    print("markerdict: ", markerdict)
    sns.swarmplot(
        x="treatment",
        y="Threshold",
        data=df,
        order=treat_order,  # treat_order,
        palette=palette,
        hue=hue,
        hue_order=hue_order,
        # markersize=4,
        # marker=['o', '^'], # markerdict,
        alpha=1,
        linewidth=0.25,
        edgecolor="black",
        dodge=dodge,
        ax=ax,
    )
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, loc="upper left", ncol=1, frameon=True)
    PH.nice_plot(ax, direction="outward")
    ax.set_xticklabels(["Sham", "106\ndB SPL", "115\ndB SPL"])
    # mpl.title("Thresholds by Treatment")
    # mpl.xlabel("Treatment")
    ax.set_ylabel("Threshold (dB SPL)")
    ax.set_xlabel("")


def plot_amplitude_data(
    filename,
    stim_type: str,
    palette=None,
    treat_order=None,
    coding: pd.DataFrame = None,
    **kwargs,
):
    plottype = kwargs.get("plottype", "bar")
    selection = kwargs.get("selection", None)
    split_col = kwargs.get("split_col", "cross")
    cross_order = kwargs.get("cross_order", ["C57Bl/6", "NF107"])

    ax = kwargs.get("ax", None)
    assert plottype in ["bar", "box"]
    # Define the path to the CSV file
    fn = filename
    df = pd.read_csv(fn)
    df = assign_treatment(df)
    if split_col == "cross":
        df = assign_cross(df, "NF107")
    df = assign_subject(df)
    df, subjects = select_subjects(df, coding=coding, selection=selection)
    # print("selection: ", selection)
    # print(df.head())
    # print(df.subject.unique())
    # print("subjects: ", subjects)
    # exit()
    hue, hue_order, dodge = split_groups(
        split=split,
        group_col="treatment",
        split_col=split_col,
        cross_order=cross_order,
        treat_order=treat_order,
    )
    # print(df.columns)
    # print(sorted(df.subject.unique()))
    # exit()
    dfp = pd.DataFrame(columns=["subject", "cross", "treatment", "maxWave1"])

    for subject in subjects:  # df["File Name"].unique():
        subdf = df[df["subject"] == subject]
        max_wave1 = subdf["Wave I amplitude (P1-T1) (μV)"].max()
        try:
            treatment = subdf["treatment"].iloc[0]
            # print("Subject: ", subject, " Treatment: ", treatment)
        except IndexError:
            print(f"Warning: No treatment found for subject {subject} in {fn}")
            print(subdf.head())
            raise IndexError(
                f"Subject {subject} not found in DataFrame or treatment missing in {fn}"
            )
        dfp = dfp._append(
            {
                "subject": subject,
                "cross": subdf["cross"].iloc[0],
                "treatment": treatment,
                "maxWave1": max_wave1,
            },
            ignore_index=True,
        )
    if ax is None:
        f, ax = mpl.subplots(1, 1)
    match plottype:
        case "bar":
            kwds = {"errorbar": ("sd", 1), "saturation": 0.45}
            func = sns.barplot
        case "box":
            kwds = {"saturation": 0.6}
            func = sns.boxplot
        case _:
            raise ValueError(f"Unknown plot type: {plottype}")
    func(
        x="treatment",
        y="maxWave1",
        data=dfp,
        palette=palette,
        ax=ax,
        order=treat_order,
        hue_order=hue_order,
        hue=hue,
        **kwds,
    )
    sns.swarmplot(
        x="treatment",
        y="maxWave1",
        data=dfp,
        palette=palette,
        order=treat_order,
        hue_order=hue_order,
        hue=hue,
        alpha=0.8,
        linewidth=0.25,
        edgecolor="black",
        dodge=dodge,
        ax=ax,
    )
    ax.set_ylim(0, 10)
    
    ax.legend(fontsize=7, loc="upper right", ncol=1, frameon=True)
    PH.nice_plot(ax, direction="outward")
    # mpl.title("Wave I Amplitude by Treatment")
    # mpl.xlabel("Treatment")
    ax.set_xticklabels(["Sham", "106\ndB SPL", "115\ndBSPL"])
    ax.set_ylabel("Maximum P1-T1 Amplitude (μV)")
    ax.set_xlabel("")
    # mpl.xticks(rotation=45)


def plot_thr_amp_data(
    filename: Union[str, Path], ax=None, palette=None, treat_order=None, split: bool = False
):
    fn = filename
    df = pd.read_csv(fn)
    df = assign_treatment(df)
    df = assign_subject(df)
    df = assign_cross(df, "NF107")
    hue, hue_order, dodge = split_groups(
        split=split,
        group_col="treatment",
        split_col="cross",
        cross_order=["C57Bl/6", "NF107"],
        treat_order=treat_order,
    )
    print("hue: ", hue)
    print(df.columns)
    dfp = pd.DataFrame(columns=["subject", "cross", "treatment", "maxWave1", "threshold"])
    for sub in df["File Name"].unique():
        subdf = df[df["File Name"] == sub]
        max_wave1 = subdf["Wave I amplitude (P1-T1) (μV)"].max()
        treatment = subdf["treatment"].iloc[0]
        threshold = subdf["Estimated Threshold"].iloc[0]
        dfp = dfp._append(
            {
                "subject": sub,
                "cross": subdf["cross"].iloc[0],
                "treatment": treatment,
                "maxWave1": max_wave1,
                "threshold": threshold,
            },
            ignore_index=True,
        )
    if ax is None:
        f, ax = mpl.subplots(1, 1)

    # principalDf = compute_pca(ax, dfp)
    # sns.scatterplot(data=principalDf, x="PC1", y="PC2", alpha=0.5, palette=palette, hue=hue, hue_order=hue_order, ax=ax)
    # sns.scatterplot(data=principalDf, x='x_scaled', y='y_scaled', alpha=0.5, style=hue,
    #                 markers=['s', 'h', 'D'], hue=hue, hue_order=hue_order, edgecolor="black", linewidth=0.3, ax=ax)
    sns.scatterplot(
        data=dfp,
        x="threshold",
        y="maxWave1",
        style="",
        markers=["o", "s"],
        size="cross",
        sizes=[28, 32],
        hue=hue,
        hue_order=hue_order,
        palette=palette,
        ax=ax,
    )
    compute_kmeans_clusters(ax, dfp)

    PH.nice_plot(ax, direction="outward")
    mpl.title("PCA of Thresholds and Wave I Amplitude")
    mpl.xlabel("Threshold (dB SPL)")
    mpl.ylabel("Wave I Amplitude (μV)")
    ax.set_ylim(0, 8)
    ax.set_xlim(0, 100)


def compute_pca(ax, dfp):
    # Perform PCA
    features = ["threshold", "maxWave1"]
    print(dfp["treatment"].unique())

    def map_treat(row):
        if row["treatment"] == "Sham":
            return 0
        elif row["treatment"] == "NE2wks106":
            return 1
        elif row["treatment"] == "NE2wks115":
            return 2
        else:
            return 3

    dfp["treatment"] = dfp.apply(map_treat, axis=1)
    x = dfp[features].values
    x_scaled = StandardScaler().fit_transform(x)  # Normalize the data

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x_scaled)
    principalDf = pd.DataFrame(data=principalComponents, columns=["PC1", "PC2"])
    principalDf["treatment"] = dfp["treatment"]
    principalDf["cross"] = dfp["cross"]
    principalDf["subject"] = dfp["subject"]
    principalDf["x_scaled"] = x_scaled[:, 0]
    principalDf["y_scaled"] = x_scaled[:, 1]
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(fontsize=7, loc="upper right", ncol=1, frameon=True)
    return principalDf


def compute_kmeans_clusters(ax, dfp):
    # kmeans clustering ?
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }
    X = dfp[["threshold", "maxWave1"]].values

    # A list holds the SSE values for each k
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    f2, ax2 = mpl.subplots(1, 1)
    mpl.plot(range(1, 11), sse, marker="o")
    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    print("Knee point: ", kl.elbow)

    km = KMeans(n_clusters=3, **kmeans_kwargs)
    km_result = km.fit(X)
    print(km_result.labels_)
    print(km_result.cluster_centers_)
    ax.plot(
        km_result.cluster_centers_[:, 0],
        km_result.cluster_centers_[:, 1],
        marker="x",
        color="red",
        linestyle="",
        markersize=8,
        label="Centroids",
    )
    for i in range(len(X)):
        ax.text(
            X[i, 0] - 1,
            X[i, 1],
            str(km_result.labels_[i]),
            fontsize=9,
            color="black",
            ha="right",
            va="center",
        )


def plot_IO_data(
    filename: Union[str, Path],
    coding: pd.DataFrame,
    stim_type: str,
    ax=None,
    palette=None,
    treat_order=None,
    individual=False,
    split: bool = False,
    **kwargs,
    # split_col:Union[str, None]="cross",
    # cross_order:Union[list, None]=None,
    # plottype: str = "bar",
):
    # Define the path to the CSV file
    assert isinstance(coding, pd.DataFrame)
    selection = kwargs.get("selection", None)

    fn = filename
    df = pd.read_csv(fn)
    df = assign_treatment(df)
    df = assign_cross(df, "NF107")
    # fix inconsistent column names in output from abraanalysis:
    if "File Name" not in df.columns:
        df.rename(columns={"Filename": "File Name"}, inplace=True)
    if "Wave I amplitude (P1-T1) (μV)" not in df.columns:
        df.rename(columns={"Wave I amplitude (P1-T1) (μV)": "Wave1_amplitude"}, inplace=True)
    df = assign_subject(df)

    # print("coding: ", coding)
    # pick out the selected data using the selection criterion
    df, subjects = select_subjects(df, selection=selection, coding=coding)

    dfp = pd.DataFrame(
        columns=[
            "subject",
            "cross",
            "treatment",
            "Stimulus",
            "Frequency (Hz)",
            "Wave1_amplitude",
            "dB SPL",
        ]
    )

    freqs = df["Frequency (Hz)"].unique()
    freq_order = [float(x) for x in sorted(freqs)]
    treats = df["treatment"].unique()
    if len(freqs) == 1 and freqs == 100.0:
        if stim_type != "Click":
            raise ValueError("Stimulus type must be 'Click' for 100 Hz frequency data")
        colors = sns.color_palette(palette, len(treats))
        if ax is None and not individual:
            f, ax = mpl.subplots(1, 1)
            axn = [ax]
        elif ax is None and individual:
            r, c = PH.getLayoutDimensions(len(subjects))
            P = PH.regular_grid(
                r,
                c,
                order="rowsfirst",
                figsize=(17, 11),
                verticalspacing=0.05,
                horizontalspacing=0.03,
            )
            # f, ax = mpl.subplots(r, c, figsize=(17, 11))
            ax = P.axarr
            axn = ax.ravel()
            for a in axn:
                PH.nice_plot(a)
        else:
            axn = [ax]
    else:
        if stim_type != "Tone":
            raise ValueError(
                "Stimulus type must be 'Tones' for tone pip data with multiple frequencies and freq != 100 Hz"
            )
        r, c = PH.getLayoutDimensions(len(subjects))
        fig, ax = mpl.subplots(r, c, figsize=(11, 8))
        axn = ax.ravel()
        for a in axn:
            PH.nice_plot(a)
        colors = sns.color_palette(palette, len(freq_order))

    # build a smaller dataframe for each subject, to put into a
    # reduced one for plotting.
    for subject in subjects:
        subdf = df[df["subject"] == subject]
        wave1 = np.array(subdf["Wave I amplitude (P1-T1) (μV)"].values)
        dbspl = np.array(subdf["dB Level"].values)
        if subject.find("WJ9") > 0:
            dbspl = dbspl[:15]
            wave1 = wave1[:15]
            # print("dbspl: ", sub, dbspl)
            # print("wave1: ", sub, wave1)

        treatment = subdf["treatment"].iloc[0]
        dfp = dfp._append(
            {
                "subject": subject,
                "cross": subdf["cross"].iloc[0],
                "treatment": treatment,
                "Wave1_amplitude": wave1,
                "dB SPL": dbspl,
                "Frequency (Hz)": subdf["Frequency (Hz)"],
            },
            ignore_index=True,
        )

    tmap = [2, 1, 0]

    npl = 0
    for i, treat in enumerate(treats):
        print("i: ", i, treat)
        if treat == "Unknown":
            continue
        data = df[df["treatment"] == treat]

        for ns, s in enumerate(data["subject"].unique()):
            subdata = data[data["subject"] == s]

            if s.find("NF107") > 0:
                marker = "s"
            else:
                marker = "o"
            if coding is not None:
                if s in coding:
                    marker = "o"
                else:
                    marker = "^"

            if stim_type == "Click":
                color = colors[tmap[i]]
                frqs = [100.0]
                if individual:
                    ax = axn[npl]
                    npl += 1
                else:
                    ax = ax
            elif stim_type == "Tone":
                frqs = [float(f) for f in subdata["Frequency (Hz)"]]
                frx = int(freq_order.index(frqs[ns]))
                color = colors[frx]
                ax = axn[npl]
                npl += 1
            else:
                color = "k"
            if len(subdata) > 0:
                # reassemble data by frequency (split the dB SPL and Wave1_amplitude into separate arrays by frequency))
                db = []
                amp = []
                u_freq = pd.Series(frqs).unique()
                # pd.set_option('display.max_columns', None)
                for ifr, freq in enumerate(u_freq):
                    freq = int(freq)
                    this_fr = subdata[subdata["Frequency (Hz)"] == freq]
                    db = np.array(this_fr["dB Level"])
                    amp = np.array(this_fr["Wave I amplitude (P1-T1) (μV)"])
                    if stim_type == "Click" and individual:
                        labl = None
                        lw = 0.5
                        alpha = 1
                    elif stim_type == "Click" and not individual:
                        labl = None, # f"{treat} {s}"
                        lw = 0.5
                        alpha = 0.5
                    elif stim_type == "Tone":
                        labl = f"{freq/1000} kHz"
                        lw = 1.5
                        color = colors[freq_order.index(freq)]
                        alpha = 0.5
                    else:
                        raise ValueError("Unknown mode: {}".format(mode))
                    ax.plot(
                        db,
                        amp,
                        marker=marker,
                        color=color,
                        label=labl,
                        linewidth=lw,
                        markersize=0,
                        alpha=alpha,
                    )
            if stim_type == "Tone" or (stim_type == "Click" and individual):
                sshort = "_".join(s.split("_")[0:5])
                ax.set_title(f"{sshort}\n{treat}", fontsize=6)

            ax.legend(fontsize=5, loc="upper left", ncol=1, frameon=False)
            if stim_type == "Click" and not individual:
                sns.lineplot(data=data, x="dB Level", y="Wave I amplitude (P1-T1) (μV)", hue="treatment", 
                             palette=palette, ax=ax, linewidth=1.5, markersize=0,
                            err_style="band", errorbar=("sd", 1), err_kws={"alpha": 0.01}, hue_order=treat_order, legend=False,
                            )
    if stim_type == "Click":
        hue = "treatment"
        hue_order = treat_order
        for a in axn:
            a.set_ylim(-0.5, 10.0)
            a.set_xlim(20, 100)

    elif stim_type == "Tone":
        hue = "Frequency (Hz)"
        hue_order = freq_order
        for a in axn:
            a.set_ylim(-0.5, 4)

    # if not split:
    #     kwds = {"style": None, "style_order": None, "markers": ["o"]}
    #     for ax in axn:
    #         sns.lineplot(
    #             x="dB Level",
    #             y="Wave I amplitude (P1-T1) (μV)",
    #             hue=hue,
    #             hue_order=hue_order,
    #             data=df,
    #             palette=palette,
    #             ax=ax,
    #             **kwds,
    #         )
    # else:
    #     kwds = {"style": "cross", "style_order": ["C57Bl/6", "NF107"], "markers": ["o", "s"]}
    axn[-1].legend(fontsize=5, loc="upper left", ncol=1, frameon=True)
    axn[-1].set_xlabel("dB SPL")

    label = r"P1-T1 Amplitude (μV)"
    axn[-1].set_ylabel(label)
    PH.nice_plot(axn[-1], direction="outward")
    # axn[-1].set_xticks(rotation=45)
    mpl.suptitle(f"ABR {stim_type} Data for {selection} Subjects", fontsize=8)
    # PH.nice_plot(ax, direction="outward")


def plot_tone_thresholds(
    filename: Union[Path, str],
    ax=None,
    palette=None,
    treat_order=None,
    coding: Union[str, None] = None,
    **kwargs,
):
    # Define the path to the CSV file
    fn = filename
    df = pd.read_csv(fn)
    selection = kwargs.get("selection", None)
    print("selection: ", selection)

    df = assign_treatment(df)
    df = df[df["Frequency"] != 24000]
    df = assign_subject(df)
    df = assign_coding(df, coding)
    df, subjects = select_subjects(df, coding=coding, selection=selection)
    # print(selection, sorted(df["subject"].unique()))

    if ax is None:
        f, ax = mpl.subplots(1, 1, figsize=(6, 6))

    # df['Frequency'] = np.log10(df['Frequency'])
    # sns.swarmplot(x="Frequency", y="Threshold", data=df,    hue="treatment",
    #     hue_order=treat_order,
    #     alpha=1.0,
    #     linewidth=0.25,
    #     edgecolor="black", ax=ax)
    # sns.boxplot(x="Frequency", y="Threshold", data=df, hue="treatment", hue_order=treat_order, palette=palette, ax=ax, saturation=0.6)
    sns.lineplot(
        x="Frequency",
        y="Threshold",
        data=df,
        hue="treatment",
        hue_order=treat_order,
        palette=palette,
        ax=ax,
        errorbar=("sd", 1),
        linewidth=1.5,
    )
    fr_grid_ep: dict = {(float, float): int}  # keys are frequency and putative threshold
    fr_grid_an: dict = {(float, float): int}
    freqs = sorted(df["Frequency"].unique())
    thrs = np.linspace(0, 100, 11)  # 10 db steps
    for f in freqs:
        for t in thrs:
            fr_grid_ep[(f, t)] = 0.0
            fr_grid_an[(f, t)] = 0.0

    for s in df["Filename"].unique():
        subdf = df[df["Filename"] == s]
        print("subdf: ", subdf)
        if subdf["experiment"].iloc[0] == "ephys":
            mk = "^"
            delx = -1
            # fr_grid_ep
        else:
            mk = "o"
            delx = 1
        if len(subdf) > 0:
            # this plots the points, but it is hard to distinguish them.
            # ax.plot(
            #     subdf["Frequency"]
            #     + 0.2*delx
            #     # + np.random.uniform(0, subdf["Frequency"] * 0.15, len(subdf["Frequency"])),
            #     + subdf["Frequency"] * 0.15,
            #     subdf["Threshold"],
            #     marker=mk,
            #     color=palette[treat_order.index(subdf["treatment"].values[0])],
            #     linewidth=0,
            #     markersize=3.5,
            #     alpha=0.6,
            # )

            # check each threshold point:
            for i, row in subdf.iterrows():
                freq = row["Frequency"]
                thr = row["Threshold"]
                if np.isnan(thr):
                    continue  # skip NaN thresholds
                frdelta = np.logspace(np.log10(freq), np.log10(freq)*1.0002, endpoint=True, num=2, base=10)[1]
                if row["experiment"] == "ephys":
                    fr_grid_ep[(freq, thr)] += frdelta
                    mk = "o"
                    xoffset = fr_grid_an[(freq, thr)]
                else:
                    fr_grid_an[(freq, thr)] -= frdelta
                    mk = "^"
                    xoffset = fr_grid_an[(freq, thr)]
                print(thr, freq, xoffset, fr_grid_ep[(freq, thr)], fr_grid_an[(freq, thr)])
                ax.plot(
                    freq + xoffset,
                    thr,
                    marker=mk,
                    color=palette[treat_order.index(row["treatment"])],
                    linewidth=0,
                    markersize=4.5,
                    alpha=0.6,
                )
            # sns.stripplot(
            #     x="Frequency",
            #     y="Threshold",
            #     data=subdf,
            #     hue="treatment",
            #     hue_order=treat_order,
            #     palette=palette,
            #     alpha=0.6,
            #     linewidth=0.25,
            #     edgecolor="black",
            #     ax=ax,
            #     native_scale=True,
            #     log_scale= True,
            #     jitter=20,
            #     dodge=True,
            # )

    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    print("subjects: ", subjects)
    # subjects = df["subject"].unique()
    subjects = [s for s in subjects if pd.notna(s) and s != "nan"]
    if subjects is not None:
        if isinstance(subjects, list):
            subjlist = ", ".join(subjects)
        else:
            subjlist = subjects
        mpl.suptitle(f"Tone Thresholds for Subjects: {subjlist}", fontsize=8)
    else:
        mpl.suptitle("Tone Thresholds, all Subjects")
    # mpl.title("Thresholds by Frequency")
    mpl.xlabel("Frequency")
    mpl.ylabel("Threshold (dB SPL)")
    mpl.xticks(rotation=45)


if __name__ == "__main__":
    # Example usage of the plotting functions
    # This is a script to plot ABR data from CSV files exported from ABRA.
    # It can plot click and tone data, and can split the data by treatment and cross.
    # The data is expected to be in a specific format, with columns for treatment, subject, cross, etc.
    # The script uses seaborn and matplotlib for plotting.
    coding_file = Path(
        "/Volumes/Pegasus_004/ManisLab_Data3/Edwards_Reginald/RE_datasets/GlyT2_EGFP_NIHL/GlyT2_NIHL_Coding.xlsx"
    )
    re_cba_path = Path("/Users/pbmanis/Desktop/Python/RE_CBA")
    if not re_cba_path.is_dir():
        raise ValueError(f"Path {re_cba_path} is not a directory")
    exp_path = Path(re_cba_path, "config", "experiments.cfg")
    AR.get_experiment(exp_path, "GlyT2_NIHL")

    print("Reading coding file: ", coding_file, coding_file.is_file())
    if coding_file.is_file():
        coding_df = pd.read_excel(coding_file, sheet_name="coding")
        ephys_subjects = coding_df["Subject"].unique()
    else:
        ephys_subjects = None

    # stim_type = "Click"
    stim_type = "Click"
    treat_order = ["Sham", "NE2wks106", "NE2wks115"]
    plot_colors = AR.experiment['plot_colors']

    palette = sns.color_palette([plot_colors['bar_background_colors'][t] for t in treat_order], n_colors=len(treat_order))
    strain = "GlyT2"
    selection = "all"  # "ephys"  # or "ephys"  # or "all" or list of subjects like ["WN8", "XR10"].
    # strain = "VGATEYFP"
    split_col = None
    cross = None
    split = False
    show_individual = False

    match stim_type:
        case "Click":
            f, ax = mpl.subplots(1, 3, figsize=(9, 4))
            f.suptitle(f"{strain} ABR Click Data ( {selection} )", fontsize=16)

            if strain == "GlyT2":
                click_IO_file = Path(
                    # re_cba_path, "abra", "GlyT2EGFP_2025-08-07T15-39_export_click_IO.csv"
                    re_cba_path, "abra", "GlyT2EGFP_2025-08-25T16-54_export_click_IO.csv"
                )
                click_threshold_file = Path(
                    # re_cba_path, "abra", "GlyT2EGFP_2025-08-07T15-39_export_click_thresholds.csv"
                    re_cba_path, "abra", "GlyT2EGFP_2025-08-25T16-53_export_click_thresholds.csv"
                )
                cross = ["C57Bl/6", "NF107"]
                split_col = "cross"
            elif strain == "VGATEYFP":
                click_IO_file = Path(
                    re_cba_path, "abra", "VGATEYFP_2025-07-23T17-50_export_click_IO.csv"
                )
                click_threshold_file = Path(
                    re_cba_path, "abra", "VGATEYFP_2025-07-23T17-49_export_click_thresholds.csv"
                )
                cross = None
                split_col = None
            if show_individual:
                kwds = {
                    "individual": True,
                    "ax": None,
                    "split": True,
                    "split_col": split_col,
                    "cross_order": cross,
                    "selection": selection,
                    "plot_colors": plot_colors,
                }
                plot_IO_data(
                    filename=click_IO_file,
                    stim_type=stim_type,
                    palette=palette,
                    treat_order=treat_order,
                    coding=coding_df,
                    **kwds,
                )
            else:
                kwds = {
                    "individual": False,
                    "ax": ax[0],
                    "split": False,
                    "split_col": split_col,
                    "cross_order": cross,
                    "selection": selection,
                    "plot_colors": plot_colors,
                }
            kwds2 = {
                "individual": False,
                "ax": ax[0],
                "split": False,
                "split_col": split_col,
                "cross_order": cross,
                "selection": selection,
                "plot_colors": plot_colors,
            }
            # now the 3 panel plot
            kwds3 = kwds2.copy()
            kwds3["ax"] = ax[2]
            kwds3.pop("individual", None)
            kwds_amp = kwds2.copy()
            kwds_amp["ax"] = ax[1]
            kwds_amp["plottype"] = "bar"

            plot_IO_data(
                filename=click_IO_file,
                stim_type=stim_type,
                palette=palette,
                treat_order=treat_order,
                coding=coding_df,
                **kwds2,
            )

            plot_amplitude_data(
                filename=click_IO_file,
                stim_type=stim_type,
                palette=palette,
                treat_order=treat_order,
                coding=coding_df,
                **kwds_amp,
            )
            plot_thresholds(
                filename=Path(click_threshold_file),
                coding=coding_df,
                stim_type=stim_type,
                palette=palette,
                treat_order=treat_order,
                **kwds3,
            )
            # mpl.tight_layout()
            # plot_thr_amp_data(filename=click_IO_file, ax=None, palette=palette, treat_order=treat_order,
            #                   split= False)
            mpl.show()

        case "Tone":

            split = True  # split the data by cross
            cwd = Path.cwd()
            # print("abra path: ", Path(re_cba_path, "abra", "Tones").is_dir())
            tonefile = Path(
                re_cba_path, "abra", "GlyT2EGFP_2025-08-07T16-00_export_tone_thresholds.csv"
            )
            toneamplitudefile = Path(
                re_cba_path, "abra", "GlyT2EGFP_2025-08-07T16-09_export_tone_IO.csv"
            )
            # print(tonefile.is_file(), toneamplitudefile.is_file())

            # subjects = ["WN8"]
            print("Selection: ", selection)
            if selection is None or selection == "None":
                exit()
            kwds = {
                "individual": False,
                "ax": None,
                "split": False,
                "split_col": split_col,
                "cross_order": cross,
                "selection": selection,
                "plot_colors": plot_colors,
            }
            kwds_tone = kwds.copy()
            plot_tone_thresholds(
                filename=tonefile,
                palette=palette,
                treat_order=treat_order,
                coding=coding_df,
                **kwds,
            )
            frpalette = sns.color_palette("nipy_spectral_r", 7)
            plot_IO_data(
                filename=toneamplitudefile,
                stim_type=stim_type,
                palette=frpalette,
                treat_order=treat_order,
                coding=coding_df,
                **kwds_tone,
            )
            mpl.tight_layout()
            mpl.show()
