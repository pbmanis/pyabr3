pyabr
=====

A python program to do abrs with TDT systems (RP2.1, PA5, and possibily with
RZ/RX processors) and an NI DAC.

Installation (Windows)
----------------------

Data acquisiton only works under windows.

Install Python 3.13 from the python.org site (don't use the Microsoft site)
Install TDT drivers from TDT (rpsvdx)
Install NIDAQ-mx (even if you are not using it!)
Install the active-x drivers from TDT. This may not be supported much longer however.

Run the installation script to create a virtual environment:
in the bash (git bash), run:

`
$ ./make_env.bat
`

Then activate the environment:

`
pyabr_venv\Scripts\activate
`

Run the test program:

To play a tone pip:

`
python tests/play_test_sounds.py pip
`

To run pyabr3:

From the repository root, run:

`python main.py`

Do not run `src/pyabr3.py` directly; `main.py` sets up the package import path and
selects the platform-appropriate configuration.

This will bring up a window with a panel on the left allowing access to parameters, loading protocols, recording, 
and a set of docks on the right showing the recorded responses, with a running average. 
Recording can be paused and restarted. 

The program is threaded, so that stimulus presentation and data acquisiton can occur simultaneously
with interactions with the graphics. The live graphs can be zoomed or panned; the graphs will however reset
on the next stimulus cycle. 

Installation (mac os)
---------------------
Note: Install on mac os only provides data analysis, not data acquisition.

Run the script:

```
./make_env.sh
```
 to create a working environment (do NOT do this as sudo).

Activate the environment:

```
source pyabr_venv/bin/activate
```

To start the test-mode interface, run `python main.py test` from the repository root.

Analysis
--------
Analysis is provided by the script *read_abr.py*. 
This program will need access to a configuration file (see the __main__ section for how this is set up).
As currently configured, it will be necessary to edit the read_abr.py file in order to
do analysis. Items such as paths to data, specifics about file names, etc, will have to be
adjusted for your configuration. 

The program expects the data to be in a specific structure under a directory:
experimental_directory_name:
    Subject_identifier (a directory)
        Clicks
            either matlab output (.txt files) or .p files from pyabr3
        Tones
            matlab output (.txt files)
        Interleaved_plateau
            pyabr3 output, .p files.

The names Clicks, Tones and Interleaved_plateau are hard-coded.
the subject identifier is expected to have a specific format:
"CBA_M_SUBID_p48_(date)_etc."

Where:
```
    CBA is the strain or cross identifier 
    M or F is the sex
    SUBID is a 1-2 letter identifier, followed by 1-3 numbers. 
    p48 is the age (in postnatal days)
    (date) is an optional date string, 
    or in some cases, a combined animal identifier.
```



The configuration file (in pyqtgraph configuration format) must have:

```
ABR_settings:
            HPF: 300   # high pass filter, Hz
            LPF: 3000  # low pass filter, Hz
            maxdur: 12.0   # msec
            stack_increment: 3.0  # spacing between stacked traces, in uV (per example)
            click_threshold_factor: 3.0  # threshold for ABR detection, clicks relative to the noise level
            tone_threshold_factor: 4.0  # threshold for ABR detection, tones relative to the noise level
        
        ABR_subject_excludes:
            CBA_F_R024_P13_NT_CBA24: "no responses"
            CBA_F_R026_P223_NT_CBA26: "old - replaced with current data "
            CBA_F_R027_P309_NT_CBA27: "old - replaced with current data"
            CBA_M_R019_P13_NT_CBA19: "something wacky with data"
            CBA_F_N003_p27_NT: "only 4 levels, 20 dB apart"
        
        # ABR parameters holds 
        ABR_parameters:
            default:
                minimum_latency: 2.2-3  # seconds
                invert: False
                scale: 1e-1
                fit_index: 2  # index of the first db level to fit in the sequence (assumes ascending)
            N0:  # subject id - most of the ones used in this project
                minimum_latency: 2.2e-3  # seconds
                invert: True
                scale: 1
                fit_index: 4
```
The settings are general settings related to filtering. plot duration and spacing,
and the automated click and tone threshold factors. The next set of 'ABR_subject_excludes'
is specifically to exclude some files in the datasets from the analysis. 
The ABR_parameters define the parameters used in analysis, such as the minimum latency
to the start of the P1 wave, whether the data needs to be inverted (or not), the
voltage scale factor, and the index of the first level of ascending SPLs to be used
for the regression line that is fit to the P1 and N1 latencies, to aid in the determination
of the threshold (and to align the data points for SPLs that are lower than
the threshold, or lower than can be properly identified by the peakdetect routine). For example,
if 10 sound pressure levels were presented (say 10 to 100 dB SPL in 10 dB increments), a value
of 5 would only use the data from 50 to 100 dB SPL for the linear regression, and
the responses at lower SPLS would be determined by extrapolation of the regression
to measure the voltage in the trace at an appropriate latency. Note that at these lower
SPLs, the voltage is taken as the mean in a small window.


Note that for the parameters, the defaults are read first, then specific subjects
are listed if there are to be any changes. All fields must be specified for each
subject. 

Included modules
----------------

It includes:

_sound.py_

A collection of routines that generate sound waveforms. Waveforms include clicks,
tone and noise pips, tone and noise trains, SAM tones and noise, dynamic ripple stimuli 
(though of limited capability), RSS stimuli, bandpass and notched noise. Test and display
routines are included.

_pystim.py_

    A module that interacts with some hardware (RP2.1, PA5, RZ5D, NIDAQ, and soundcards) to play
generated sounds.

_pysounds.py_

    An early verion module that only knows about the RP2.1, PA5, and NIDAQ. This is in the
*archived* folder.

_calreader.py_

    Read and display calibration files generated in matlab ABR program.

_stimController_

*This module suite provides the following in conjunction with TDT hardware:*

    1. Generation of waveforms for acoustic stimulation, including tones, clicks, noises, bandpass and notched noise, modulated tones and noises, RSS, DMR, and comodulated masking stimuli (not all are currently implemented).

    2. The suite uses a graphical window to show the waveforms and long-term spectra, as well as an option to display the frequency-time spectra for the stimuli. 

    3. The suite provides for the presentation of stimuli, including control of their timing, repetitions, etc. 

    4. The suite interfaces with system soundcards (for testing), National Instruments DAC card(s)*, and Tucker-Davis Technologies RP2.1, RZ5D DSP processors and PA5 attenuators.

    5. For recording responses to acoustic stimulation, the suite expects to connect to an RZ5D with a specific circuit that is loaded. In this setup, the RZ5D (using appropriately configured OpenWorkbench) generates trigger pulses for the stimuli, and records multiple channels from the preamplifier. The current suite in turn controls the RZ5D, starting the acquisition, and following the storage of data in the data tank. The goal is that most of the experimental interaction with the user takes place with the current suite, rather than through the RZ5D/OpenWorkbench/OpenScope windows, although these need to be monitored. Communication with the RZ5D is limited to setting up the standard CoreSweepControl timing parameters, and initiating mode changes (Record, Preview, Standby, Idle). 


This sofware does not provide:
------------------------------

1. Complex stimulus control.

This software does provide:
---------------------------

1. data acquisition
2. data analysis. However, the analysis module is somewhat customized, and the
script will need to be modified (e.g., paths, datasets, etc) in order for it to be useful.

Dependencies
------------

See the requirements.txt file for dependencies needed for installation
Note specifically the need for tdtpy, 

* Tested with NI6371 only.

Analysis
========

There are several different ways to analyze this data. 

The built-in analysis routines work, but are rather touchy and may require more programming to work for specific cases. 

The preferred method:
---------------------

The preferred method uses a 2-step "pipeline" starting with read_abr.py to convert the data to a format that can be used by the ABRA program (Erra et al. 2025, PMID:38948763).
This can be done by editing the "main" section of read_abr to read your data set. Note that reading the 
data sets depends on the _filename/directory_  structure, which is parsed by a regex. 

Below is example code for parsing in the main part of read_abr; the call to export_abra does the actual export.
Whether you exprort click or tone data depends on the value of "stim_type".

After parsing the files into a CSV format that is suitable for ABRA, 
run plot_ABRA_csv.py to plot the data. The "__main__" routine in plot_ABRA_csv.py
will need to be edited to point to the right files and to select what will be plotted. 


'''
    AR = AnalyzeABR()
    ABR4 = read_abr4.READ_ABR4()

    config_file_name = "Path to an ephys config file/config/experiments.cfg"
    expt = "GlyT2_NIHL"  # definition of experiment in the config file
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
        if tests:  # pick a few subjects to run before running the whole thing.
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
        test_plots = False

        do_click_io_analysis(
            AR=AR,
            ABR4=ABR4,
            subject_data=subdata,
            subject_prefix="CBA_",  # must be set to the start of the directory name holding the data
            output_file="CBA_Age_ABRs_Clicks_combined.pdf",  # may not be generated ?
            categorize="age_category",
            requested_stimulus_type="Click",
            # example_subjects=["CBA_F_N002_p27_NT", "CBA_M_N017_p572_NT"],
            test_plots=test_plots,
        )

    if expt == "GlyT2_NIHL":
        AR.get_experiment(config_file_name, "GlyT2_NIHL")
        directory_names = {  # values are symbol, symbol size, and relative gain factor
            "/Volumes/Pegasus_004/ManisLab_Data3/abr_data/Reggie_NIHL": ["o", 3.0, 1.0],
        }

        subdata = get_datasets(directory_names, filter="VGATEYFP")
        print("Subdata: ", subdata.keys())
        # select subjects for tuning analysis parameters in the configuration file.
        tests = False
        if tests:
            stim_type = "Click"
            test_subjs = ["UJ7"]  # , "XT9", "XT11", "N007"]
            newsub = {stim_type: []}
            # print(subdata.keys())
            for sub in subdata:
                # print(sub)
                for d in subdata[sub]:
                    # print(d)
                    if d["subject"] in test_subjs:
                        newsub[stim_type].append(d)

            subdata = newsub

        test_plots = False

        # uncomment this to write the click csv files for the ABRA program.
        # for subj in subdata['Click']:
        #     print("Subject: ", subj
        #     )
        #     export_for_abra(AR=AR, ABR4=ABR4, subj = subj, requested_stimulus_type="Click")
        stim ="Tone"
        stim = "Interleaved_plateau"
        stim="Click"
        # print("\nSubdata IP: ", subdata[stim])
        # print("\nSubdata Click: ", subdata["Click"])
        # print("\nSubdata Tone: ", subdata["Tone"])
        for n, subj in enumerate(subdata[stim]):
            print("Subject: ", subj["subject"])
            export_for_abra(AR=AR, ABR4=ABR4, subj=subj, requested_stimulus_type=stim)
    '''
