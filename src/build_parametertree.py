# Define parameters that control aquisition and buttons for pyabr3

from pyqtgraph.parametertree import Parameter, ParameterTree

def build_parametertree(known_protocols, current_protocol, stimuli:dict, data_path_names=("default",)):
    """Build the control panel.

    Bug: the Parameters group looked editable, but edits were discarded, and its limits
    clamped the displayed values (a 0.0001 s click showed as 0.001 s). Protocols are meant
    to be used as saved, so the group is now read-only, without limits. See also buildGUI
    in pyabr3.py, which now shows the protocol's values at startup.
    """
    data_path_choices = [str(name) for name in data_path_names]
    if len(data_path_choices) > 1:  # with several locations, the user must choose one
        data_path_choices.insert(0, "Select...")
    params = [
        {
            "name": "Protocol",  # for selecting stimulus protocol
            "tip": "Stimulus protocol to present. The list is read from protocols/ "
            "when the program starts.",
            "type": "list",
            "limits": [str(p) for p in known_protocols],
            "value": str(current_protocol),
        },
        {
            "name": "Data Path",  # for selecting where recordings are saved
            "tip": "Where recordings are saved.",
            "type": "list",
            "limits": data_path_choices,
            "value": data_path_choices[0],
        },
        {
            "name": "Parameters",  # for displaying stimulus parameters
            "tip": "Values from the protocol file. They cannot be edited here; to change "
            "them, save a new protocol file.",
            "type": "group",
            "children": [
                {
                    "name": "wave_duration",
                    "tip": "Duration of each stimulus waveform (s).",
                    "type": "float",
                    "value": stimuli["wave_duration"],
                },  # waveform duration in milli seconds
                {
                    "name": "stimulus_duration",
                    "tip": "Duration of each click or tone pip (s).",
                    "type": "float",
                    "value": 5e-3,
                },  # seconds
                {
                    "name": "stimulus_risefall",
                    "tip": "Rise and fall time of each tone pip (s).",
                    "type": "float",
                    "value": 5e-4,
                },  # seconds
                {
                    "name": "delay",
                    "tip": "Time from the start of the waveform to the first stimulus (s).",
                    "type": "float",
                    "value": 3e-3,
                },  # seconds
                {
                    "name": "nreps",
                    "tip": "Number of repetitions averaged at each level and frequency.",
                    "type": "float",
                    "value": 50,
                },  # number of repetitions
                {
                    "name": "stimulus_period",
                    "tip": "Time between waveform presentations (s).",
                    "type": "float",
                    "value": 1,
                },  # seconds
                {
                    "name": "nstim",
                    "tip": "Number of clicks or pips in each waveform.",
                    "type": "int",
                    "value": 30,
                },  # number of pips
                {
                    "name": "interval",
                    "tip": "Time between the clicks or pips in a waveform (s).",
                    "type": "float",
                    "value": 25e-3,
                },  # seconds
                {
                    "name": "alternate",
                    "tip": "Alternate the polarity of successive clicks.",
                    "type": "bool",
                    "value": True,
                },
                {
                    "name": "default_frequency",
                    "tip": "Tone frequency used when the protocol has no frequency list (Hz).",
                    "type": "float",
                    "value": 4000.0,
                },
                {
                    "name": "default_spl",
                    "tip": "Level used when the protocol has no level list (dB SPL).",
                    "type": "float",
                    "value": 80.0,
                },
                {
                    "name": "freqlist",
                    "tip": "Tone frequencies presented (Hz).",
                    "type": "str",
                    "value": " [2000.0, 4000.0, 8000.0, 12000.0, 16000.0, 20000.0, 24000.0, 32000.0, 48000.0]",
                },
                {
                    "name": "dblist",
                    "tip": "Stimulus levels presented (dB SPL). In the binaural modes these "
                    "are the levels of the left path.",
                    "type": "str",
                    "value": "[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]",
                },
                {
                    "name": "mode",
                    "tip": "How the stimulus is presented: freefield (one speaker, the "
                    "left path), monaural_left, monaural_right, binaural, or "
                    "bic_interleaved (left, right and binaural in one sweep).",
                    "type": "str",
                    "value": "freefield",
                },
                {
                    "name": "dblist_right",
                    "tip": "Stimulus levels of the right path (dB SPL), paired by position "
                    "with dblist. Binaural modes only.",
                    "type": "str",
                    "value": "",
                    "visible": False,
                },
                {
                    "name": "itd",
                    "tip": "Interaural time difference (s): the time of the right path "
                    "stimulus relative to the left in binaural slots. Positive means the "
                    "right path lags. Binaural modes only.",
                    "type": "float",
                    "value": 0.0,
                    "visible": False,
                },
            ],
        },
        {
            "name": "Calibration",  # which calibration each output path uses
            "tip": "The calibration files in use. 'Left' is the first output path (NI ao0, "
            "attenuator 1), the only one in free-field mode. 'Right' is the second output "
            "path (ao1, attenuator 2), used in binaural modes.",
            "type": "group",
            "children": [
                {
                    "name": "Left (free-field) path",
                    "tip": "Calibration of the left path (NI ao0, attenuator 1).",
                    "type": "str",
                    "value": "",
                    "readonly": True,
                },
                {
                    "name": "Right path",
                    "tip": "Calibration of the right path (NI ao1, attenuator 2). Without a "
                    "separate file, the right path uses the left calibration.",
                    "type": "str",
                    "value": "",
                    "readonly": True,
                },
            ],
        },
        {"name": "Subject Data",
            "tip": "Saved with the data.",
            "type": "group",
            "children": [
                {
                    "name": "Subject ID",  # open comment field
                    "tip": "Required before Start Acquisition. It becomes part of the data "
                    'folder name, so it cannot contain \\ / : * ? " < > |',
                    "type": "str",
                    "value": "",
                },
                {"name": "Age", "type": "str", "value": ""},
                {"name": "Sex", "type": "str", "value": ""},
                {"name": "Weight", "type": "float", "value": 0.0},
                {"name": "Strain", "type": "str", "value": ""},
                {"name": "Genotype", "type": "str", "value": "WT"},
                {"name": "Treat Group", "type": "str", "value": ""},
            ],
        },
        {
            "name": "Actions",
            "type": "group",
            "children": [
                {
                    "name": "New Filename",
                    "tip": "Has no effect on the names of saved files.",
                    "type": "action",
                },
                {
                    "name": "Test Acquisition",
                    "tip": "Presents the protocol and displays responses. Saves nothing.",
                    "type": "action",
                },
                {
                    "name": "Start Acquisition",
                    "tip": "Presents the protocol, displays responses, and saves them. "
                    "Needs a Subject ID.",
                    "type": "action",
                },
                {"name": "Pause", "tip": "Holds the presentation.", "type": "action"},
                {
                    "name": "Resume",
                    "tip": "Continues a paused presentation from the same place.",
                    "type": "action",
                },
                {
                    "name": "Stop",
                    "tip": "Ends the presentation. It cannot be resumed.",
                    "type": "action",
                },
                {"name": "Load Data File", "tip": "Opens the data viewer.", "type": "action"},
                {
                    "name": "Read Cal File",
                    "tip": "Chooses the speaker calibration to use (the left, free-field path).",
                    "type": "action",
                },
                {
                    "name": "Read Right Cal File",
                    "tip": "Chooses the calibration of the right path (second speaker), "
                    "used in binaural modes.",
                    "type": "action",
                },
                {"name": "Calibrate", "tip": "Opens the calibration window.", "type": "action"},
            ],
        },
        {
            "name": "Status",
            "tip": "Hardware found, the current mode, and progress through the protocol.",
            "type": "group",
            "children": [
                {
                    "name": "Devices",
                    "tip": "Acquisition and stimulus hardware found at startup.",
                    "type": "str",
                    "value": "None",
                    "readonly": True,
                },
                {
                    "name": "Mode",
                    "tip": "What the program is doing now.",
                    "type": "str",
                    "value": "Ready",
                    "readonly": True,
                },
                {
                    "name": "dBSPL",
                    "tip": "Level of the current stimulus (dB SPL).",
                    "type": "float",
                    "value": 0.0,
                    "readonly": True,
                },
                {
                    "name": "Freq (kHz)",
                    "tip": "Frequency of the current tone pip (kHz).",
                    "type": "float",
                    "value": 0.0,
                    "readonly": True,
                },
                {"name": "Wave #",
                    "tip": "Current waveform / number of waveforms in the protocol.",
                    "type": "str",
                    "value": "0/0",
                    "readonly": True,
                    },

                {"name": "Rep #",
                    "tip": "Current repetition / repetitions per waveform.",
                    "type": "str",
                    "value": "0/0",
                    "readonly": True,
                    },

            ],
        },
        {
            "name": "Quit",
            "tip": "Stops the presentation and closes the program.",
            "type": "action",
        },
    ]

    ptree = ParameterTree()
    ptreedata = Parameter.create(name="params", type="group", children=params)
    for child in ptreedata.child("Parameters").children():  # set by the protocol file
        child.setOpts(readonly=True)
    ptree.setParameters(ptreedata)
    show_tips_on_names(ptreedata)
    ptree.setMaximumWidth(350)

    return ptree, ptreedata


def show_tips_on_names(param):
    """pyqtgraph shows a "tip" only on the value widget or button: show it on the name too,
    so that the groups (which have no widget) also have their tips."""
    for child in param.children():
        tip = child.opts.get("tip")
        if tip:
            for item in child.items:
                item.setToolTip(0, tip)
        show_tips_on_names(child)
