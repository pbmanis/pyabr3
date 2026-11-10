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
            "type": "list",
            "limits": [str(p) for p in known_protocols],
            "value": str(current_protocol),
        },
        {
            "name": "Data Path",  # for selecting where recordings are saved
            "type": "list",
            "limits": data_path_choices,
            "value": data_path_choices[0],
        },
        {
            "name": "Parameters",  # for displaying stimulus parameters
            "type": "group",
            "children": [
                {
                    "name": "wave_duration",
                    "type": "float",
                    "value": stimuli["wave_duration"],
                },  # waveform duration in milli seconds
                {
                    "name": "stimulus_duration",
                    "type": "float",
                    "value": 5e-3,
                },  # seconds
                {
                    "name": "stimulus_risefall",
                    "type": "float",
                    "value": 5e-4,
                },  # seconds
                {
                    "name": "delay",
                    "type": "float",
                    "value": 3e-3,
                },  # seconds
                {
                    "name": "nreps",
                    "type": "float",
                    "value": 50,
                },  # number of repetitions
                {
                    "name": "stimulus_period",
                    "type": "float",
                    "value": 1,
                },  # seconds
                {
                    "name": "nstim",
                    "type": "int",
                    "value": 30,
                },  # number of pips
                {
                    "name": "interval",
                    "type": "float",
                    "value": 25e-3,
                },  # seconds
                {"name": "alternate", "type": "bool", "value": True},
                {
                    "name": "default_frequency",
                    "type": "float",
                    "value": 4000.0,
                },
                {"name": "default_spl", "type": "float", "value": 80.0},
                {
                    "name": "freqlist",
                    "type": "str",
                    "value": " [2000.0, 4000.0, 8000.0, 12000.0, 16000.0, 20000.0, 24000.0, 32000.0, 48000.0]",
                },
                {
                    "name": "dblist",
                    "type": "str",
                    "value": "[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]",
                },
            ],
        },
        {"name": "Subject Data",
            "type": "group",
            "children": [
                {"name": "Subject ID", "type": "str", "value": ""},  # open comment field
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
                {"name": "New Filename", "type": "action"},
                {"name": "Test Acquisition", "type": "action"},
                {"name": "Start Acquisition", "type": "action"},
                {"name": "Pause", "type": "action"},
                {"name": "Resume", "type": "action"},
                {"name": "Stop", "type": "action"},
                {"name": "Load Data File", "type": "action"},
                {"name": "Read Cal File", "type": "action"},
                {"name": "Calibrate", "type": "action"},
            ],
        },
        {
            "name": "Status",
            "type": "group",
            "children": [
                {"name": "Devices", "type": "str", "value": "None", "readonly": True},
                {"name": "Mode", "type": "str", "value":"Ready", "readonly": True},
                {"name": "dBSPL", "type": "float", "value": 0.0, "readonly": True},
                {
                    "name": "Freq (kHz)",
                    "type": "float",
                    "value": 0.0,
                    "readonly": True,
                },
                {"name": "Wave #",
                    "type": "str",
                    "value": "0/0",
                    "readonly": True,
                    },

                {"name": "Rep #",
                    "type": "str",
                    "value": "0/0",
                    "readonly": True,
                    },

            ],
        },
        {
            "name": "Quit",
            "type": "action",
        },
    ]

    ptree = ParameterTree()
    ptreedata = Parameter.create(name="params", type="group", children=params)
    for child in ptreedata.child("Parameters").children():  # set by the protocol file
        child.setOpts(readonly=True)
    ptree.setParameters(ptreedata)
    ptree.setMaximumWidth(350)

    return ptree, ptreedata
