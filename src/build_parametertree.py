# Define parameters that control aquisition and buttons...
from pyqtgraph.parametertree import Parameter, ParameterTree

def build_parametertree(known_protocols, current_protocol, stimuli:dict):
    params = [
        {
            "name": "Protocol",  # for selecting stimulus protocol
            "type": "list",
            "limits": [str(p) for p in known_protocols],
            "value": str(current_protocol),
        },
        {
            "name": "Parameters",  # for displaying stimulus parameters
            "type": "group",
            "children": [
                {
                    "name": "wave_duration",
                    "type": "float",
                    "value": stimuli["wave_duration"],
                    "limits": [0.1, 10.0],
                },  # waveform duration in milli seconds
                {
                    "name": "stimulus_duration",
                    "type": "float",
                    "value": 5e-3,
                    "limits": [1e-3, 20e-3],
                },  # seconds
                {
                    "name": "stimulus_risefall",
                    "type": "float",
                    "value": 5e-4,
                    "limits": [1e-4, 10e-4],
                },  # seconds
                {
                    "name": "delay",
                    "type": "float",
                    "value": 3e-3,
                    "limits": [1e-3, 1.0],
                },  # seconds
                {
                    "name": "nreps",
                    "type": "float",
                    "value": 50,
                    "limits": [1, 2000],
                },  # number of repetitions
                {
                    "name": "stimulus_period",
                    "type": "float",
                    "value": 1,
                    "limits": [0.005, 10.0],
                },  # seconds
                {
                    "name": "nstim",
                    "type": "int",
                    "value": 30,
                    "limits": [1, 1000],
                },  # number of pips
                {
                    "name": "interval",
                    "type": "float",
                    "value": 25e-3,
                    "limits": [1e-3, 1e-1],
                },  # seconds
                {"name": "alternate", "type": "bool", "value": True},
                {
                    "name": "default_frequency",
                    "type": "float",
                    "value": 4000.0,
                    "limits": [100.0, 100000.0],
                },
                {"name": "default_spl", "type": "float", "value": 80.0, "limits": [0.0, 100.0]},
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
                {"name": "Save Visible", "type": "action"},
                {"name": "Load Data File", "type": "action"},
                {"name": "Read Cal File", "type": "action"},
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
    ptree.setParameters(ptreedata)
    ptree.setMaximumWidth(350)

    return ptree, ptreedata
