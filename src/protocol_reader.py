from pyqtgraph import configfile
import convert_nested_ordered_dict
from pathlib import Path

# How the stimulus is presented. "freefield" (one speaker) is the original behavior.
PRESENTATION_MODES = (
    "freefield",
    "monaural_left",
    "monaural_right",
    "binaural",
    "bic_sequential",
    "bic_interleaved",
)
DEFAULT_MODE = "freefield"


def get_mode(protocol: dict) -> str:
    """Return the presentation mode of a protocol.

    The "mode" key (under "stimuli") is optional; a protocol without it is free-field.
    The protocol dictionary is not modified, so free-field data files are unchanged.
    Raises ValueError for an unknown mode.
    """
    mode = protocol["stimuli"].get("mode", DEFAULT_MODE)
    if mode not in PRESENTATION_MODES:
        raise ValueError(
            f"Unknown stimulus mode {mode!r}; it must be one of {', '.join(PRESENTATION_MODES)}"
        )
    return mode


def required_hardware_for_mode(required_hardware: list, mode: str) -> list:
    """Return the hardware to request for a presentation mode: the configured list, plus
    the second attenuator ("PA5_2", the right path) for the binaural modes when the
    configuration has an attenuator at all. Free-field gets the list unchanged.
    """
    hardware = list(required_hardware)
    if mode != "freefield" and "PA5" in hardware and "PA5_2" not in hardware:
        hardware.append("PA5_2")
    return hardware


class ProtocolReader():
    def __init__(self, ptreedata):
        self.ptreedata = ptreedata
        self._protocol = None

    def read_protocol(self, protocolname, update: bool = False):
        """
        Read the current protocol
        """
        try:
            protocol = configfile.readConfigFile(Path("protocols", protocolname))
        except ValueError:
            raise ValueError("ProtocolReader:read_protocol: Protocol file incorrectly read")
        # print("protocol: ", protocol)
        # if isinstance(protocol['stimuli']["dblist"], str):
        #     protocol['stimuli']["dblist"] = list(eval(str))
        # if "freqlist" in list(protocol["stimuli"].keys()):
        #     if isinstance(protocol['stimuli']["freqlist"], str):
        #         protocol['stimuli']["freqlist"] = list(eval(str))
        protocol = convert_nested_ordered_dict.convert_nested_ordered_dict(protocol)
        get_mode(protocol)  # reject an unknown mode when the protocol is read
        # paste the raw string into the text box for reference
        self._protocol = protocol  # the parameters in a dictionary...
        if update:
            self.update_protocol()
    
    def get_current_protocol(self):
        if self._protocol is not None:
            return self._protocol
        else:
            raise ValueError("ProtocolReader:get_protocol: No protocol has been read yet")
    
    def update_protocol(self, ptreedata):
        self.ptreedata = ptreedata
        children = self.ptreedata.children()
        data = None
        for child in children:
            if child.name() == "Parameters":
                data = child
        if data is None:
            return
        for child in data.children():
            if child.name() == "wave_duration":
                child.setValue(float(self._protocol["stimuli"]["wave_duration"]))
            if child.name() == "stimulus_duration":
                child.setValue(float(self._protocol["stimuli"]["stimulus_duration"]))
            if child.name() == "stimulus_risefall":
                child.setValue(float(self._protocol["stimuli"]["stimulus_risefall"]))
            if child.name() == "delay":
                child.setValue(float(self._protocol["stimuli"]["delay"]))
            if child.name() == "nreps":
                child.setValue(int(self._protocol["stimuli"]["nreps"]))
            if child.name() == "stimulus_period":
                child.setValue(float(self._protocol["stimuli"]["stimulus_period"]))
            if child.name() == "nstim":
                child.setValue(int(self._protocol["stimuli"]["nstim"]))
            if child.name() == "interval":
                child.setValue(self._protocol["stimuli"]["interval"])
            if child.name() == "alternate":
                child.setValue(self._protocol["stimuli"]["alternate"])
            if child.name() == "default_frequency":
                child.setValue(self._protocol["stimuli"]["default_frequency"])
            if child.name() == "default_spl":
                child.setValue(self._protocol["stimuli"]["default_spl"])
            if child.name() == "freqlist":
                child.setValue(self._protocol["stimuli"]["freqlist"])
            if child.name() == "dblist":
                child.setValue(self._protocol["stimuli"]["dblist"])
        # the binaural fields: shown only in the binaural modes (free-field protocols have none)
        stimuli = self._protocol["stimuli"]
        mode = get_mode(self._protocol)
        binaural = mode != DEFAULT_MODE
        for child in data.children():
            if child.name() == "mode":
                child.setValue(mode)
            if child.name() == "dblist_right":
                child.setValue(str(stimuli.get("dblist_right", "")))
                child.show(binaural)
            if child.name() == "itd":
                child.setValue(float(stimuli.get("itd", 0.0)))
                child.show(binaural)
        # self.make_and_plot()
        # self.stimulus_waveform.enableAutoRange()
        # self.Dock_Recording.raiseDock()


    def get_protocol(self):
        """get_protocol Read the current protocol information and put it into
        the protocol dictionary
        """
        children = self.ptreedata.children()
        data = None
        for child in children:
            if child.name() == "Parameters":
                data = child
        if data is None:
            return
        for child in data.children():
            if child.name() == "wave_duration":
                self._protocol["stimuli"]["wave_duration"] = child.value()
            if child.name() == "stimulus_duration":
                self._protocol["stimuli"]["stimulus_duration"] = child.value()
            if child.name() == "stimulus_risefall":
                self._protocol["stimuli"]["stimulus_risefall"] = child.value()
            if child.name() == "delay":
                self._protocol["stimuli"]["delay"] = child.value()
            if child.name() == "nreps":
                self._protocol["stimuli"]["nreps"] = child.value()
            if child.name() == "stimulus_period":
                self._protocol["stimuli"]["stimulus_period"] = child.value()
            if child.name() == "nstim":
                self._protocol["stimuli"]["nstim"] = child.value()
            if child.name() == "interval":
                self._protocol["stimuli"]["interval"] = child.value()
            if child.name() == "alternate":
                self._protocol["stimuli"]["alternate"] = child.value()
            if child.name() == "default_frequency":
                self._protocol["stimuli"]["default_frequency"] = child.value()
            if child.name() == "default_spl":
                self._protocol["stimuli"]["default_spl"] = child.value()
            if child.name() == "freqlist":
                self._protocol["stimuli"]["freqlist"] = child.value()
            if child.name() == "dblist":
                self._protocol["stimuli"]["dblist"] = child.value()
