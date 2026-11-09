from pyqtgraph import configfile
from src import convert_nested_ordered_dict
from pathlib import Path

class ProtocolReader():
    def __init__(self, ptreedata):
        self.ptreedata = ptreedata
        self._protocol = None

    def read_protocol(self, protocolname, update: bool = False):
        """
        Read the current protocol
        """
        protocol = configfile.readConfigFile(Path("protocols", protocolname))
        # print("protocol: ", protocol)
        # if isinstance(protocol['stimuli']["dblist"], str):
        #     protocol['stimuli']["dblist"] = list(eval(str))
        # if "freqlist" in list(protocol["stimuli"].keys()):
        #     if isinstance(protocol['stimuli']["freqlist"], str):
        #         protocol['stimuli']["freqlist"] = list(eval(str))
        protocol = convert_nested_ordered_dict.convert_nested_ordered_dict(protocol)
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
