import numpy as np
from pathlib import Path
from typing import Union
import pyqtgraph as pg
import pyqtgraph.configfile as configfile
import src.sound as sound
import src.protocol_reader as protocol_reader
import src.read_calibration as read_calibration


class WaveGenerator:
    def __init__(self, caldata):
        self.protocol = None
        self.frequency = 500000.0
        self.use_attens = True
        self.wave_matrix = {}
        self.get_protocols()
        self.caldata = caldata
        self.multi_attn_flatten_spectra = False  # try to flatten using
            # mutiple attenaution levels and less DAC dynamic range

    def setup(self, protocol: Union[str, Path] = None, frequency: float = 500000.0,
              config:dict=None):
        self._set_protocol(protocol)
        self._set_output_frequency(frequency)
        self.config = config
        self.wave_matrix = {}

    def _set_protocol(self, protocol):
        self.protocol = protocol

    def _set_output_frequency(self, frequency: float):
        self.sfout = frequency

    def compute_attn_db_for_dbspl(self, freq, dBSPL, max_spl:float=0.):
        """compute_attn_for_dbspl
        based on the current calibration, compute the attenuation
        required for a given frequency to appear at the
        specified dB SPL
        """
        ifr = np.argsort(self.caldata["freqs"])
        freqs = self.caldata["freqs"][ifr]
        maxdb = self.caldata["maxdb"][ifr]
        # for i, frx in enumerate(freqs):
        #     print(f"{frx:8.1f}  {maxdb[i]:8.3f}")
        max_dBSPL = np.interp([freq], freqs, maxdb)
        attn_dB = max_dBSPL - dBSPL
        return attn_dB, max_dBSPL
    
    def adjust_dbspl(self, dbspl_nominal):
        """ adjust_dbspl
            calculate the dbspl target for the signal to be put in the waveform buffer
            and pair with an attenuator setting to acheive the nominal SPL
            (levels are adjusted later)
        """
        if dbspl_nominal <= 30:
            attenuator = 70.
            dbspl = dbspl_nominal + 70.
        elif dbspl_nominal <= 70:
            attenuator = 30.
            dbspl = dbspl_nominal + 30.
        elif dbspl_nominal <= 100.:
            attenuator = 0.
            dbspl = dbspl_nominal + 0.
        else:
            attenuator = 0.
            dbspl = dbspl_nominal + 0.
        return dbspl, attenuator

    def make_waveforms(self, wavetype: str, dbspls=None, frequencies=None):
        """
        Generate all the waveforms we will need for this protocol.
        Waveforms, held in self.wave_matrix,
        are in a N x (wave) shape, where N is the number
        of different stimuli. Waveforms are played out in the order
        they appear in this array.
        """
        self.wave_matrix = {}
        if dbspls is None:  # get the list?
            dbspls = self.protocol["stimuli"]["dblist"]
        if dbspls is None:  # still None ? use the default
            dbspls = [self.protocol["stimuli"]["default_spl"]]
        if frequencies is None:
            frequencies = self.protocol["stimuli"]["freqlist"]
        if frequencies is None:
            frequencies = [self.protocol["stimuli"]["default_frequency"]]
        
        # stimuli requiring more attenuation will use the external PA5
        # attenuators, and the DAC signal will be adjusted accordingly
        match wavetype:
            case "click":
                print("doing click")
                # clicks are always generated at the maximal voltage, and
                # attenuated with the attenuator
                # every collection epoch is for one SPL level
                freqs = [0] * len(dbspls)  # set to all zeros
                attenuator = 0.0 # additional attenuation in wave key
                starts = np.cumsum(
                    np.ones(self.protocol["stimuli"]["nstim"])
                    * self.protocol["stimuli"]["interval"]
                )
                wave = None

                starts += self.protocol["stimuli"]["delay"]
                for i, db in enumerate(dbspls):
                    if wave is None:  # only compute once, for the reference level.
                        wave = sound.ClickTrain(
                            rate=self.sfout,
                            duration=self.protocol["stimuli"]["wave_duration"],
                            dbspl=self.config[
                                "click_reference_dbspl"
                            ],  # fixed for clicks, using attenuator...
                            click_duration=self.protocol["stimuli"]["stimulus_duration"],
                            click_starts=starts,
                            alternate=self.protocol["stimuli"]["alternate"],
                        )
                        wave.generate()
                        click_waveform = read_calibration.SPLCAL_Click_Voltage*wave.sound/np.max(wave.sound)
                    attenuator = self.config["click_reference_dbspl"]-db  # compute attenuation
                    self.wave_matrix[("click", db, freqs[i])] = {
                        "sound": click_waveform,
                        "rate": self.sfout,
                        "attenuation": [attenuator],
                    }
                    self.wave_time = wave.time
                self.nwaves = len(dbspls)

            case "tonepip":
                # tonepips are always generated at the maximal voltage, and
                # attenuated with the attenuator
                # every collection epoch is for one frequency and SPL combination
                starts = np.cumsum(
                    np.ones(self.protocol["stimuli"]["nstim"])
                    * self.protocol["stimuli"]["interval"]
                )
                starts += self.protocol["stimuli"]["delay"]
                # print("tonepip ", frequency, dbspl)
                dbref = 94.0
                for j, frequency in enumerate(frequencies):
                    wave = None
                    for i, dbspl in enumerate(dbspls):
                        if wave is None:  # compute once per frequency
                            wave = sound.TonePip(
                                rate=self.sfout,
                                duration=self.protocol["stimuli"]["wave_duration"],
                                f0=frequency,
                                dbspl=dbref,
                                pip_duration=self.protocol["stimuli"]["stimulus_duration"],
                                ramp_duration=self.protocol["stimuli"]["stimulus_risefall"],
                                pip_start=starts,
                                alternate=self.protocol["stimuli"]["alternate"],
                            )
                            # print("tonepip generate: ", db, fr)
                            wave.generate()
                        # compute the attenuator setting for this frequency
                        # and db spl
                        tone_waveform = read_calibration.SPLCAL_Tone_Voltage * wave.sound/np.max(wave.sound)
                        attenuator = self.compute_attn_db_for_dbspl(frequency, dbspl)
                        print("tonepip freq, atten: ", frequency, attenuator)
                        self.wave_matrix[("tonepip", dbspl, frequency)] = {
                            "sound": tone_waveform,
                            "rate": self.sfout,
                            "attenuation": [attenuator[0]],
                        }
                        self.wave_time = wave.time
                self.nwaves = len(dbspls) * len(frequencies)

            case "interleaved_plateau":
                # Stimulus levels are interleaved with changes in frequency
                # because the DAC has a limited dynamic range (16 bits; ~96 dB)
                # we only use the "top" end of the DAC (e.g., 30 dB range, with
                # the maximal voltage being "0") and do multiple banks with 
                # additional external attenuation via the PA5 attenuators.

                if dbspls is None:
                    dbspls = self.protocol["stimuli"]["default_spl"]
                if frequencies is None:
                    frequencies = self.protocol["stimuli"]["default_frequency"]
                dt = self.protocol["stimuli"]["stimulus_period"]
                s0 = self.protocol["stimuli"]["delay"]
                dbs = eval(self.protocol["stimuli"]["dblist"])
                freqs = eval(self.protocol["stimuli"]["freqlist"])
                wave_duration = (
                    s0 + len(dbs) * len(freqs) * dt + dt
                )  # duration of the waveform
                self.dblist = []
                self.freqlist = []
                self.attenuatorlist = []
                self.pip_starts = []
                self.pip_durations = []
                n = 0
                dbref = 94.0
                attenuator = 120.
                max_spl = np.max(dbs)
                for j, dbspl_nominal in enumerate(dbs):
                    for i, frequency in enumerate(freqs):
                        dbattn, maxdBSPL = self.compute_attn_db_for_dbspl(frequency, dbspl_nominal, max_spl)     
                        atten_f = dbattn[0]  # single frequency
                        if atten_f < 0:
                            raise ValueError(f"Cannot reach {dbspl_nominal} at {frequency} Hz (requires attenuation < 0 db: {atten_f:6.1f})")
                        elif atten_f < attenuator:
                            attenuator = atten_f   

                        maxdBSPL_f = maxdBSPL[0]  # single level
                        print(f"    Adjusted atten at fr: {frequency:7.1f}, attn: {atten_f:6.1f}", end=" ")
                        # if dbattn < 0:
                        #     dbattn = maxdBSPL  # no signal if we cannot reach the highest level
                        wave_n = sound.TonePip(
                            rate=self.sfout,
                            duration=wave_duration,
                            f0=frequency,
                            dbspl=dbref,  # sets to 1V output (1Pa = 94 dbSPL)
                            pip_duration=self.protocol["stimuli"]["stimulus_duration"],
                            ramp_duration=self.protocol["stimuli"]["stimulus_risefall"],
                            pip_start=[s0 + (n * dt)],
                            alternate=False,  # do alternation separately here
                        )
                        self.dblist.append(dbspl_nominal)
                        self.freqlist.append(frequency)
                        self.attenuatorlist.append(atten_f)
                        self.pip_starts.append([s0 + n * dt])
                        self.pip_durations.append(
                            self.protocol["stimuli"]["stimulus_duration"]
                        )
                        wave_n.generate()
                        # scale to max of 10V (e.g., calibration level)
                        tone_waveform = read_calibration.SPLCAL_Tone_Voltage * wave_n.sound/np.max(wave_n.sound)
                        # attenuate 
                        v_factor = 10**(-atten_f/20.)
                        bits = int(0.1*v_factor*2**15)
                        print(f"dbspl_nom: {dbspl_nominal:5.1f} max: {maxdBSPL_f:5.1f} v_factor: {v_factor:8.5f} {bits:5d}")
                        tone_waveform = tone_waveform *v_factor
                        self.wave_time = wave_n.time
                        if i == 0 and j == 0:
                            self.wave_out = tone_waveform
                            self.wave_time = wave_n.time
                        else:
                            self.wave_out += tone_waveform
                        # print(f"     max wave out: {np.max(tone_waveform):8.4f} V, (started with: {read_calibration.SPLCAL_Tone_Voltage*np.max(wave_n.sound):8.4f})")
                        # print("   max w: ", np.max(self.wave_out))
                        n += 1
                    self.wave_matrix[("interleaved_plateau", attenuator)] = {
                        "sound": self.wave_out,
                        "rate": self.sfout,
                        "dbspls": self.dblist,
                        "frequencies": self.freqlist,
                        "attenuation": self.attenuatorlist,
                        "pip_starts": self.pip_starts,
                        "pip_durations": self.pip_durations,
                    }
                vmax = np.max(self.wave_matrix['interleaved_plateau', attenuator]["sound"])
                full_scale = read_calibration.SPLCAL_Tone_Voltage/vmax
                db_diff = 20*np.log10(full_scale/vmax)
                print("db diff: ", db_diff, vmax, full_scale)
                wm = self.wave_matrix[("interleaved_plateau", attenuator)]
                wm["sound"] = wm["sound"]*full_scale
                wm['attenuation'] = wm['attenuation'] + db_diff
                # attenuator = 0
                self.wave_matrix[("interleaved_plateau", attenuator)] = wm
                print("attenuator: ", attenuator)
                for i in range(len(wm["frequencies"])):
                    print(f"{wm['frequencies'][i]:7.1f} {wm['dbspls'][i]:5.1f} {wm['attenuation'][i]:5.1f}, max: {np.max(wm['sound']):6.2f}") 
                self.nwaves = 1

            # case "interleaved_ramp":
            #     if dbspl is None:
            #         dbspl = self.protocol["stimuli"]["default_spl"]
            #     if frequency is None:
            #         frequency = self.protocol["stimuli"]["default_frequency"]
            #     dt = self.protocol["stimuli"]["stimulus_period"]
            #     s0 = self.protocol["stimuli"]["delay"]
            #     dbs = eval(self.protocol["stimuli"]["dblist"])
            #     freqs = eval(self.protocol["stimuli"]["freqlist"])
            #     self.dblist = []
            #     self.freqlist = []
            #     self.pip_starts = []
            #     self.pip_durations = []
            #     wave_duration = (
            #         s0 + len(dbs) * len(freqs) * dt + dt
            #     )  # duration of the waveform
            #     self.nwaves = self.protocol["stimuli"]["nreps"]
            #     n = 0
            #     for i, frequency in enumerate(freqs):
            #         for j, dbspl in enumerate(dbs):

            #             wave_n = sound.TonePip(
            #                 rate=self.sfout,
            #                 duration=wave_duration,
            #                 f0=frequency,
            #                 dbspl=dbspl,
            #                 pip_duration=self.protocol["stimuli"]["stimulus_duration"],
            #                 ramp_duration=self.protocol["stimuli"]["stimulus_risefall"],
            #                 pip_start=[s0 + n * dt],
            #                 alternate=False,  # do alternation separately here. self.protocol["stimuli"]["alternate"],
            #             )
            #             self.dblist.append(dbspl)
            #             self.freqlist.append(frequency)
            #             self.pip_starts.append([s0 + n * dt])
            #             self.pip_durations.append(
            #                 self.protocol["stimuli"]["stimulus_duration"]
            #             )
            #             if i == 0 and j == 0:
            #                 wave_n.generate()
            #                 self.wave_out = wave_n.sound
            #                 self.wave_time = wave_n.time
            #             else:
            #                 self.wave_out += wave_n.generate()
            #             n += 1

            #     self.wave_matrix["interleaved_ramp", 0] = {
            #         "sound": self.wave_out,
            #         "rate": self.sfout,
            #         "dbspls": self.dblist,
            #         "frequencies": self.freqlist,
            #         "pip_starts": self.pip_starts,
            #         "pip_durations": self.pip_durations,
            #     }
            #     self.nwaves = 1  # self.protocol["stimuli"]["nreps"]

            # case "interleaved_random":
            #     if dbspl is None:
            #         dbspl = self.protocol["stimuli"]["default_spl"]
            #     if frequency is None:
            #         frequency = self.protocol["stimuli"]["default_frequency"]
            #     dt = self.protocol["stimuli"]["stimulus_period"]
            #     s0 = self.protocol["stimuli"]["delay"]
            #     dbs = eval(self.protocol["stimuli"]["dblist"])
            #     freqs = eval(self.protocol["stimuli"]["freqlist"])
            #     wave_duration = (
            #         s0 + len(dbs) * len(freqs) * dt + dt
            #     )  # duration of the waveform
            #     print("Wave duration: ", wave_duration)
            #     n = 0
            #     dbfr = np.tile(dbs, len(freqs))
            #     frdb = np.tile(freqs, len(dbs))
            #     indices = np.arange(len(dbfr))
            #     np.random.shuffle(indices)  # in place.
            #     self.dblist = dbfr[indices]
            #     self.freqlist = frdb[
            #         indices
            #     ]  # save the order so we can match the responses
            #     for n, isn in enumerate(indices):
            #         wave_n = sound.TonePip(
            #             rate=self.sfout,
            #             duration=wave_duration,
            #             f0=frdb[isn],
            #             dbspl=dbfr[isn],
            #             pip_duration=self.protocol["stimuli"]["stimulus_duration"],
            #             ramp_duration=self.protocol["stimuli"]["stimulus_risefall"],
            #             pip_start=[s0 + n * dt],
            #             alternate=False,  # do alternation separately here. self.protocol["stimuli"]["alternate"],
            #         )
            #         if n == 0:
            #             wave_n.generate()
            #             self.wave_out = wave_n.sound
            #             self.wave_time = wave_n.time
            #         else:
            #             self.wave_out += wave_n.generate()

            #         self.wave_matrix["interleaved_random", isn] = {
            #             "sound": self.wave_out,
            #             "rate": self.sfout,
            #             "dbspls": self.dblist,
            #             "frequencies": self.freqlist,
            #             "indices": isn,
            #         }
            #     self.nwaves = 1
                # self.protocol["stimuli"]["nreps"]
            case _:
                raise ValueError(f"Unrecongnized wavetype: {wavetype:s}")

    def dbscale(self, v_in: np.array, v_scale_factor: float = 1.0, dbattn: float = 0):
        """ dbscale: Convert a voltage array from the "standard" 94 dB (1V data)
        to a value referenced to the calibration, with the appropriate attenuation.

        v_in is the voltage waveform, scaled to 1.0 V. 
        v_scale_factor refers to the voltage used during the calibrations (typically 10V
        see read_calibration and the matlab calibration files)

        dbattn is the attenuation in dB. Note that if dbattn is negative, then the
        stimulus cannot reach the requested level, so we set the waveform to ZEROs.

        """
        vscale = 10.0 ** (-dbattn / 20.0)  # re 1.0V p-p
        if dbattn < 0:
            return np.zeros_like(v_in)
        vout = v_in * v_scale_factor * v_in * vscale
        # print("dbattn: ", dbattn, "vscale: ", vscale)
        return vout

    def plot_stimulus_wave(self, n: int = 0, plot_object=None):
        """
        Plot the most recent waveform
        """
        if plot_object is None:
            app = pg.mkQApp("Stimulus waveform (wave_generator)")
            win = pg.GraphicsLayoutWidget(show=True, title="ABR Data Plot")
            win.resize(640, 480)
            win.setWindowTitle(f"Protocol: {str(Path(self.current_protocol).parent)}")
            wplot = win.addPlot()
        else:
            wplot = plot_object
        first_sound = self.wave_matrix[list(self.wave_matrix.keys())[n]]
        t = np.arange(
            0,
            len(first_sound["sound"]) / first_sound["rate"],
            1.0 / first_sound["rate"],
        )
        wplot.clear()
        wplot.plot(
            t,
            first_sound["sound"],
            pen=pg.mkPen("y"),
        )
        wplot.setXRange(0, np.max(t))
        if plot_object is None:
            pg.exec()
        # self.stimulus_waveform.autoRange()

    def get_protocols(self):
        self.known_protocols = [p.name for p in Path("protocols").glob("*.cfg")]
        self.current_protocol = self.known_protocols[3]
        self.PR = protocol_reader.ProtocolReader(ptreedata=None)
        self.PR.read_protocol(protocolname=self.current_protocol, update=False)
        self.stim = self.PR.get_current_protocol()



if __name__ == "__main__":

    WG = WaveGenerator()

    WG.setup(protocol=WG.stim, frequency=500000)
    WG.make_waveforms(WG.protocol["protocol"]["stimulustype"])
    WG.plot_stimulus_wave()
