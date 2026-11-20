import numpy as np
from pathlib import Path
from typing import Union
import pyqtgraph as pg
import pyqtgraph.configfile as configfile
import src.sound as sound
import src.protocol_reader as protocol_reader
import src.read_calibration as read_calibration

""" Wave generator : part of pyabr3
This class generates a clicks, tone pips, and "interleaved ramps"
for use in ABR experiments. The waveforms are generated at the maximal
voltage, and attenuated using the attenuators on the TDT system.

Genearlly, the waveforms are generated at a high sample rate (250-500 kHz)
to minimize distortion. For this we use a National Instruments NI6371 DAC
card. The waveforms are then attenuated using the PA5 attenuators,
Some attention is paid to trying to maximize the dynamic range, as the
DAC is "only" 16 bits (96 dB).

"""

# the conditions presented, in order, by each of the modes that present them one at a time
SEQUENTIAL_CONDITIONS = {
    "monaural_left": ("L",),
    "monaural_right": ("R",),
    "binaural": ("B",),
    "bic_sequential": ("L", "R", "B"),
}
MAX_ATTENUATION = 120.0  # dB, the attenuator setting for an ear that does not play


class UnreachableLevelError(ValueError):
    """A protocol asks for a level that would need negative attenuation."""


class WaveGenerator:
    def __init__(self, caldata):
        self.protocol = None
        self.frequency = 500000.0
        self.use_attens = True
        self.wave_matrix = {}
        self.get_protocols()
        self.caldata = caldata  # the (left) speaker; the only one in free-field mode
        self.caldata_right = None  # right speaker; None means the right ear uses caldata
        self.multi_attn_flatten_spectra = False  # try to flatten using
            # mutiple attenaution levels and less DAC dynamic range

    def setup(self, protocol: Union[str, Path], frequency: float = 500000.0,
              config:Union[dict, None]=None):
        self._set_protocol(protocol)
        self._set_output_frequency(frequency)
        self.config = config
        self.wave_matrix = {}

    def _set_protocol(self, protocol):
        self.protocol = protocol

    def _set_output_frequency(self, frequency: float):
        self.sfout = frequency

    def caldata_for(self, ear: str = "left") -> dict:
        """Return the calibration of the speaker for one ear ("left" or "right").
        Without a separate right calibration (caldata_right is None), both ears use
        caldata, so free-field operation is unchanged.
        """
        if ear not in ("left", "right"):
            raise ValueError(f"ear must be 'left' or 'right', not {ear!r}")
        if ear == "right" and self.caldata_right is not None:
            return self.caldata_right
        return self.caldata

    def click_reference_dbspl(self, ear: str = "left") -> float:
        """Return the click level at 0 dB attenuation for one ear. The right ear uses
        "click_reference_dbspl_right" from the configuration if it is there, otherwise
        "click_reference_dbspl" (the only value in a free-field configuration).
        """
        self.caldata_for(ear)  # checks the ear
        if ear == "right" and "click_reference_dbspl_right" in self.config:
            return self.config["click_reference_dbspl_right"]
        return self.config["click_reference_dbspl"]

    def compute_attn_db_for_dbspl(self, freq, dBSPL, max_spl:float=0., ear: str = "left"):
        """compute_attn_for_dbspl
        based on the current calibration (of the speaker for the given ear),
        compute the attenuation required for a given frequency to appear at the
        specified dB SPL
        """
        caldata = self.caldata_for(ear)
        ifr = np.argsort(caldata["freqs"])
        freqs = caldata["freqs"][ifr]
        maxdb = caldata["maxdb"][ifr]
        # for i, frx in enumerate(freqs):
        #     print(f"{frx:8.1f}  {maxdb[i]:8.3f}")
        max_dBSPL = np.interp([freq], freqs, maxdb)
        attn_dB = max_dBSPL - dBSPL
        return attn_dB, max_dBSPL
    
    def check_tone_levels(self, frequencies, dbspls, ear: str = "left"):
        """Raise UnreachableLevelError, listing every frequency where a requested level
        is above the calibrated maximum.

        Bug: tone pips above the calibrated maximum were sent to the attenuator with
        negative attenuation, with no warning. pyabr3.acquire now reports this before
        anything is presented, from this check and the one for clicks in make_waveforms.
        """
        problems = []
        for frequency in frequencies:
            too_high = [
                db for db in dbspls if self.compute_attn_db_for_dbspl(frequency, db, ear=ear)[0][0] < 0
            ]
            if len(too_high) > 0:
                maxdb = self.compute_attn_db_for_dbspl(frequency, 0.0, ear=ear)[1][0]
                problems.append(f"{frequency:.0f} Hz: maximum {maxdb:.1f} dB SPL, requested {too_high}")
        if len(problems) > 0:
            raise UnreachableLevelError("Cannot reach these levels:\n" + "\n".join(problems))

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

        Bug: interleaved_plateau stored a wave_matrix entry for each level, so
        presenter_thread.py ran nreps sweeps per level instead of nreps in all, and
        played the first entry, which lacks the final scaling, so levels were too low
        by the attenuator setting. There is now one entry, with the whole waveform.
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
        
        # binaural interaction component stimuli (mode key; free-field when absent)
        mode = protocol_reader.get_mode(self.protocol)
        if mode == "bic_interleaved":
            self._make_bic_interleaved(wavetype, dbspls, frequencies)
            return
        elif mode != "freefield":  # one condition (or each in turn) per set of sweeps
            self._make_bic_sequential(mode, wavetype, dbspls, frequencies)
            return

        # stimuli requiring more attenuation will use the external PA5
        # attenuators, and the DAC signal will be adjusted accordingly
        match wavetype:
            case "click":
                print("doing click")
                too_high = [db for db in dbspls if db > self.config["click_reference_dbspl"]]
                if len(too_high) > 0:
                    raise UnreachableLevelError(
                        f"Cannot reach these click levels: maximum "
                        f"{self.config['click_reference_dbspl']:.1f} dB SPL, requested {too_high}"
                    )
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
                self.check_tone_levels(frequencies, dbspls)
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
                self.check_tone_levels(freqs, dbs)
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
                            pip_start=[s0 + n * dt],
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
                raise ValueError(f"Unrecognized wavetype: {wavetype:s}")

    @staticmethod
    def _as_list(value) -> list:
        """Protocol lists are lists, or strings to evaluate (as for interleaved_plateau)."""
        if isinstance(value, str):
            value = eval(value)
        return [float(v) for v in np.atleast_1d(value)]

    def _bic_levels_and_frequencies(self, wavetype: str, dbspls, frequencies, ears=("left", "right")):
        """The left and right levels (dB SPL, paired by position) and the frequencies of a
        binaural protocol. The right levels are dblist_right, or the left levels if absent.
        Levels that an ear in "ears" cannot reach raise UnreachableLevelError.
        """
        stim = self.protocol["stimuli"]
        dbs_left = self._as_list(dbspls)
        if stim.get("dblist_right") is None:
            dbs_right = list(dbs_left)
        else:
            dbs_right = self._as_list(stim["dblist_right"])
        if len(dbs_left) != len(dbs_right):
            raise ValueError(
                f"dblist ({len(dbs_left)}) and dblist_right ({len(dbs_right)}) must have the same length"
            )
        levels = {"left": dbs_left, "right": dbs_right}
        if wavetype == "click":
            freqs = [0.0]
            for ear in ears:
                too_high = [db for db in levels[ear] if db > self.click_reference_dbspl(ear)]
                if len(too_high) > 0:
                    raise UnreachableLevelError(
                        f"Cannot reach these click levels ({ear} ear): maximum "
                        f"{self.click_reference_dbspl(ear):.1f} dB SPL, requested {too_high}"
                    )
        elif wavetype == "tonepip":
            freqs = self._as_list(frequencies)
            for ear in ears:
                self.check_tone_levels(freqs, levels[ear], ear=ear)
        else:
            raise ValueError(f"Binaural presentation needs click or tonepip stimuli, not {wavetype!r}")
        return dbs_left, dbs_right, freqs

    def _bic_template(self, wavetype: str, frequency: float, duration: float) -> np.ndarray:
        """One click or tone pip at the start of a waveform of the given duration, at full
        scale (its level is set by the attenuator), as the click and tonepip stimuli."""
        stim = self.protocol["stimuli"]
        if wavetype == "click":
            template = sound.ClickTrain(
                rate=self.sfout,
                duration=duration,
                dbspl=self.click_reference_dbspl("left"),  # only the shape is used
                click_duration=float(stim["stimulus_duration"]),
                click_starts=[0.0],
                alternate=False,
            )
            template.generate()
            return read_calibration.SPLCAL_Click_Voltage * template.sound / np.max(template.sound)
        template = sound.TonePip(
            rate=self.sfout,
            duration=duration,
            f0=frequency,
            dbspl=94.0,  # only the shape is used
            pip_duration=float(stim["stimulus_duration"]),
            ramp_duration=stim["stimulus_risefall"],
            pip_start=[0.0],
            alternate=False,  # do alternation separately here
        )
        template.generate()
        return read_calibration.SPLCAL_Tone_Voltage * template.sound / np.max(template.sound)

    def _make_bic_interleaved(self, wavetype: str, dbspls, frequencies):
        """
        Interleaved binaural interaction component (BIC) stimuli, for clicks or tone pips.

        As in interleaved_plateau, all conditions are in one sweep, so drift in the
        recording affects all of them equally. A sweep is nstim groups of slots, one
        slot every "interval" seconds. A group is L, R, Bin (left only, right only,
        both ears). With "alternate", each slot is presented twice, in opposite polarity
        (click sign, or tone pip phase): L+ L- R+ R- B+ B-. The alternation is
        applied here, per slot, as in interleaved_plateau.

        There is one wave_matrix entry per (frequency, level pair), each presented for
        nreps sweeps, as for tonepip. Levels (and the ITD) are the same throughout a
        sweep; the left level is from dblist, the right level from dblist_right (the
        same as the left if absent). The two lists are paired by position. The key is
        (wavetype, left dB SPL, frequency, right dB SPL, itd), so key[0:3] are as for
        click and tonepip.

        The entry holds the left channel ("sound", NI ao0) and the right channel
        ("sound_right", ao1), both at full scale; the level of each ear is set by its
        attenuator: "attenuation" is [left, right]. itd (seconds) is the time of the
        right ear stimulus relative to the left in Bin slots; positive means the right
        ear lags (the left leads). The monaural slots have no delay. The entry also holds
        the slot description: "slot_conditions" ("L", "R", "B"), "slot_polarities"
        (+1, -1) and "slot_starts" (s, the nominal start of the slot, before the ITD).
        """
        stim = self.protocol["stimuli"]
        nstim = int(stim["nstim"])
        interval = float(stim["interval"])
        delay = float(stim["delay"])
        itd = float(stim.get("itd", 0.0))
        alternate = bool(stim["alternate"])
        pip_duration = float(stim["stimulus_duration"])
        if nstim < 1:
            raise ValueError("nstim must be at least 1 for an interleaved BIC protocol")
        if abs(itd) + pip_duration >= interval:
            raise ValueError(
                f"|itd| ({abs(itd):g} s) plus the stimulus duration must be less than the interval ({interval:g} s)"
            )
        dbs_left, dbs_right, freqs = self._bic_levels_and_frequencies(wavetype, dbspls, frequencies)

        # the slots in a group: condition and polarity
        if alternate:
            group = [("L", 1), ("L", -1), ("R", 1), ("R", -1), ("B", 1), ("B", -1)]
        else:
            group = [("L", 1), ("R", 1), ("B", 1)]
        slots = group * nstim
        slot_starts = [delay + k * interval for k in range(len(slots))]
        wave_duration = delay + len(slots) * interval + interval  # as interleaved_plateau
        n_wave = int(round(wave_duration * self.sfout))
        # in B slots the lagging ear is delayed by |itd|
        itd_left = max(0.0, -itd)
        itd_right = max(0.0, itd)

        self.wave_matrix = {}
        for frequency in freqs:
            template_wave = self._bic_template(wavetype, frequency, interval)
            for db_left, db_right in zip(dbs_left, dbs_right):
                left = np.zeros(n_wave)
                right = np.zeros(n_wave)
                for (condition, polarity), start in zip(slots, slot_starts):
                    for channel, ear_in_slot, ear_delay in (
                        (left, "L", itd_left),
                        (right, "R", itd_right),
                    ):
                        if condition == "B":
                            t0 = start + ear_delay
                        elif condition == ear_in_slot:
                            t0 = start
                        else:
                            continue
                        i0 = int(round(t0 * self.sfout))
                        n = min(len(template_wave), n_wave - i0)
                        channel[i0 : i0 + n] += polarity * template_wave[:n]
                if wavetype == "click":
                    attn_left = self.click_reference_dbspl("left") - db_left
                    attn_right = self.click_reference_dbspl("right") - db_right
                else:
                    attn_left = self.compute_attn_db_for_dbspl(frequency, db_left, ear="left")[0][0]
                    attn_right = self.compute_attn_db_for_dbspl(frequency, db_right, ear="right")[0][0]
                self.wave_matrix[(wavetype, db_left, frequency, db_right, itd)] = {
                    "sound": left,
                    "sound_right": right,
                    "rate": self.sfout,
                    "mode": "bic_interleaved",
                    "attenuation": [float(attn_left), float(attn_right)],
                    "itd": itd,
                    "frequency": frequency,
                    "dbspl_left": db_left,
                    "dbspl_right": db_right,
                    "slot_conditions": [c for c, _ in slots],
                    "slot_polarities": [p for _, p in slots],
                    "slot_starts": list(slot_starts),
                    "pip_durations": pip_duration,
                }
                self.wave_time = np.arange(n_wave) / self.sfout
        self.nwaves = len(self.wave_matrix)

    def _make_bic_sequential(self, mode: str, wavetype: str, dbspls, frequencies):
        """
        Binaural stimuli presented one condition at a time: a set of sweeps (nreps) of the
        left only, the right only, and/or the binaural stimulus. The modes are
        monaural_left (L), monaural_right (R), binaural (B) and bic_sequential (L, then R,
        then B, for each frequency and level pair).

        A sweep is as for click and tonepip: nstim clicks or pips, interval seconds apart,
        starting at delay, in a waveform of wave_duration, with alternating polarity if
        "alternate". Levels, frequencies, itd, the key and the entry are as in
        _make_bic_interleaved, and the entry also has "condition" ("L", "R" or "B"). The
        key is (wavetype, left dB SPL, frequency, right dB SPL, itd, condition). The
        attenuator of an ear that does not play is at its maximum (120 dB).
        Only the ears that play need to reach the requested level.
        """
        conditions = SEQUENTIAL_CONDITIONS[mode]
        stim = self.protocol["stimuli"]
        nstim = int(stim["nstim"])
        interval = float(stim["interval"])
        delay = float(stim["delay"])
        itd = float(stim.get("itd", 0.0))
        alternate = bool(stim["alternate"])
        pip_duration = float(stim["stimulus_duration"])
        wave_duration = float(stim["wave_duration"])
        if nstim < 1:
            raise ValueError("nstim must be at least 1")
        starts = [delay + k * interval for k in range(nstim)]
        polarities = [-1 if (alternate and k % 2 == 1) else 1 for k in range(nstim)]
        if starts[-1] + abs(itd) + pip_duration > wave_duration:
            raise ValueError(
                f"The last stimulus ({starts[-1] + abs(itd) + pip_duration:g} s) does not fit in "
                f"wave_duration ({wave_duration:g} s)"
            )
        ears = []
        if any(c in ("L", "B") for c in conditions):
            ears.append("left")
        if any(c in ("R", "B") for c in conditions):
            ears.append("right")
        dbs_left, dbs_right, freqs = self._bic_levels_and_frequencies(
            wavetype, dbspls, frequencies, ears=tuple(ears)
        )
        n_wave = int(round(wave_duration * self.sfout))
        itd_left = max(0.0, -itd)
        itd_right = max(0.0, itd)
        self.wave_matrix = {}
        for frequency in freqs:
            template_wave = self._bic_template(wavetype, frequency, interval)
            for db_left, db_right in zip(dbs_left, dbs_right):
                for condition in conditions:
                    channels = {}
                    for ear, plays, ear_delay in (
                        ("left", condition in ("L", "B"), itd_left),
                        ("right", condition in ("R", "B"), itd_right),
                    ):
                        channel = np.zeros(n_wave)
                        if plays:
                            for start, polarity in zip(starts, polarities):
                                t0 = start + (ear_delay if condition == "B" else 0.0)
                                i0 = int(round(t0 * self.sfout))
                                n = min(len(template_wave), n_wave - i0)
                                channel[i0 : i0 + n] += polarity * template_wave[:n]
                        channels[ear] = channel
                    attenuation = []
                    for ear, plays, db in (
                        ("left", condition in ("L", "B"), db_left),
                        ("right", condition in ("R", "B"), db_right),
                    ):
                        if not plays:
                            attenuation.append(MAX_ATTENUATION)
                        elif wavetype == "click":
                            attenuation.append(float(self.click_reference_dbspl(ear) - db))
                        else:
                            attenuation.append(
                                float(self.compute_attn_db_for_dbspl(frequency, db, ear=ear)[0][0])
                            )
                    self.wave_matrix[(wavetype, db_left, frequency, db_right, itd, condition)] = {
                        "sound": channels["left"],
                        "sound_right": channels["right"],
                        "rate": self.sfout,
                        "mode": mode,
                        "condition": condition,
                        "attenuation": attenuation,
                        "itd": itd,
                        "frequency": frequency,
                        "dbspl_left": db_left,
                        "dbspl_right": db_right,
                        "slot_conditions": [condition] * nstim,
                        "slot_polarities": list(polarities),
                        "slot_starts": list(starts),
                        "pip_durations": pip_duration,
                    }
        self.wave_time = np.arange(n_wave) / self.sfout
        self.nwaves = len(self.wave_matrix)

    def dbscale(self, v_in: np.ndarray, v_scale_factor: float = 1.0, dbattn: float = 0):
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
            wplot = win.addPlot()  # type: ignore
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


    configfilename = "config/abrs_test.cfg"
    # get the latest calibration file:
    cfg = configfile.readConfigFile(configfilename)
    fn = Path(cfg['calfile'])
   
    WG = WaveGenerator(caldata=read_calibration.get_calibration_data(fn) )

    WG.setup(protocol=WG.stim, frequency=500000)  # type: ignore
    WG.make_waveforms(WG.protocol["protocol"]["stimulustype"]) # type: ignore
    WG.plot_stimulus_wave() # type: ignore
