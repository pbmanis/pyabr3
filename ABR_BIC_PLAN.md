# Binaural interaction component (BIC) acquisition and analysis — plan

## Goal
Record BIC data the same way ABRs are recorded now, with:
1. Control of a second attenuator.
2. Calibration of two speakers, each assigned to a stimulus path.
3. A stimulus mode (free field, binaural, monaural left or right) added to the protocols without breaking the current free-field structure.
4. Data tagged by condition: monaural left, monaural right, binaural, and for binaural the interaural time difference (ITD).
5. Two ways of collecting binaural data:
   - (a) A full L response, a full R response and a full binaural response, one after another.
   - (b) "Interleaved BIC": L, R and Bin interleaved within a sweep so that slow drift in recording conditions affects all conditions equally (like the interleaved plateau method).
6. A small analyzer, separate from the acquisition, that reads the interleaved-BIC files and displays the three conditions. The BIC formula (reference to come) is added later.

## Decisions from the user
- **Polarity alternation** flips within each pair of stimuli. With alternation the order in a group is `L+ L- R+ R- B+ B-`, where `+`/`-` is click polarity or tone-pip phase. Without alternation the order is `L R B`.
- **`nstim`** is the number of such groups ("triplets"). Sweep stimulus time is 6 × 40 ms × `nstim` with alternation and 3 × 40 ms × `nstim` without (plus the leading delay, as in the plateau protocol).
- **Slot spacing** is 40 ms (nominal), as a protocol parameter.
- **Hardware:** the second PA5 is installed. NI `ao1` is wired to the second attenuator and second speaker driver. Assumed fixed assignment: left = `ao0` + PA5 device 1, right = `ao1` + PA5 device 2.
- **Stimulus type:** clicks or tone pips, chosen by the existing `stimulustype` together with the mode.

## Requirement: no change to current operation
Nothing here may change how the program currently runs, including the existing tests. Safeguards:
- **Opt-in only.** Every new key (`mode`, `dblist_right`, `itd`, `calfile_left`, `calfile_right`) is optional. When absent, the code takes the existing path: one speaker, `ao0`, PA5 #1, the single `calfile`.
- **Hardware is conditional.** The second PA5 and NI `ao1` are used only when the mode asks for them. A system with one PA5, or a protocol without a mode, behaves as it does now. A missing second PA5 is reported only if a binaural or BIC mode is selected.
- **Same output as before.** In free-field mode the wave matrix, attenuation values and NI write (a single-channel array) must be identical to today's. A regression test will compare them for the existing protocols.
- **Saved files.** Only additive keys, written only in the new modes. Free-field files keep exactly their current contents, so the existing readers and analysis code are untouched.
- **Existing protocols and GUI.** Existing protocol files load unchanged. The new GUI controls are hidden or inactive for free-field mode.
- **Each step is checked** against the existing tests (never with `--audit`), and Claude confirms that free-field output is unchanged before moving to the next step.

## Left and right are output paths
"Left" and "right" in the code and the config name the two output paths, not anatomical sides:
- **Left** = NI `ao0` + PA5 device 1 + the calibration in `calfile` (or `calfile_left`). This is the only path in free-field mode, so the free-field speaker is the "left" one even though it is not necessarily on the left.
- **Right** = NI `ao1` + PA5 device 2 + `calfile_right`. It is used only in binaural modes. Without `calfile_right` the right path uses the left calibration, which is correct only for matched speakers.
- The GUI must label these as paths (for example "left (free-field)"), not as ears, wherever the user chooses one.

## Findings: constraints from the current code
- `pystim3.acquire_with_devices` accepts `waver` but ignores it. `load_and_arm` adds only NI `ao0` to the task, and `Presenter` plays the same wave in both channels.
- Only one PA5 is connected (`device_id=1`). `setAttens(left, right)` sets only that one, and the second-PA5 lines are commented out.
- `caldata` is one dict, loaded from the single `calfile` in the config. `WaveGenerator.compute_attn_db_for_dbspl` uses it for everything.
- Protocols have no mode key. Saved files carry `protocol`, `wave_number` and `repetition`, but no per-condition tag.
- CLAUDE.md puts `protocols/`, `config/`, `calfiles/` and `nidaq/` off limits for Claude. The new protocol file and the new config keys must be made by the user. Claude supplies the exact text.

## Design
1. **Mode as an optional protocol key.**
   - Under `stimuli:`, add `mode` with values `freefield`, `monaural_left`, `monaural_right`, `binaural` or `bic_interleaved`.
   - A missing key means `freefield`, so existing protocols and saved files behave exactly as now.
   - Binaural parameters: `dblist` for the left level, `dblist_right` for the right level, and `itd` in seconds.
2. **Per-ear calibration.**
   - Optional config keys `calfile_left` and `calfile_right`. If absent, the one `calfile` serves both ears.
   - `caldata` becomes a per-channel pair internally, and `WaveGenerator` takes an `ear` argument.
3. **Hardware, in `pystim3.py` only.**
   - Connect PA5 #2 with `device_id=2`, and make `setAttens` drive both.
   - Add NI `ao1` to the same task, and write a 2×N array.
   - Both DAC channels then share one sample clock and one start trigger, so ITD is exact to one sample (2 µs at 500 kHz).
4. **Tagging.** Each saved sweep gets additive keys: `ear_mode`, the per-slot condition list, `itd`, per-ear levels, and the polarity of each slot. Old readers ignore them.
5. **Interleaved BIC stimulus** (new `wavetype`, in `wave_generator.make_waveforms`).
   - Slots 40 ms apart, in the order above, with the group repeated `nstim` times.
   - L-only slots put zero on the right DAC channel, R-only slots put zero on the left, and Bin slots have both with the ITD applied.
   - One attenuator setting per ear for the whole sweep.
   - Tone-pip phase alternation uses the existing `alternate` mechanism in `sound.piptone`.
6. **Approach (a)** reuses the existing per-key `wave_matrix` loop. `Presenter` needs the condition added to the wave key. This is smaller than (5) and doesn't depend on it.
7. **GUI:** mode selector, second level list, ITD and a calibration choice per ear. These go in `build_parametertree.py` and `protocol_reader.py`.
8. **BIC viewer.**
   - New module `src/bic_viewer.py` with a pure function (no Qt) that reads the files of one run directory and splits each sweep into slots.
   - It averages across repetitions and, when polarity alternates, across the two polarities, and returns the three condition averages.
   - A thin dock in the main window plots L, R and Bin in three colours with labels.
   - The BIC formula goes in later as a fourth step.
9. **Tests with no hardware:** wave construction (slot timing, zero channels, ITD in samples) and slot splitting with polarity averaging.

## Progress
- Step 1 (mode keys and defaults): done. `protocol_reader.get_mode`.
- Step 2 (per-ear calibration): done. `WaveGenerator.caldata_for`, `calfile_right`.
- Step 3 (hardware): done, untested on the rig. PA5 #2 (`PyStim.PA5_2`, requested with `"PA5_2"` in `required_hardware`), `setAttens`, `play_sound(..., two_channel=True)` writing `ao0:1` on one task, and a calibration "Output path" choice. A non-free-field mode is refused in `acquire` until step 4.
- Config keys for the user to add to `config/abrs.cfg`: `calfile_right`, `click_reference_dbspl_right`.
- Step 4 (the interleaved BIC wave): done. `WaveGenerator._make_bic_interleaved`, used by `make_waveforms` when `mode: bic_interleaved`. `stimulustype` is `click` or `tonepip`. One wave_matrix entry per (frequency, level pair), key `(wavetype, dB left, frequency, dB right, itd)`, with `sound` (left, ao0), `sound_right` (right, ao1), `attenuation` `[left, right]`, and the slot description (`slot_conditions`, `slot_polarities`, `slot_starts`). Positive `itd` means the right ear lags. The modes `monaural_left`, `monaural_right` and `binaural` raise `NotImplementedError` (approach (a) is later).
- Step 5 (presentation and tagging): done, untested on the rig. In `acquire`, non-free-field modes request `PA5_2` (`protocol_reader.required_hardware_for_mode`); a missing second PA5 gives a message box. `Presenter` plays BIC entries with `play_sound(sound, sound_right, attns=attenuation, two_channel=True)`. Saved sweeps in non-free-field modes carry `ear_mode` and a `conditions` dictionary (`slot_conditions`, `slot_polarities`, `slot_starts`, `itd`, `frequency`, `dbspl_left`, `dbspl_right`, `attenuation`), from `PyABR.get_condition_tags`; `sound_right` is emptied in the saved stimulus copy. Free-field files are unchanged. The guard that refused non-free-field protocols is removed; `monaural_left`, `monaural_right` and `binaural` still raise `NotImplementedError` from `make_waveforms`.
- Step 7 (the viewer): done. The code is in the existing viewer package rather than `src/bic_viewer.py` as first planned. `src/viewer/bic_files.py` has the analysis (no Qt): each trial is filtered and resampled as in `abr_files`, cut into one response per slot (each starting 1 ms before the nominal slot start, so 0 ms is the slot start), and averaged by condition, which includes both polarities when they alternate, then across trials, in uV. `src/viewer/bic_viewer.py` has `BICPanel` (stimulus selector, high/low pass, three coloured and labelled traces), docked as "BIC" in the main window and usable alone (`python -m src.viewer.bic_viewer [trial file]`). The BIC itself (the comparison of the three conditions) is not computed yet; it waits for the reference formula.
- Step 6 (GUI): done. The parameter tree's Parameters group is read-only by design (protocols are used as saved; an earlier fix removed editing because edits were discarded), so the new fields are displayed, not edited: `mode`, and, only in the binaural modes, `dblist_right` and `itd`. A new Calibration group shows the file used by the left (free-field) path and the right path ("same as left" without a separate file), and a "Read Right Cal File" action chooses the right one. The saved files of the non-free-field modes also carry `calibration_right`. Mode, levels and ITD are set in the protocol file, not in the window.
- BIC computation: done in the viewer. `BICResponses.bic` is the binaural response minus the sum of the left and right responses (abr_binaural - abr_sum; J Neurosci 38(40):8563, Fig 1 C-F). The panel plots the three conditions and L+R above, and the BIC below, with an option to overlay the BIC of every stimulus in the recording, to follow it as the ITD varies (one level per recording, as in the paper). No amplitude or latency measure of the BIC yet.

- Approach (a), conditions presented one at a time: done, untested on the rig. The modes `monaural_left` (L), `monaural_right` (R), `binaural` (B) and the new `bic_sequential` (L, then R, then B for each frequency and level pair) are made by `WaveGenerator._make_bic_sequential`. A sweep is like a click or tonepip sweep (`nstim` stimuli `interval` apart, in a waveform of `wave_duration`, alternating polarity if `alternate`), played on the ears of the condition; the ear that does not play has its attenuator at 120 dB. Monaural modes ignore `itd`; only the ears that play need to reach the level. Entries are keyed `(wavetype, dB left, frequency, dB right, itd, condition)`, are tagged like the interleaved ones, and are played by `Presenter` on two channels (any entry with a `sound_right`). The viewer combines the waves of one stimulus (frequency, levels, ITD) into one result, so a `bic_sequential` run shows L, R, B and the BIC; a single-condition run shows that condition alone. The level checks and the pip/click template are shared with the interleaved code.

## Order of work
Each step is a separate change, and Claude asks before each one:
1. Mode keys and defaults, with no behaviour change.
2. Per-ear calibration.
3. Hardware: PA5 #2 and NI two-channel output.
4. The interleaved BIC wave.
5. Tagging in the saved files.
6. GUI controls.
7. The viewer and its tests.

Approach (a) goes after step 4.

## Things to check
- `src/calibration/hardware.py`, the calibration program, needs a way to say which speaker and channel is being calibrated, and to save separate cal files. Check this before step 2.
- The record window: the acquisition length is rounded up to a multiple of 1024 samples (`_compute_in_sampling_info`). The viewer must slice slots using the nominal slot times, not the rounded length.
- A binaural slot with an ITD extends one channel past the nominal slot. Make sure the slot spacing leaves room for the largest ITD.
- The new protocol file is made by the user. Claude supplies text such as:

```
stimuli:
    mode: 'bic_interleaved'
    nstim: 10            # groups of L, R, Bin
    interval: 40e-3      # slot spacing
    alternate: True      # L+ L- R+ R- B+ B-
    itd: 0.0             # seconds
    dblist: "[60.0]"
    dblist_right: "[60.0]"
```
