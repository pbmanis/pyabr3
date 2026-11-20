# ABR acquisition: empty / variable-length sweeps — analysis and plan

## What the code does (findings)

**1. The handshake call may exit at once.**
- `pystim3.load_and_arm` called `databuffer.acquire(trigger, "running", False)`.
- The vendored signature is `acquire(trigger, handshake_tag, end_condition, ...)`. So `False` is the end condition, and acquisition ends when `running == 0` (`tdtpy/tdt/dsp_buffer.py`, `acquire`/`_acquire`).
- `_acquire` fires the trigger and then polls straight away. If `running` still reads 0 because the circuit hasn't latched it yet, the loop exits immediately with almost no data. That would give zero-length reads.
- `poll_interval` is 0.1 s, so a late `running` edge is read at a 100 ms granularity. That could also explain trials of different lengths.

**2. The result of `acquire()` was thrown away and then re-read.**
- `acquire()` returns a (trial, channel, sample) array. It was stored in `self.ch1`, then overwritten by `databuffer.read(None)`.
- `read(None)` returns only the samples that arrived *since the last read*, in whole blocks (`abstract_ring_buffer.py`).
- Its length therefore depends on how much `acquire` had already consumed. That would vary from trial to trial and sometimes be empty.
- This is a hypothesis. The `sweep:` diagnostics should confirm or reject it.

**3. The NI-versus-A/D ordering isn't the problem.**
- `acquire` blocked before the `is_task_done` loop, so the DAC finishing early doesn't cut the A/D short.

**4. The wrap ambiguity is a risk on the RPvds side.**
- If `sampled_wave_n` equals the buffer size, the index `sampled_wave_i` wraps to 0 when the buffer fills. The software can't tell "finished" from "not started" using the index alone.
- The vendored reader falls back on a software wrap counter when there is no `sampled_wave_c` cycle tag. That is fragile if polling is slow.

**5. The repeat logic could not work as structured.**
- `Presenter.next_stimulus` incremented `repetition_counter` unconditionally, then emitted `signal_data_ready`. `update_ABR` runs in the GUI thread through a queued signal, so its `return False` went nowhere.
- `TrialCounter` is the display counter. `repetition_counter` is the real trial count, and it advanced on bad reads. Saved file names also use `repetition_counter`.

## Plan (status)

1. **Diagnostics (no behaviour change).** DONE. `_acquire_sweep` prints one `sweep:` line per trial: `running` before and after the trigger, poll count, time to completion, pending samples against the target, and `write_index`.
2. **Replace `acquire()` with an explicit read.** DONE. `pystim3._acquire_sweep`:
   - It resets the read index, fires the trigger, and polls `pending()` every 5 ms (`SWEEP_POLL_S`) with a timeout.
   - It then reads exactly `acquisition_points` samples.
   - It returns `None` on a timeout, a buffer overrun or a wrong shape.
   - The second `read(None)` is gone.
3. **Report validity from the data path.** DONE.
   - `retrieveRP21_inputs` returns `None` unless the sweep is exactly `acquisition_points` long.
   - `Presenter.retrieve_data` returns a bool and emits `signal_data_ready` only for a good sweep.
4. **Fix the counter / retry in `Presenter.next_stimulus`.** DONE (untested on hardware).
   - The presentation is now in `_present_once()`, which returns True only for a good sweep.
   - `next_stimulus` repeats it up to `MAX_RETRIES` (3) more times, so up to 4 attempts. `repetition_counter` is incremented once per trial, not per attempt.
   - After the attempts are exhausted the trial is skipped with a status message. The counter has advanced, so a skipped trial leaves a gap in the numbering and the count falls short by one.
   - No flag is passed to the GUI. `update_ABR` only ever sees good data.
5. **Optional circuit change in RPvds (needs hardware).**
   - Add a latched `sweep_done` tag. It is set when the index reaches `sampled_wave_n` and cleared on a software reset trigger.
   - Optionally add a `sampled_wave_c` cycle counter.
   - Software then waits on a flag that cannot be missed, instead of on a wrapping index. Do this only if steps 1–4 don't fully fix it.
6. **Tests.** TODO. Add a unit test for the retry and counter logic using a fake `sound_class` that returns empty data on some calls. This needs no hardware.

## RPvds notes for step 5

Latched flag: an iCompare plus a JK flip-flop.
- https://www.tdt.com/docs/rpvdsex/reference/integer-math/#icompare
- **Compare input:** feed the iCompare the same buffer-index signal that drives `sampled_wave_i`. Test `>= sampled_wave_n - 1`, since the index stops at n−1 or wraps to 0.
- **J input:** the iCompare output.
- **K input (reset):** use a dedicated software trigger rather than relying on the circuit start. The flip-flop is expected to be fine at startup, but it is not certain that Halt/Run clears flip-flop state in every case. If a dedicated trigger is used, `_acquire_sweep` should fire it before the acquisition trigger.
- Alternatively, gate J with the existing `running` signal so a stale compare can't set it.
- **Read-back:** expose the flip-flop output as a tag, then add it to `_acquire_sweep` as the preferred done-signal, keeping the `pending()` check as a fallback.

Cycle tag versus `sampled_wave_i`:
- `sampled_wave_i` is the position inside the buffer (0 to n−1) and wraps.
- A cycle tag counts how many times it has wrapped. Together the pair (cycle, index) gives the total samples written, which distinguishes "just started" from "wrapped after one full pass".
- If the done flip-flop is used, the cycle tag is not needed.

Things to check in the circuit when back at the hardware:
- Whether the data buffer is a SerStore or a SerSource. SerStore (A/D to buffer, read by the PC) is the expected one for acquisition. SerSource is the playback side, from the PC to the circuit.
- The vendored reader computes `write_index = index_tag * compression / channels`, so the buffer's compression and channel settings affect how the index is interpreted. The iCompare threshold must use the raw index units.
- Whether `sampled_wave_n` sets the buffer size or just the stop count.

## When back at the hardware
Run a block of trials and keep the console output. Look at the `sweep:` lines, including any failures:
- Is `running` already 0 after the trigger (`running a->b`)?
- Does the time to completion match the record duration?
- How often do `pending` or the shape come up short?
- How often is a trial retried or skipped (`Presenter: ... bad sweep`)?

## Notes added after step 4

**Step 4 details** (`presenter_thread.py`)
- The presentation and retrieval moved into a new `_present_once()`, which returns True only for a good sweep. `next_stimulus` calls it in a loop.
- A bad sweep is repeated up to `MAX_RETRIES` (3) more times, so up to 4 attempts per trial.
- `repetition_counter` now advances once per trial, not per attempt. Bad sweeps are no longer emitted, so `update_ABR` and `TrialCounter` only see good data.
- If all attempts fail, the trial is skipped with the status "Trial skipped: no data from RP2.1". The counter has advanced, so that trial leaves a gap in the file numbering and the count falls short by one. Skipping was chosen over retrying forever, so a dead RP2.1 can't hang the thread.
- Open option: stop the acquisition after repeated failures instead of skipping. This is a small change if wanted.
- The bug docstring is on `_present_once`. The unreachable early-return in the click branch was changed to `return True`, to match the new return type.
- Step 4 has not been run on hardware.

**Cycle tag versus `sampled_wave_i`**
- `sampled_wave_i` is the position inside the buffer, from 0 to n−1, and it wraps.
- A cycle tag counts those wraps. Together they give the total samples written. That distinguishes "just started" from "wrapped after exactly one full pass", which matters when `sampled_wave_n` equals the buffer size.
- If the latched flip-flop `sweep_done` flag is built, the cycle tag is not needed.

**SerStore or SerSource**
- A SerStore is the buffer the circuit writes into (A/D to buffer) and the PC reads. That is what acquisition should use.
- A SerSource is the PC-to-circuit playback side.
- The `.rcx` can't be read here, so check in RPvds which one holds `sampled_wave`.
- The vendored reader computes `write_index = index_tag * compression / channels`. If the SerStore uses compression or multiple channels, the iCompare threshold has to use the raw index units.

## Update: the latched data_ready flag (step 5, circuit change) is used by the software
`abrs_v3.rcx` now has `compare_count` (the value the iCompare compares the buffer index with) and `data_ready` (the JK flip-flop output, set when the comparison is true and reset by the edge-detected zBusB start trigger). The buffer is a SerStore. `PyStim._acquire_sweep` (pystim3.py) uses them when the circuit has both tags:
- Before the trigger it sets `compare_count` to the last raw index, `n * channels / compression - 1`. This works with an EQ or a GE test, so the test type does not have to be changed. (A GE test is the more tolerant of the two; with EQ the index must pass through exactly that value.)
- It fires the start trigger (`RP21_TriggerSource`, "B"), which also resets `data_ready`, and reads `data_ready` once straight away. If it is still set, the trigger did not clear it, so this sweep is read by the buffer index instead (a message is printed).
- The sweep is complete when `data_ready` is set (polled every 5 ms, with the same timeout as before). After a 2 ms settle (`DATA_READY_SETTLE_S`) for the last samples to be stored, the n samples are read from the start of the buffer (`buf._read(0, n)`), without the wrap counting of `pending()`.
- A circuit without the tags is read by the buffer index, as before.
- The `sweep:` line now also shows the value of `data_ready` after the trigger and whether the sweep was completed by the flag or the index.

Things to check on the rig:
- That `data_ready` reads 0 just after the trigger. If the circuit is stopped and restarted for each trial, the flip-flop should be clear at the start anyway.
- That the index the iCompare sees is the SerStore index in raw units. If it is not, `compare_count` will not match. The sweep would then time out, and the `sweep:` line would show it.
- If the last sample of a sweep looks wrong, increase `DATA_READY_SETTLE_S`.

## Update: first test on the rig, and no fallback
The first run with `data_ready` printed, for every trial: `data_ready is still set after the trigger`, then `by index, polls=106-205, t=1.23-1.26 s, pending=29696/29696, write_index=0.0`. So the sweeps were read correctly by the buffer index (29696 samples is 1.216 s at 24.4 kHz, and the index wrapped to 0 at the end), but the flag was already high just after the start trigger: the zBusB edge detect is not clearing it, or it is being set again at once.

Changes made:
- A circuit that lacks `sampled_wave`, `sampled_wave_n`, `data_ready` or `compare_count` is now an error (`pystim3.CircuitError`), raised when the circuit is loaded (`setup_RP21`, ABR mode only; the calibration circuit is not checked). There is no buffer-index fallback any more.
- A flag that is still set just after the start trigger is also a `CircuitError`, not a bad sweep: repeating the trial cannot fix it. The message gives the value of `data_ready` before the trigger, `compare_count` and the buffer index. Reading them:
  - set before and after: the flip-flop is not being reset (the K input, or the edge detect of zBusB).
  - clear before and set after: the trigger is setting it, so the comparison is true at the start (for example if the compare is against a value that the index meets at 0, or if J is not gated by the sweep).
- In the program the error stops the acquisition and shows a message box ("Hardware"), instead of quitting the program or counting skipped trials: `Presenter.run` catches it and sends `signal_hardware_error`; `PyABR.hardware_error` stops and shows it. At startup (and for each acquisition) a circuit without the tags shows the same message.
- `Stimulus_Parameters.RP21_ResetTrigger` (default `None`) can name an extra trigger, for example a software trigger wired to K, which is fired before the start trigger and must clear `data_ready` within 50 ms (`CircuitError` if not). It is only needed if the start trigger cannot be made to reset the flag.

Until the circuit resets `data_ready` on the start trigger (or `RP21_ResetTrigger` is set), acquisition will stop with the message on the first trial.

## Update: the flag was set before the trigger, on the first trial
The error on the rig was: before the trigger True, `compare_count` 29695, buffer index 0, and still set after the trigger. Two changes in `pystim3.py`:
- `compare_count` is now also set in `load_and_arm` before the circuit starts (`_compare_count`, the last raw index). Before, it was set only after the start, so its default (0) could equal the idle buffer index (0) and set `data_ready` at power-up (an EQ test; a GE test would too). If the flag is now clear before the first trigger, that was the cause of the first-trial error.
- After the start trigger, a flag that is still set is waited for to clear, for up to 0.1 s or half of the sweep if that is shorter (`DATA_READY_CLEAR_S`). If it clears (a slow reset), the sweep goes on and "data_ready cleared N ms after the start trigger" is printed. If not, `CircuitError` says "stayed set", with the value before the trigger, `compare_count` and the buffer index.
What it shows on the rig:
- clear before the first trigger, and cleared by the trigger: working.
- clear before the first trigger, but set before the next trial's trigger (the flag left by the last sweep, because stopping and starting the circuit does not clear it) and not cleared by the trigger: the reset (K, or the zBusB edge detect) does not work, and `RP21_ResetTrigger`, or a different reset in the circuit, is needed.

## Update: sanity line, announced repeats, and no length adjustment
Rig result: trial 1 had `data_ready` clear before and after the trigger; later trials had it set before (left by the last sweep) and cleared 10-12 ms after the start trigger, which the software waits for. The sweeps were 29696 samples.
- `pystim3.PyStim._check_sweep` prints, for every sweep, `sanity: std=..., min=..., max=..., trailing zeros=N, same as previous sweep=False`. A sweep is rejected (the trial is repeated) if it ends in more than `MAX_TRAILING_ZEROS` (16) exact zeros (the buffer was not filled to the end, or the input is dead), or is identical to the previous sweep (the buffer was not overwritten). Every rejection prints `****** BAD SWEEP (the trial will be repeated): <reason>`.
- `Presenter.next_stimulus` announces each repeat: `bad sweep (attempt 1 of 4); REPEATING THE TRIAL (N repeat(s) so far in this run)`, and `TRIAL SKIPPED` after `MAX_RETRIES` repeats. `Presenter.n_repeats` counts the repeats in a run. A repeat does not count as a trial.
- `PyABR.update_ABR` no longer pads a short sweep with its mean or cuts a long one. A sweep of the wrong length (or empty) stops the acquisition with a message (`PyABR.sweep_length_error`), because averaging it would hide a fault. The acquisition only passes on sweeps of exactly `acquisition_points`, so this should never appear.
- Test with the microphone on the analog input: the response of the microphone is the same on every trial, so in the average it should stay at the same amplitude and latency while the noise falls as 1/sqrt(trials). A sweep that is late or early, or stale, would smear or shrink it.

## Update: the repeat of the start of the sweep after 1090.55 ms
On the rig, with the microphone signal, the timing was good (the waveform about 200 us after the pulse), the sanity lines were as expected, and the data jumped at 1090.55 ms, after which the start of the waveform seemed to repeat to the end of the sweep.
- 1090.55 ms is sample 26624 (= 26 x 1024) at 24414.0625 Hz, and the sweep is 29696 samples (1.2163 s). So the buffer in the circuit holds only 26624 samples (1.0905 s) and the rest of the sweep does not fit. The index still counts to the end of the sweep, so `data_ready` is set, and the read past the end of the buffer repeats its start.
- Check this in RPvds: the size of the SerStore (and the memory use of the circuit; the RP2.1 has a limited memory).
- `PyStim._check_buffer_size` (called by `load_and_arm` before the circuit starts) now compares `buffer.size_max` (the capacity of the buffer in the compiled circuit) with `acquisition_points`, and raises `CircuitError` with both sizes in samples and seconds. It also checks that the size tag reads back as what was set. The acquisition stops with the message.
- To record this protocol, either make the SerStore at least 29696 samples (if the memory allows; 32768 would be a multiple of 1024), or set `wave_duration` in the protocol so that the sweep is at most 1.0905 s (the 40 stimuli end at about 1 s, so about 1.05 s would still hold the responses).

## Update: the buffer size is set at design time; the record-length check
Confirmed on the rig: the SerStore Size is fixed when the circuit is designed (the circuit now has 256000 words, 10.4858 s at 24414.0625 Hz), and the `sampled_wave_n` tag does not enlarge it. The program therefore relies on the size compiled into the circuit.
- `PyStim.buffer_capacity()` reads that size from the circuit (`size_max` of the buffer, from `GetTagSize`, in samples per channel).
- `PyStim.check_recording_fits(duration)` computes the record for a waveform (`recording_points`: duration x sample rate, rounded up, then up to a multiple of 1024) and raises `CircuitError` if it is more than the capacity. `PyABR.acquire` calls it with the longest waveform of the protocol (`longest_waveform_duration`) before anything is presented or a folder is made, and shows the message. `load_and_arm` checks again when it arms the circuit (`_check_buffer_size`).
- The message gives the calculation, both sizes in samples and seconds, that the size is set at design time, the longest `wave_duration` that fits (rounded down), and the Size that the circuit would need.
- A duration that is exactly a whole number of samples (the longest that fits) is no longer rounded up a block by floating point error (`waveform_samples`).
- Documented in the manual: `doc_tex/chapters/03_acquisition.tex`, "The record must fit in the buffer of the circuit", with the calculation, tables of examples (record lengths; longest waveform for 26624 and 256000 samples at three sample rates) and how to enlarge the buffer in RPvdsEx. The numbers in the tables are checked in `tests/test_recording_length.py`.
