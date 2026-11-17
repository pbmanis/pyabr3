# Figures needed for the manual

Save each one in `doc_tex/figures/` with the name shown, as `.png`, `.jpg` or `.pdf`.
The manual uses a file as soon as it is there, in place of the box on the page. Run
`make` in `doc_tex/` to rebuild. No change to the `.tex` files is needed.

## Needed from the rig

| File name | Chapter | What to capture |
|---|---|---|
| `online_recording` | 4 | Recording tab during a real recording, with a clear ABR in the Average panel (for example, clickIO at 80 to 90 dB SPL). Include the Status group. |
| `tdt_circuit_abr` | 6 | Export of `tdt/abrs_v2.rcx` from RPvdsEx. |
| `tdt_circuit_mic` | 6 | Export of `tdt/mic_record.rcx` from RPvdsEx. |
| `hw_rack_front` | 6 | Photograph of the front of the rack, with each device labeled. |
| `hw_rack_back` | 6 | Photograph of the rear, with trigger, output and input cables identified. |
| `hw_speaker_setup` | 6 | Photograph of the speaker and animal (or a stand-in), showing distance and angle. |
| `hw_electrodes` | 6 | Photograph or drawing of the electrode positions. |
| `hw_calibrator` | 6 | Photograph of the microphone seated in the calibrator. |
| `ni_max_device` | 6 | NI MAX screen showing the device name (Dev1). |

## Needed from data not in the repository

| File name | Chapter | What to capture |
|---|---|---|
| `viewer_matlab` | 5 | Data viewer with a MATLAB ABR4 run open. |

## Already present, and worth replacing

These were made on a Mac with no hardware. They show the real layout of each window. To
replace one, save the new image over it with the same name.

| File name | Chapter | Now shows | Better |
|---|---|---|---|
| `main_window` | 1 | Idle window; Devices lists the sound card | Rig, listing NIDAQ, RP21 and PA5 |
| `cal_speaker` | 2 | Simulated hardware | Real speaker calibration |
| `cal_microphone` | 2 | Simulated hardware | Real microphone calibration |
| `cal_check` | 2 | Simulated hardware | Real 75 dB check |

`cal_verify_dialog`, `cal_chooser`, `calibration_tab`, `viewer_clicks`, `viewer_clicks_sd`,
`viewer_interleaved` and `viewer_thresholds` use stored lab data, or show dialogs that
look the same on the rig.
