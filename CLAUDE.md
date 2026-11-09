# pyabr3   — Claude Code context


## Project overview
Python package (v0.4.1+, requires Python 3.13) for Auditory Brainstem Reponse
data acquisition. This is not a general-use library — it is
actively developed for a single lab's workflow.



## Package layout
```
src/
  tdt/   # Talking to Tucker-Davis hardware (A/D side of acquisition)
  tests/ # some tests.
config/  # configuration files
calfiles/ # calibration files 
nidaq/ # interface to the National Instruments DAQ (DAC) board used for stimulus presentation
protocols/ 


## Coding conventions
- Never execute git commands.
- Never delete a file or directory.
- Do not touch the files in the config, calfiles, nidaq or protocols directories.
- Always ask before making any change to a source file, and make changes incrementally (ask for every change, even if they are related).
- When fixing a bug, write a short docstring as to what the bug was, and what was done to fix it (including pointing to related files that had to be changed to accomadate the fix). Be succicint.
- `CP("color", message)` is the project's coloured print utility.

## Related files
 - Configurations are in the config/ directory
## Testing / running
- Some tests exist, but more "unit tests" are needed.
- `uv` is used for dependency management (`uv.lock`, `pyproject.toml`).
- Python 3.13 is required (pinned in `pyproject.toml`).

## Testing convention — REQUIRED after analysis code changes
After editing any file in the following modules, run the relevant tests
before reporting a task complete.

  Rules:
    - NEVER pass `--audit` to pytest; audit mode is for human use only
      at the terminal.
    - If any test fails, investigate and fix before declaring done.

- 
