# Sphinx configuration for the pyabr3 API documentation.
#
# Build from the top directory of the repository:
#     make -C doc html
# The pages are written to doc/build/html/index.html

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Several modules read config/ and protocols/ relative to the working directory,
# and some of them make Qt or matplotlib objects when they are imported.
os.chdir(ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

# src/ is also needed because src/protocol_reader.py uses "import convert_nested_ordered_dict"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from src import __version__  # noqa: E402

project = "pyabr3"
author = "Paul B. Manis"
copyright = "2024-2026, Paul B. Manis"
version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# The acquisition hardware libraries are only present on the rig
autodoc_mock_imports = ["tdt", "tdtpy", "nidaq", "nidaqmx"]
autodoc_default_options = {
    "members": True,
    "undoc-members": True,  # about half of the functions do not have docstrings yet
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"

napoleon_google_docstring = False
napoleon_numpy_docstring = True

exclude_patterns = ["build"]
html_theme = "sphinx_rtd_theme"
html_title = f"pyabr3 {release} API"
