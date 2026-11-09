"""
Dialog for choosing a speaker calibration file.

Lists the current files (for example, frequency_MF1.cal) and those in the
calibration_history folder, with the most recent first.

The time shown is the "Created on" time in the MAT-file header, which is written when
the calibration is saved. Times from the file system are not used, because they change
when files are copied or moved.
"""

import datetime
import re
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtWidgets

from src.calibration.engine import HISTORY_DIR

FILE_PATTERN = "frequency_*.cal"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
HEADER_TIME_FORMAT = "%a %b %d %H:%M:%S %Y"
MAT_HEADER_BYTES = 116
UNKNOWN_TIME = "unknown"


def created_time(path: Path) -> datetime.datetime | None:
    """Return the time in the header of a .cal file, or None if there is no readable time."""
    with open(path, "rb") as fh:
        header = fh.read(MAT_HEADER_BYTES).split(b"\x00")[0].decode("ascii", "replace")
    match = re.search(r"Created on:\s*(.*\S)", header)
    if match is None:
        return None
    try:
        return datetime.datetime.strptime(match.group(1), HEADER_TIME_FORMAT)
    except ValueError:
        return None


def list_calibration_files(cal_dir: Path) -> list:
    """Return (path, time created) for the calibration files, most recent first.

    Files without a readable time are listed last.
    """
    cal_dir = Path(cal_dir)
    paths = list(cal_dir.glob(FILE_PATTERN)) + list((cal_dir / HISTORY_DIR).glob(FILE_PATTERN))
    found = [(p, created_time(p)) for p in paths]
    return sorted(
        found, key=lambda item: item[1] or datetime.datetime.min, reverse=True
    )


class CalibrationFileChooser(QtWidgets.QDialog):
    def __init__(self, cal_dir: Path, current: Path | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Calibration File")
        self.resize(620, 420)
        self.cal_dir = Path(cal_dir)

        self.table = QtWidgets.QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["File", "Created"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.doubleClicked.connect(self.accept)
        self.folder_label = QtWidgets.QLabel()
        folder_button = QtWidgets.QPushButton("Choose Folder...")
        folder_button.clicked.connect(self.choose_folder)
        self.show_folder(self.cal_dir, current)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.folder_label)
        layout.addWidget(self.table)
        layout.addWidget(folder_button)
        layout.addWidget(buttons)

    def show_folder(self, cal_dir: Path, current: Path | None = None):
        self.cal_dir = Path(cal_dir)
        self.files = list_calibration_files(self.cal_dir)
        self.folder_label.setText(f"Calibration files in {self.cal_dir!s}")
        self.table.setRowCount(len(self.files))
        for row, (path, created) in enumerate(self.files):
            name = path.relative_to(self.cal_dir).as_posix()
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            text = created.strftime(TIME_FORMAT) if created else UNKNOWN_TIME
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(text))
        self.table.clearSelection()
        self.select_file(current)

    def choose_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Calibration Folder", str(self.cal_dir)
        )
        if folder:
            self.show_folder(Path(folder))

    def select_file(self, target: Path | None):
        for row, (path, _) in enumerate(self.files):
            if target is not None and path == Path(target):
                self.table.selectRow(row)
                return
        if self.files:
            self.table.selectRow(0)

    def selected_path(self) -> Path | None:
        rows = self.table.selectionModel().selectedRows()
        return self.files[rows[0].row()][0] if rows else None


def choose_calibration_file(cal_dir: Path, current: Path | None = None, parent=None):
    """Show the chooser. Returns the chosen file, or None if there is no selection."""
    dialog = CalibrationFileChooser(cal_dir, current, parent)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    return dialog.selected_path()
