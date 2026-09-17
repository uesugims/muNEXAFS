"""Registration control window."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QRadioButton,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..io.stxm import ScanStack
from ..processing.registration import RegistrationResult, register_translation_stack


class _RegistrationWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, scan: ScanStack, data_stack, reference: int, upsample: int, mode: str) -> None:
        super().__init__()
        self.scan = scan
        self.data_stack = data_stack
        self.reference = reference
        self.upsample = upsample
        self.mode = mode

    @Slot()
    def run(self) -> None:
        try:
            if self.data_stack is None:
                raise ValueError("Input images are not loaded")
            result = register_translation_stack(
                self.data_stack,
                reference_index=self.reference,
                upsample_factor=self.upsample,
                reference_mode=self.mode,
            )
            self.finished.emit(result)
        except Exception as exc:  # report errors in the GUI thread
            self.failed.emit(str(exc))


class RegistrationWindow(QMainWindow):
    """Run and inspect subpixel translation registration."""

    resultReady = Signal(object)

    def __init__(self, scan: ScanStack, data_stack=None, source_label: str = "Transmission") -> None:
        super().__init__()
        self.scan = scan
        self.data_stack = scan.transmission if data_stack is None else data_stack
        self.source_label = source_label
        self._thread: QThread | None = None
        self._worker: _RegistrationWorker | None = None
        self.result: RegistrationResult | None = None
        self.setWindowTitle("muNEXAFS — Registration")
        self.resize(760, 860)

        central = QWidget(self)
        layout = QVBoxLayout(central)
        form = QFormLayout()
        self.reference_spin = QSpinBox()
        self.reference_spin.setRange(0, max(0, len(scan.frame_paths) - 1))
        self.fixed_radio = QRadioButton("Fixed reference frame")
        self.previous_radio = QRadioButton("Previous registered frame")
        self.fixed_radio.setChecked(True)
        self.fixed_radio.toggled.connect(self.reference_spin.setEnabled)
        form.addRow("Reference mode", self.fixed_radio)
        form.addRow("", self.previous_radio)
        self.upsample_spin = QSpinBox()
        self.upsample_spin.setRange(1, 100)
        self.upsample_spin.setValue(10)
        form.addRow("Reference frame", self.reference_spin)
        form.addRow("Subpixel upsample factor", self.upsample_spin)
        layout.addLayout(form)
        self.run_button = QPushButton("Run registration")
        self.run_button.clicked.connect(self.run_registration)
        layout.addWidget(self.run_button)
        self.status_label = QLabel(f"Ready — {source_label} / phase correlation / FFT")
        layout.addWidget(self.status_label)

        # Preview of the registration result.  After Run, scroll through the
        # registered frames (or toggle "Show original" to compare) to check the
        # drift correction before applying it.
        self.preview = pg.ImageView(view=pg.PlotItem())
        self.preview.ui.roiBtn.hide()
        self.preview.ui.menuBtn.hide()
        self.preview.setMinimumHeight(320)
        layout.addWidget(self.preview, stretch=1)
        preview_row = QHBoxLayout()
        self.show_original_check = QCheckBox("Show original (compare)")
        self.show_original_check.setToolTip("Show the input frame instead of the registered one, to compare before/after")
        self.show_original_check.toggled.connect(self._update_preview)
        preview_row.addWidget(self.show_original_check)
        self.preview_slider = QSlider(Qt.Orientation.Horizontal)
        self.preview_slider.setRange(0, max(0, len(scan.frame_paths) - 1))
        self.preview_slider.setEnabled(False)
        self.preview_slider.valueChanged.connect(self._update_preview)
        preview_row.addWidget(self.preview_slider, stretch=1)
        self.preview_label = QLabel("—")
        preview_row.addWidget(self.preview_label)
        layout.addLayout(preview_row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Frame", "Shift X (px)", "Shift Y (px)"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(180)
        layout.addWidget(self.table)
        self.apply_button = QPushButton("Apply registered stack")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply_result)
        layout.addWidget(self.apply_button)
        self.setCentralWidget(central)

    def run_registration(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self.run_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.status_label.setText("Registering…")
        self._thread = QThread(self)
        self._worker = _RegistrationWorker(
            self.scan, self.data_stack, self.reference_spin.value(), self.upsample_spin.value(),
            "previous" if self.previous_radio.isChecked() else "fixed"
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._registration_finished)
        self._worker.failed.connect(self._registration_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    @Slot(object)
    def _registration_finished(self, result: RegistrationResult) -> None:
        self.result = result
        self.table.setRowCount(len(result.shifts_xy))
        for index, (shift_x, shift_y) in enumerate(result.shifts_xy):
            self.table.setItem(index, 0, QTableWidgetItem(str(index + 1)))
            self.table.setItem(index, 1, QTableWidgetItem(f"{shift_x:.5f}"))
            self.table.setItem(index, 2, QTableWidgetItem(f"{shift_y:.5f}"))
        self.status_label.setText("Registration complete — review the preview, then Apply")
        self.run_button.setEnabled(True)
        self.apply_button.setEnabled(True)
        self.preview_slider.setEnabled(True)
        self.preview_slider.setRange(0, result.registered.shape[0] - 1)
        self._update_preview()

    def _update_preview(self) -> None:
        if self.result is None:
            return
        frame = int(self.preview_slider.value())
        show_original = self.show_original_check.isChecked()
        stack = self.data_stack if show_original else self.result.registered
        if stack is None or frame >= stack.shape[0]:
            return
        image = np.asarray(stack[frame], dtype=float)
        finite = image[np.isfinite(image)]
        levels = None if finite.size == 0 else (float(finite.min()), float(finite.max()))
        self.preview.setImage(image, autoLevels=True, autoRange=True, levels=levels)
        energy = self.scan.energies_eV[frame] if frame < len(self.scan.energies_eV) else float("nan")
        which = "original" if show_original else "registered"
        self.preview_label.setText(f"Frame {frame + 1}/{self.result.registered.shape[0]} — {energy:.6g} eV ({which})")

    @Slot(str)
    def _registration_failed(self, message: str) -> None:
        self.status_label.setText(f"Registration failed: {message}")
        self.run_button.setEnabled(True)

    def apply_result(self) -> None:
        if self.result is not None:
            self.resultReady.emit(self.result)
