"""Pre-map generation window."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QMainWindow, QPushButton, QSpinBox, QVBoxLayout, QWidget

from ..io.stxm import ScanStack
from ..processing.premap import net_absorption_map, peak_map, pre_edge_subtract
from ..processing.multivariate import decompose_stack


class PreMapWindow(QMainWindow):
    mapsReady = Signal(object)

    def __init__(self, scan: ScanStack, stack: np.ndarray, source_label: str) -> None:
        super().__init__()
        self.scan = scan
        self.stack = stack
        self.source_label = source_label
        self.setWindowTitle(f"muNEXAFS — Pre-map ({source_label})")
        self.resize(440, 320)
        central = QWidget(self)
        layout = QVBoxLayout(central)
        form = QFormLayout()
        low, high = float(scan.energies_eV[0]), float(scan.energies_eV[-1])
        # The net-absorption map integrates (OD - pre-edge baseline) over the
        # post-edge range.  Default the pre-edge to the first fifth of the scan
        # and the post-edge to the remainder; the user brackets the actual edge.
        split = low + 0.2 * (high - low)
        self.pre_low_spin = QDoubleSpinBox()
        self.pre_high_spin = QDoubleSpinBox()
        self.post_low_spin = QDoubleSpinBox()
        self.post_high_spin = QDoubleSpinBox()
        for spin, value in (
            (self.pre_low_spin, low), (self.pre_high_spin, split),
            (self.post_low_spin, split), (self.post_high_spin, high),
        ):
            spin.setRange(min(low, high), max(low, high))
            spin.setDecimals(4)
            spin.setValue(value)
        form.addRow("Pre-edge start (eV)", self.pre_low_spin)
        form.addRow("Pre-edge end (eV)", self.pre_high_spin)
        form.addRow("Post-edge start (eV)", self.post_low_spin)
        form.addRow("Post-edge end (eV)", self.post_high_spin)
        layout.addLayout(form)
        self.denoise_check = QCheckBox("Low-rank reconstruction denoising")
        self.denoise_check.setChecked(False)
        layout.addWidget(self.denoise_check)
        self.clip_negative = QCheckBox("Clip negative reconstructed OD")
        self.clip_negative.setChecked(True)
        layout.addWidget(self.clip_negative)
        self.denoise_method = QComboBox(); self.denoise_method.addItems(["PCA low-rank (centered)", "SVD low-rank (uncentered)"])
        form.addRow("Denoising method", self.denoise_method)
        self.components_spin = QSpinBox(); self.components_spin.setRange(1, min(50, stack.shape[0])); self.components_spin.setValue(min(3, stack.shape[0]))
        form.addRow("Components", self.components_spin)
        output = QPushButton("Create net absorption + peak maps")
        output.clicked.connect(self.create_maps)
        layout.addWidget(output)
        self.setCentralWidget(central)

    def create_maps(self) -> None:
        pre_edge_range = (self.pre_low_spin.value(), self.pre_high_spin.value())
        post_edge_range = (self.post_low_spin.value(), self.post_high_spin.value())
        source = self.stack
        denoised = None
        method = None
        if self.denoise_check.isChecked():
            method = "SVD (uncentered)" if self.denoise_method.currentIndex() == 1 else "Low-rank reconstruction (centered)"
            denoised = decompose_stack(self.stack, self.components_spin.value(), method).reconstructed
            if self.clip_negative.isChecked():
                denoised = np.maximum(denoised, 0.0)
            source = denoised
        self.mapsReady.emit({
            "net": net_absorption_map(source, self.scan.energies_eV, pre_edge_range, post_edge_range),
            "peak": peak_map(source, self.scan.energies_eV, post_edge_range),
            "presub_stack": pre_edge_subtract(source, self.scan.energies_eV, pre_edge_range),
            "pre_edge_range": list(pre_edge_range),
            "post_edge_range": list(post_edge_range),
            "source": self.source_label,
            "denoised_stack": denoised,
            "multivariate": {"method": method, "components": self.components_spin.value(),
                             "clip_negative": self.clip_negative.isChecked()} if denoised is not None else None,
        })
