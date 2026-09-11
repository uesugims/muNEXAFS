"""Optical-density display window."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..io.stxm import ScanStack
from ..processing.absorbance import OpticalDensityResult, compute_optical_density


class ODWindow(QMainWindow):
    """Display direct/transmission or OD for a selected scan frame."""

    odComputed = Signal(object)
    roiChanged = Signal(object)

    def __init__(self, scan: ScanStack, saved_roi: list[int] | None = None) -> None:
        super().__init__()
        self.scan = scan
        self.od_result: OpticalDensityResult | None = None
        self.direct_roi: pg.RectROI | None = None
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(850, 760)
        self.setWindowTitle("muNEXAFS — Optical density")

        central = QWidget(self)
        layout = QVBoxLayout(central)
        controls = QHBoxLayout()
        self.direct_radio = QRadioButton("Direct transmission")
        self.od_radio = QRadioButton("Optical density (OD)")
        self.direct_radio.setChecked(True)
        self.od_radio.setEnabled(False)
        controls.addWidget(self.direct_radio)
        controls.addWidget(self.od_radio)
        self.roi_button = QPushButton("Select direct-beam ROI")
        controls.addWidget(self.roi_button)
        self.clear_roi_button = QPushButton("Clear ROI")
        self.clear_roi_button.setEnabled(False)
        controls.addWidget(self.clear_roi_button)
        layout.addLayout(controls)
        self.image_view = pg.ImageView(view=pg.PlotItem())
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        # OD ROI selection must not pan or zoom the image.  The ROI itself
        # remains draggable because it is a separate graphics item.
        self.image_view.getView().setMouseEnabled(x=False, y=False)
        layout.addWidget(self.image_view, stretch=1)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, len(scan.frame_paths) - 1)
        layout.addWidget(self.slider)
        self.frame_label = QLabel()
        layout.addWidget(self.frame_label)
        self.roi_label = QLabel("Direct-beam ROI: not selected — OD is not calculated")
        layout.addWidget(self.roi_label)
        self.setCentralWidget(central)

        self.direct_radio.toggled.connect(lambda checked: checked and self._show_frame(self.slider.value()))
        self.od_radio.toggled.connect(lambda checked: checked and self._show_frame(self.slider.value()))
        self.slider.valueChanged.connect(self._show_frame)
        self.roi_button.clicked.connect(self.create_direct_roi)
        self.clear_roi_button.clicked.connect(self.clear_direct_roi)
        if scan.transmission is not None:
            height, width = scan.transmission.shape[1:]
            self.image_view.getView().setRange(
                xRange=(0, width), yRange=(0, height), padding=0, disableAutoRange=True
            )
        self._show_frame(0)
        if saved_roi is not None and len(saved_roi) == 4:
            self.select_direct_roi(*[int(v) for v in saved_roi])

    def create_direct_roi(self) -> None:
        """Create a draggable rectangle and switch the view to ROI selection."""

        if self.scan.transmission is None:
            return
        height, width = self.scan.transmission.shape[1:]
        roi_width = max(4, min(20, width // 8))
        roi_height = max(4, min(20, height // 8))
        if self.direct_roi is None:
            self.direct_roi = pg.RectROI(
                [(width - roi_width) / 2, (height - roi_height) / 2],
                [roi_width, roi_height],
                pen=pg.mkPen("#ffcc00", width=2),
                movable=True,
                resizable=True,
            )
            self.image_view.getView().addItem(self.direct_roi)
            self.direct_roi.sigRegionChanged.connect(self._roi_changed)
        else:
            self.direct_roi.setZValue(20)
        self.image_view.getView().setMouseEnabled(x=False, y=False)
        self.clear_roi_button.setEnabled(True)
        self._calculate_od_from_roi()

        if self.scan.transmission is not None:
            height, width = self.scan.transmission.shape[1:]
            self.image_view.getView().setRange(
                xRange=(0, width), yRange=(0, height), padding=0, disableAutoRange=True
            )

    def clear_direct_roi(self) -> None:
        if self.direct_roi is not None:
            self.image_view.getView().removeItem(self.direct_roi)
            self.direct_roi.deleteLater()
            self.direct_roi = None
        self.image_view.getView().setMouseEnabled(x=True, y=True)
        self.od_result = None
        self.direct_radio.setChecked(True)
        self.od_radio.setEnabled(False)
        self.clear_roi_button.setEnabled(False)
        self.roi_label.setText("Direct-beam ROI: not selected — OD is not calculated")
        self._show_frame(self.slider.value())

    def _roi_changed(self) -> None:
        self._calculate_od_from_roi()
        self._show_frame(self.slider.value())

    def _calculate_od_from_roi(self) -> None:
        if self.scan.transmission is None or self.direct_roi is None:
            return
        height, width = self.scan.transmission.shape[1:]
        pos = self.direct_roi.pos()
        size = self.direct_roi.size()
        x0 = max(0, min(width - 1, int(np.floor(pos.x()))))
        y0 = max(0, min(height - 1, int(np.floor(pos.y()))))
        x1 = max(x0 + 1, min(width, int(np.ceil(pos.x() + size.x()))))
        y1 = max(y0 + 1, min(height, int(np.ceil(pos.y() + size.y()))))
        direct = self.scan.transmission[:, y0:y1, x0:x1]
        i0 = np.nanmean(direct, axis=(1, 2))
        self.od_result = compute_optical_density(self.scan.transmission, i0)
        self.od_radio.setEnabled(True)
        self.roiChanged.emit([x0, y0, x1 - x0, y1 - y0])
        self.odComputed.emit(self.od_result)
        self.roi_label.setText(
            f"Direct-beam ROI: x={x0}:{x1}, y={y0}:{y1} "
            f"({x1 - x0} × {y1 - y0} pixels)"
        )

    def select_direct_roi(self, x: int, y: int, width: int, height: int) -> None:
        """Set an ROI programmatically; useful for tests and saved sessions."""

        if self.direct_roi is None:
            self.create_direct_roi()
        assert self.direct_roi is not None
        self.direct_roi.setPos(x, y)
        self.direct_roi.setSize([width, height])
        self._calculate_od_from_roi()

    def _show_frame(self, frame: int) -> None:
        if self.scan.transmission is None:
            return
        image = self.scan.transmission[frame]
        if self.od_radio.isChecked() and self.od_result is not None:
            image = self.od_result.optical_density[frame]
        finite = image[np.isfinite(image)]
        levels = None if finite.size == 0 else (float(finite.min()), float(finite.max()))
        self.image_view.setImage(image, autoLevels=True, autoRange=False, levels=levels)
        mode = "Optical density (OD)" if self.od_radio.isChecked() else "Direct transmission"
        self.setWindowTitle(f"muNEXAFS — {mode} — {self.scan.energies_eV[frame]:.6g} eV")
        self.frame_label.setText(
            f"Frame {frame + 1}/{len(self.scan.frame_paths)} — "
            f"{self.scan.energies_eV[frame]:.6g} eV"
        )
