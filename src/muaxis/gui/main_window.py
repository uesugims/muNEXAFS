"""Main application window for the muNEXAFS GUI."""

from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import pyqtgraph as pg
# All image data in this application is numpy ``(row, col)`` = ``(y, x)``.
# pyqtgraph defaults to ``col-major``, which would place the first array axis
# (rows) along the screen X axis — displaying every image transposed and, worse,
# mapping ROI X/Y to the wrong numpy axes so ROI spectra sample the transposed
# region.  Select ``row-major`` once, before any ImageItem is created, so that
# screen X = column and screen Y = row throughout the app.
pg.setConfigOption("imageAxisOrder", "row-major")
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QComboBox,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QTextEdit,
    QWidget,
)

from ..io.stxm import ScanStack, StxmFormatError, read_stxm_scan
from .od_window import ODWindow
from .registration_window import RegistrationWindow
from .premap_window import PreMapWindow
from .segmentation_window import SegmentationWindow
from .fitting_window import FittingWindow
from .multivariate_window import MultivariateWindow
from .report_window import ReportWindow


class ImageWindow(QMainWindow):
    """Independent image viewer; multiple instances may be opened."""

    def __init__(self, scan: ScanStack, frame: int = 0) -> None:
        super().__init__()
        self.scan = scan
        self.frame = frame
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(850, 760)

        central = QWidget(self)
        layout = QVBoxLayout(central)
        self.image_view = pg.ImageView(view=pg.PlotItem())
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        layout.addWidget(self.image_view, stretch=1)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, len(scan.frame_paths) - 1)
        self.slider.setValue(frame)
        layout.addWidget(self.slider)
        self.frame_label = QLabel()
        layout.addWidget(self.frame_label)
        self.setCentralWidget(central)
        self.slider.valueChanged.connect(self.show_frame)
        self.show_frame(frame)

    def show_frame(self, frame: int) -> None:
        if self.scan.transmission is None:
            return
        self.frame = frame
        image = self.scan.transmission[frame]
        finite = image[np.isfinite(image)]
        levels = None if finite.size == 0 else (float(finite.min()), float(finite.max()))
        self.image_view.setImage(image, autoLevels=True, autoRange=True, levels=levels)
        self.setWindowTitle(f"muNEXAFS — {self.scan.energies_eV[frame]:.6g} eV")
        self.frame_label.setText(
            f"Frame {frame + 1}/{len(self.scan.frame_paths)} — "
            f"{self.scan.energies_eV[frame]:.6g} eV"
        )


class MainWindow(QMainWindow):
    """Dataset and display hub.

    Analysis modules will consume the selected ``ScanStack`` through the
    session API.  This window deliberately contains no OD, registration, or
    fitting calculation.
    """

    scanLoaded = Signal(object)

    def __init__(self, initial_path: str | Path | None = None) -> None:
        super().__init__()
        self.scan: ScanStack | None = None
        self._current_frame = 0
        self._image_windows: list[ImageWindow] = []
        self._od_windows: list[ODWindow] = []
        self._registration_windows: list[RegistrationWindow] = []
        self._registered_stack: np.ndarray | None = None
        self._registration_config: dict | None = None
        self._premap_windows: list[PreMapWindow] = []
        self._premap_maps: dict[str, np.ndarray] = {}
        self._premap_config: dict | None = None
        self._premap_display: dict[str, dict] = {}
        self._analysis_stack: np.ndarray | None = None
        # Pre-edge-subtracted OD stack: a viewable per-frame layer created by
        # Pre-map, and the composition-based clustering input for Segmentation.
        self._presub_stack: np.ndarray | None = None
        self._presub_layer_name: str | None = None
        self._segmentation_windows: list[SegmentationWindow] = []
        self._segmentation_result: object | None = None
        self._segmentation_config: dict | None = None
        self._fitting_windows: list[FittingWindow] = []
        self._multivariate_windows: list[MultivariateWindow] = []
        self._report_windows: list[ReportWindow] = []
        self._fitting_result: object | None = None
        self._fitting_config: dict | None = None
        self._multivariate_result: object | None = None
        self._updating_map_display = False
        self._spectrum_rois: list[pg.RectROI] = []
        self._spectrum_curves: list[pg.PlotDataItem] = []
        self.normalization_combo: QComboBox | None = None
        self.norm_e1: QDoubleSpinBox | None = None
        self.norm_e2: QDoubleSpinBox | None = None
        self.energy_line: pg.InfiniteLine | None = None
        self.energy_cursor_label: QLabel | None = None
        self._updating_energy_line = False
        self._od_result: object | None = None
        self._od_roi_bounds: list[int] | None = None
        self._recent_path = Path.home() / ".muaxis_recent_scans.json"
        self._session_index_path = Path.home() / ".muaxis_sessions.json"
        self._session_fallback_path = Path.cwd() / ".muaxis_sessions.json"
        self._session_index: dict[str, dict] = self._read_session_index()
        self._session_path: Path | None = None
        self._restoring_session = False
        self._closing = False

        self.setWindowTitle("muNEXAFS — STXM-NEXAFS")
        self.resize(1440, 900)
        self._build_actions()
        self._build_central_view()
        self._build_docks()
        self._connect_signals()
        self._load_recent_scans()

        if initial_path is not None:
            self.load_scan(Path(initial_path))

    def _build_actions(self) -> None:
        open_action = QAction("Open scan…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_scan_dialog)
        self.save_action = QAction("Save session…", self)
        self.save_action.setEnabled(False)
        self.save_action.triggered.connect(self.save_session_dialog)
        self.new_image_action = QAction("New image window", self)
        self.new_image_action.setEnabled(False)
        self.new_image_action.triggered.connect(self.open_image_window)
        self.select_roi_action = QAction("Select spectrum ROI", self)
        self.select_roi_action.setEnabled(False)
        self.select_roi_action.triggered.connect(self.create_spectrum_roi)
        self.clear_roi_action = QAction("Clear spectrum ROI", self)
        self.clear_roi_action.setEnabled(False)
        self.clear_roi_action.triggered.connect(self.clear_spectrum_roi)
        self.od_action = QAction("OD window", self)
        self.od_action.setEnabled(False)
        self.od_action.triggered.connect(self.open_od_window)
        self.registration_action = QAction("Registration", self)
        self.registration_action.setEnabled(False)
        self.registration_action.triggered.connect(self.open_registration_window)
        self.premap_action = QAction("Pre-map", self)
        self.premap_action.setEnabled(False)
        self.premap_action.triggered.connect(self.open_premap_window)
        self.status_label = QLabel("No scan loaded")
        toolbar = QToolBar("Status", self)
        toolbar.setObjectName("statusToolBar")
        toolbar.addWidget(self.status_label)
        self.addToolBar(toolbar)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(open_action)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        roi_menu = self.menuBar().addMenu("ROI")
        roi_menu.addAction(self.select_roi_action)
        roi_menu.addAction(self.clear_roi_action)

        process_menu = self.menuBar().addMenu("Process")
        process_menu.addAction(self.od_action)
        # Reserved entries keep subsequent analysis modules discoverable.
        process_menu.addAction(self.registration_action)
        process_menu.addAction(self.premap_action)
        segmentation_action = QAction("Segmentation", self)
        segmentation_action.setEnabled(False)
        segmentation_action.triggered.connect(self.open_segmentation_window)
        self.segmentation_action = segmentation_action
        process_menu.addAction(segmentation_action)
        self.fitting_action = QAction("PTEE(-R2)", self)
        self.fitting_action.setEnabled(False); self.fitting_action.triggered.connect(self.open_fitting_window); process_menu.addAction(self.fitting_action)
        self.multivariate_action = QAction("PCA + clustering", self)
        self.multivariate_action.setEnabled(False); self.multivariate_action.triggered.connect(self.open_multivariate_window); process_menu.addAction(self.multivariate_action)

        self.window_menu = self.menuBar().addMenu("Window")
        self.window_menu.aboutToShow.connect(self._rebuild_window_menu)
        self._rebuild_window_menu()
        output_menu = self.menuBar().addMenu("Output")
        report_action = QAction("Report output…", self)
        report_action.triggered.connect(self.open_report_window)
        self.report_action = report_action
        output_menu.addAction(report_action)

    def _build_central_view(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        self.image_view = pg.ImageView(view=pg.PlotItem())
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_view.setMinimumSize(600, 450)
        layout.addWidget(self.image_view, stretch=3)

        self.spectrum_plot = pg.PlotWidget(title="Select spectrum ROI")
        self.spectrum_plot.getPlotItem().setMouseEnabled(x=False, y=False)
        self.spectrum_plot.setLabel("bottom", "Energy", units="eV")
        self.spectrum_plot.setLabel("left", "Transmission", units="counts")
        norm_controls = QHBoxLayout()
        norm_controls.addWidget(QLabel("Spectrum normalization:"))
        self.normalization_combo = QComboBox()
        self.normalization_combo.addItems(["None", "Each plot min–max", "Two energy values"])
        self.normalization_combo.currentIndexChanged.connect(self._normalization_changed)
        norm_controls.addWidget(self.normalization_combo)
        self.norm_e1 = QDoubleSpinBox()
        self.norm_e2 = QDoubleSpinBox()
        for spin in (self.norm_e1, self.norm_e2):
            spin.setDecimals(4)
            spin.setEnabled(False)
            spin.valueChanged.connect(self._normalization_changed)
        norm_controls.addWidget(QLabel("E1"))
        norm_controls.addWidget(self.norm_e1)
        norm_controls.addWidget(QLabel("E2"))
        norm_controls.addWidget(self.norm_e2)
        norm_controls.addStretch()
        layout.addLayout(norm_controls)
        self.energy_cursor_label = QLabel("Energy cursor: —")
        layout.addWidget(self.energy_cursor_label)
        self.energy_line = pg.InfiniteLine(
            angle=90, movable=True, pen=pg.mkPen("#ffeb3b", width=2), hoverPen=pg.mkPen("#ffffff", width=3)
        )
        self.energy_line.sigPositionChanged.connect(self._energy_line_changed)
        self.spectrum_plot.addItem(self.energy_line)
        # Whole-image mean spectrum: shown when no spectrum ROI is selected, and
        # hidden as soon as one or more ROIs exist (their per-region means replace
        # it).  Kept as a persistent curve so the main spectrum plot is never
        # cleared wholesale (which would also drop the energy cursor line).
        self.whole_image_curve = self.spectrum_plot.plot(
            pen=pg.mkPen("#9e9e9e", width=2, style=Qt.PenStyle.DashLine), name="Whole-image mean"
        )
        self.whole_image_curve.setVisible(False)
        # Clicking or dragging anywhere on the spectrum plot moves the energy
        # cursor (the vertical bar) to that energy, selecting the frame — this
        # replaces the frame slider.
        self.spectrum_plot.scene().sigMouseClicked.connect(self._spectrum_mouse_clicked)
        self.spectrum_plot.scene().sigMouseMoved.connect(self._spectrum_mouse_moved)
        layout.addWidget(self.spectrum_plot, stretch=1)
        self.setCentralWidget(central)
        self.image_view.ui.histogram.region.sigRegionChanged.connect(self._map_levels_changed)
        self.image_view.ui.histogram.region.sigRegionChangeFinished.connect(self._map_levels_changed)
        self.image_view.ui.histogram.region.show()
        _hide_gradient_ticks(self.image_view)

    def _build_docks(self) -> None:
        self.recent_list = QListWidget(self)
        self.recent_list.setMinimumWidth(180)
        self.recent_list.setToolTip("Previously opened STXM header files")

        scan_dock = QDockWidget("Recent scans", self)
        scan_dock.setObjectName("scanFramesDock")
        left_panel = QWidget(self)
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Recent scans"))
        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("Sort:"))
        self.recent_sort_combo = QComboBox()
        self.recent_sort_combo.addItems(["Filename", "Filename (reverse)", "Name", "Name (reverse)"])
        self.recent_sort_combo.currentIndexChanged.connect(self._sort_recent_scans)
        sort_row.addWidget(self.recent_sort_combo)
        left_layout.addLayout(sort_row)
        left_layout.addWidget(self.recent_list, stretch=1)
        left_layout.addWidget(QLabel("Measurement name"))
        name_row = QHBoxLayout()
        self.measurement_name_edit = QLineEdit()
        self.measurement_name_edit.setPlaceholderText("Optional measurement name")
        self.measurement_name_edit.editingFinished.connect(self._save_measurement_name)
        name_row.addWidget(self.measurement_name_edit)
        self.save_name_button = QPushButton("Save")
        self.save_name_button.clicked.connect(self._save_measurement_name)
        name_row.addWidget(self.save_name_button)
        left_layout.addLayout(name_row)
        left_layout.addWidget(QLabel("Measurement summary"))
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMinimumHeight(155)
        self.summary_box.setMaximumHeight(230)
        left_layout.addWidget(self.summary_box)
        scan_dock.setWidget(left_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, scan_dock)
        scan_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        scan_dock.setMinimumWidth(180)
        scan_dock.setMaximumWidth(230)

        controls = QWidget(self)
        controls_layout = QVBoxLayout(controls)
        self.open_button = QPushButton("Open .hdr…")
        controls_layout.addWidget(self.open_button)
        # Frame selection is driven by the energy cursor on the spectrum plot;
        # this box is the numeric read-out and lets the user jump to a frame by
        # typing its number and pressing Enter (there is no frame slider).
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame"))
        self.frame_edit = QLineEdit()
        self.frame_edit.setEnabled(False)
        self.frame_edit.setMaximumWidth(70)
        self.frame_edit.setToolTip("Type a frame number and press Enter to jump to it")
        frame_row.addWidget(self.frame_edit)
        self.frame_total_label = QLabel("/ —")
        frame_row.addWidget(self.frame_total_label)
        frame_row.addStretch()
        controls_layout.addLayout(frame_row)
        self.frame_energy_label = QLabel("Energy: —")
        controls_layout.addWidget(self.frame_energy_label)
        # ``frame_label`` still carries the per-frame / map status text used
        # throughout ``_show_frame``.
        self.frame_label = QLabel("—")
        controls_layout.addWidget(self.frame_label)
        controls_layout.addWidget(QLabel("Current image layer"))
        self.layer_list = QListWidget()
        self.layer_list.addItem("Transmission (raw)")
        self.layer_list.setEnabled(False)
        controls_layout.addWidget(self.layer_list)
        controls_layout.addWidget(QLabel("Map color palette"))
        self.map_palette_combo = QComboBox()
        self.map_palette_combo.addItems(["Jet", "Fire", "Grayscale"])
        self.map_palette_combo.setEnabled(False)
        self.map_palette_combo.currentIndexChanged.connect(self._map_palette_changed)
        controls_layout.addWidget(self.map_palette_combo)
        self.reset_map_levels_button = QPushButton("Reset map color range")
        self.reset_map_levels_button.setEnabled(False)
        self.reset_map_levels_button.clicked.connect(self._reset_map_levels)
        controls_layout.addWidget(self.reset_map_levels_button)
        controls_layout.addStretch()

        info_dock = QDockWidget("Display / dataset", self)
        info_dock.setObjectName("displayDatasetDock")
        info_dock.setWidget(controls)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, info_dock)

    def _connect_signals(self) -> None:
        self.open_button.clicked.connect(self.open_scan_dialog)
        self.frame_edit.returnPressed.connect(self._frame_edit_entered)
        self.layer_list.currentRowChanged.connect(self._layer_changed)
        self.recent_list.itemClicked.connect(self._recent_item_activated)

    def open_scan_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open STXM header",
            str(Path.home()),
            "STXM header (*.hdr);;All files (*)",
        )
        if path:
            self.load_scan(Path(path))

    def load_scan(self, path: Path) -> None:
        try:
            scan = read_stxm_scan(path)
        except (OSError, StxmFormatError, ValueError) as exc:
            QMessageBox.critical(self, "Cannot open STXM scan", str(exc))
            self.status_label.setText("Load failed")
            return

        self.scan = scan
        # Suppress session writes for the whole load.  Configuring the
        # normalization widgets below emits value-changed signals that reach
        # ``_persist_scan_session``; if that runs before the guard is set it
        # overwrites the sidecar with the blank load-time UI state — layer=-1
        # and the *previous* scan's still-uncleared fitting/analysis config —
        # destroying the saved active layer (and cross-contaminating scans)
        # before ``_restore_scan_session`` has even read it.
        self._restoring_session = True
        self._configure_normalization(scan.energies_eV)
        self._update_summary()
        self._session_path = None
        self._reset_analysis_state()
        self._current_frame = 0
        self.frame_total_label.setText(f"/ {len(scan.frame_paths)}")
        self.frame_edit.setEnabled(True)
        self.layer_list.setEnabled(True)
        self.select_roi_action.setEnabled(True)
        self.save_action.setEnabled(True)
        self.new_image_action.setEnabled(True)
        self.od_action.setEnabled(True)
        self.registration_action.setEnabled(True)
        self.premap_action.setEnabled(True)
        self.multivariate_action.setEnabled(True)
        self.segmentation_action.setEnabled(self._od_result is not None)
        self.fitting_action.setEnabled(self._od_result is not None)
        self.multivariate_action.setEnabled(True)
        self.status_label.setText(f"Loaded: {path.name}")
        self._add_recent_scan(path)
        self.scanLoaded.emit(scan)
        self._restore_scan_session(path)
        # Session restoration may recreate OD after the initial action setup;
        # refresh dependent Process menu items once restoration is complete.
        self.segmentation_action.setEnabled(self._od_result is not None)
        self.fitting_action.setEnabled(self._od_result is not None)
        self._load_measurement_name(path)
        # Do not write here.  The sidecar has just been read and may contain
        # analysis state that is still being reconstructed.  Writing the
        # initial blank UI state at this point used to erase the sidecar when
        # any later restore step raised an exception.  Subsequent user
        # actions and closeEvent persist the fully restored state.
        self._show_frame(self._current_frame)
        self._restoring_session = False

    def _frame_count(self) -> int:
        return len(self.scan.frame_paths) if self.scan is not None else 0

    def _select_frame(self, frame: int) -> None:
        if self.scan is None:
            return
        frame = int(np.clip(frame, 0, max(0, self._frame_count() - 1)))
        self._show_frame(frame)
        self._persist_scan_session()

    def _frame_edit_entered(self) -> None:
        """Jump to the frame number typed in the frame box (1-based)."""
        if self.scan is None:
            return
        try:
            index = int(self.frame_edit.text()) - 1
        except (TypeError, ValueError):
            # Restore the current value on invalid input.
            self.frame_edit.setText(str(self._current_frame + 1))
            return
        self._select_frame(index)

    def _spectrum_mouse_clicked(self, event: object) -> None:
        """Move the energy cursor to the clicked energy (frame selection)."""
        try:
            scene_pos = event.scenePos()
        except AttributeError:
            return
        self._move_energy_cursor_to_scene(scene_pos)

    def _spectrum_mouse_moved(self, scene_pos: object) -> None:
        """Scrub the energy cursor while the left mouse button is held down."""
        if not (QApplication.mouseButtons() & Qt.MouseButton.LeftButton):
            return
        self._move_energy_cursor_to_scene(scene_pos)

    def _move_energy_cursor_to_scene(self, scene_pos: object) -> None:
        if self.scan is None or self.energy_line is None or not self.frame_edit.isEnabled():
            return
        plot_item = self.spectrum_plot.getPlotItem()
        if not plot_item.sceneBoundingRect().contains(scene_pos):
            return
        energy = float(plot_item.vb.mapSceneToView(scene_pos).x())
        lo, hi = float(np.min(self.scan.energies_eV)), float(np.max(self.scan.energies_eV))
        energy = min(max(energy, lo), hi)
        # Setting the line value triggers ``_energy_line_changed``, which snaps
        # to the nearest frame and updates the image.
        self.energy_line.setValue(energy)

    def _layer_changed(self, _row: int) -> None:
        self._show_frame(self._current_frame)
        self._persist_scan_session()

    def _show_frame(self, frame: int) -> None:
        if self.scan is None or self.scan.transmission is None:
            return
        self._current_frame = frame
        layer_name = self.layer_list.currentItem().text() if self.layer_list.currentItem() else ""
        if layer_name == "PTEE RGBY map" and self._fitting_result is not None:
            self.frame_edit.setEnabled(False); self.map_palette_combo.setEnabled(False); self.reset_map_levels_button.setEnabled(False)
            # rgb_y holds premultiplied display colours already in [0, 1].  The
            # image view is shared with the grayscale/map layers, so its item
            # keeps their levels (e.g. OD's ~[0, 3]).  Without an explicit range
            # those levels would scale the colours toward black and the map
            # would look blank; pin levels to [0, 1] so it renders at full
            # brightness regardless of the previously shown layer.
            self.image_view.setImage(self._fitting_result.rgb_y, autoLevels=False, autoRange=True, levels=(0.0, 1.0))  # type: ignore[attr-defined]
            self.frame_label.setText("Map — PTEE RGBY"); self.statusBar().showMessage("PTEE RGBY map"); self._update_spectrum(); return
        if layer_name == "SVD/PCA RGB map" and self._multivariate_result is not None:
            self.frame_edit.setEnabled(False)
            # rgb_map is an RGBA image already in [0, 1].  The image view is
            # shared with the grayscale map layers, so its item keeps their
            # levels (a peak map's are the energy range, e.g. ~[280, 300]).
            # Without an explicit range those levels would crush the colours to
            # black and the map would look blank; pin levels to [0, 1] so it
            # renders correctly regardless of the previously shown layer.
            self.image_view.setImage(self._multivariate_result.rgb_map, autoLevels=False, autoRange=True, levels=(0.0, 1.0))  # type: ignore[attr-defined]
            self.frame_label.setText("Map — SVD/PCA RGB"); self.statusBar().showMessage("SVD/PCA RGB map"); self._update_spectrum(); return
        if layer_name == "PCA cluster map" and self._multivariate_result is not None:
            self.frame_edit.setEnabled(False)
            # rgb_map is an RGBA cluster map already in [0, 1]; pin levels so the
            # colours render correctly regardless of the previously shown layer.
            self.image_view.setImage(self._multivariate_result.rgb_map, autoLevels=False, autoRange=True, levels=(0.0, 1.0))  # type: ignore[attr-defined]
            self.frame_label.setText("Map — PCA clusters"); self.statusBar().showMessage("PCA cluster map"); self._update_spectrum(); return
        if layer_name in self._premap_maps:
            self.frame_edit.setEnabled(False)
            self.map_palette_combo.setEnabled(True)
            self.reset_map_levels_button.setEnabled(True)
            display = self._premap_display.setdefault(layer_name, {"palette": "Jet", "levels": None})
            palette_names = {"Jet": 0, "Fire": 1, "Grayscale": 2}
            self._updating_map_display = True
            self.map_palette_combo.setCurrentIndex(palette_names.get(display["palette"], 0))
            self.image_view.setColorMap(_map_colormap(display["palette"]))
            _hide_gradient_ticks(self.image_view)
            image = self._premap_maps[layer_name]
            levels = display.get("levels")
            if levels is None:
                finite = image[np.isfinite(image)]
                levels = None if finite.size == 0 else (float(finite.min()), float(finite.max()))
            self.image_view.setImage(image, autoLevels=True, autoRange=True, levels=levels)
            self._updating_map_display = False
            if display.get("levels") is None:
                current_levels = self.image_view.getLevels()
                if current_levels is not None:
                    display["levels"] = [float(value) for value in current_levels]
            self.frame_label.setText(f"Map — {layer_name}")
            self.statusBar().showMessage(f"{layer_name} | color levels adjustable in the right histogram")
            self._update_spectrum()
            return
        self.frame_edit.setEnabled(True)
        self.map_palette_combo.setEnabled(False)
        self.reset_map_levels_button.setEnabled(False)
        self.image_view.setColorMap(_gray_colormap())
        _hide_gradient_ticks(self.image_view)
        self._set_energy_line(frame)
        image = self._selected_layer_image(frame)
        finite = image[np.isfinite(image)]
        levels = None if finite.size == 0 else (float(finite.min()), float(finite.max()))
        self.image_view.setImage(image, autoLevels=True, autoRange=True, levels=levels)
        self.frame_label.setText(
            f"{frame + 1}/{len(self.scan.frame_paths)} — "
            f"{self.scan.energies_eV[frame]:.6g} eV"
        )
        self._update_frame_readout(frame)
        metadata = self.scan.header.frames[frame]
        current = "—" if metadata.storage_ring_current is None else f"{metadata.storage_ring_current:.6g}"
        self.statusBar().showMessage(
            f"Frame {frame + 1}/{len(self.scan.frame_paths)} | "
            f"Energy {metadata.energy_eV:.6g} eV | Current {current} | "
            f"{self.scan.frame_paths[frame].name}"
        )
        self._update_spectrum()

    def _set_energy_line(self, frame: int) -> None:
        if self.scan is None or self.energy_line is None:
            return
        energy = float(self.scan.energies_eV[frame])
        self._updating_energy_line = True
        self.energy_line.setValue(energy)
        self._updating_energy_line = False
        if self.energy_cursor_label is not None:
            self.energy_cursor_label.setText(f"Energy cursor: {energy:.6g} eV")

    def _update_frame_readout(self, frame: int) -> None:
        """Refresh the frame-number box and its energy label (no signals)."""
        if self.scan is None:
            return
        self.frame_edit.blockSignals(True)
        self.frame_edit.setText(str(frame + 1))
        self.frame_edit.blockSignals(False)
        self.frame_energy_label.setText(f"Energy: {self.scan.energies_eV[frame]:.6g} eV")

    def _energy_line_changed(self, line: pg.InfiniteLine) -> None:
        if self._updating_energy_line or self.scan is None:
            return
        energy = float(line.value())
        index = int(np.argmin(np.abs(self.scan.energies_eV - energy)))
        if self.energy_cursor_label is not None:
            self.energy_cursor_label.setText(
                f"Energy cursor: {self.scan.energies_eV[index]:.6g} eV"
            )
        if index != self._current_frame:
            self._select_frame(index)

    def _selected_layer_image(self, frame: int) -> np.ndarray:
        assert self.scan is not None and self.scan.transmission is not None
        layer = self.layer_list.currentItem().text() if self.layer_list.currentItem() else ""
        if layer == self._presub_layer_name and self._presub_stack is not None:
            return self._presub_stack[frame]
        if layer == "Optical density (OD)" and self._od_result is not None:
            return self._od_result.optical_density[frame]  # type: ignore[attr-defined]
        if layer in {"Registered transmission", "Registered OD"} and self._registered_stack is not None:
            return self._registered_stack[frame]
        return self.scan.transmission[frame]

    def _selected_layer_stack(self) -> np.ndarray:
        assert self.scan is not None and self.scan.transmission is not None
        layer = self.layer_list.currentItem().text() if self.layer_list.currentItem() else ""
        if layer == self._presub_layer_name and self._presub_stack is not None:
            return self._presub_stack
        if "(OD)" in layer and self._od_result is not None:
            return self._od_result.optical_density  # type: ignore[attr-defined]
        if layer == "Optical density (OD)" and self._od_result is not None:
            return self._od_result.optical_density  # type: ignore[attr-defined]
        if layer in {"Registered transmission", "Registered OD"} and self._registered_stack is not None:
            return self._registered_stack
        if layer in {"PTEE RGBY map", "SVD/PCA RGB map", "PCA cluster map", "Segmentation labels"} and self._od_result is not None:
            return self._od_result.optical_density  # type: ignore[attr-defined]
        return self.scan.transmission

    def _update_spectrum(self) -> None:
        if self.scan is None or self.scan.transmission is None:
            return
        stack = self._selected_layer_stack()
        height, width = self.scan.transmission.shape[1:]
        layer_name = self.layer_list.currentItem().text() if self.layer_list.currentItem() else ""
        label = "OD" if ("OD" in layer_name or layer_name in {"PTEE RGBY map", "SVD/PCA RGB map", "PCA cluster map", "Segmentation labels"}) else "transmission"
        normalized = self.normalization_combo is not None and self.normalization_combo.currentIndex() != 0
        self.spectrum_plot.setLabel("left", f"Normalized {label}" if normalized else label, units=None)
        if self._spectrum_rois:
            # With one or more ROIs, the whole-image mean is replaced by the
            # per-ROI means.
            self.whole_image_curve.setVisible(False)
            for roi, curve in zip(self._spectrum_rois, self._spectrum_curves):
                x0, y0, x1, y1 = self._roi_bounds(roi, width, height)
                values = np.nanmean(stack[:, y0:y1, x0:x1], axis=(1, 2))
                values = self._normalize_spectrum(values)
                curve.setData(self.scan.energies_eV, values)
                curve.setVisible(True)
            self.spectrum_plot.setTitle(f"Mean {label} — {len(self._spectrum_rois)} ROI(s)")
        else:
            # No ROI selected: show the whole-image mean spectrum.
            values = self._normalize_spectrum(np.nanmean(stack, axis=(1, 2)))
            self.whole_image_curve.setData(self.scan.energies_eV, values)
            self.whole_image_curve.setVisible(True)
            self.spectrum_plot.setTitle(f"Whole-image mean {label}")

    def _configure_normalization(self, energies: np.ndarray) -> None:
        if self.norm_e1 is None or self.norm_e2 is None:
            return
        lo, hi = float(np.min(energies)), float(np.max(energies))
        for spin, value in ((self.norm_e1, lo), (self.norm_e2, hi)):
            spin.blockSignals(True)
            spin.setRange(lo, hi)
            spin.setValue(value)
            spin.blockSignals(False)
        if self.energy_line is not None:
            self.energy_line.setBounds((lo, hi))
            self.energy_line.setValue(lo)
        self.spectrum_plot.setXRange(lo, hi, padding=0)
        self._normalization_changed()

    def _normalization_changed(self, *_args: object) -> None:
        enabled = self.normalization_combo is not None and self.normalization_combo.currentIndex() == 2
        if self.norm_e1 is not None:
            self.norm_e1.setEnabled(enabled)
        if self.norm_e2 is not None:
            self.norm_e2.setEnabled(enabled)
        if self.normalization_combo is not None:
            layer_name = self.layer_list.currentItem().text() if self.layer_list.currentItem() else ""
            y_label = "OD" if "OD" in layer_name else "Transmission"
            if self.normalization_combo.currentIndex() != 0:
                y_label = f"Normalized {y_label}"
            self.spectrum_plot.setLabel(
                "left", y_label,
                units=None,
            )
        self._update_spectrum()
        self._persist_scan_session()

    def _normalize_spectrum(self, values: np.ndarray) -> np.ndarray:
        if self.normalization_combo is None or self.normalization_combo.currentIndex() == 0:
            return values
        result = np.asarray(values, dtype=float).copy()
        finite = np.isfinite(result)
        if not finite.any():
            return result
        if self.normalization_combo.currentIndex() == 1:
            low, high = np.nanmin(result), np.nanmax(result)
        else:
            assert self.scan is not None and self.norm_e1 is not None and self.norm_e2 is not None
            e1, e2 = self.norm_e1.value(), self.norm_e2.value()
            if e1 == e2:
                return np.full_like(result, np.nan)
            low, high = np.interp([e1, e2], self.scan.energies_eV, result)
        scale = high - low
        return (result - low) / scale if np.isfinite(scale) and scale != 0 else np.full_like(result, np.nan)

    def _update_summary(self) -> None:
        if self.scan is None:
            self.summary_box.clear()
            return
        energies = self.scan.energies_eV
        h = self.scan.header
        x_step = self._axis_step(h.x_um)
        y_step = self._axis_step(h.y_um)
        first_time = h.frames[0].acquired_at or "—"
        lines = [
            f"Label: {h.label or '—'}",
            f"Date/time: {first_time}",
            f"Energy: {energies[0]:.6g}–{energies[-1]:.6g} eV",
            f"Frames: {len(energies)}",
            f"Pixels: {self.scan.shape[2]} × {self.scan.shape[1]}" if self.scan.shape else "Pixels: —",
            f"X step: {x_step}",
            f"Y step: {y_step}",
            f"Dwell time: {h.dwell_time_ms:g} ms" if h.dwell_time_ms is not None else "Dwell time: —",
        ]
        self.summary_box.setPlainText("\n".join(lines))

    @staticmethod
    def _axis_step(axis: np.ndarray | None) -> str:
        if axis is None or len(axis) < 2:
            return "—"
        return f"{float(np.median(np.diff(axis))):.6g} µm"

    @staticmethod
    def _roi_bounds(roi: pg.RectROI, width: int, height: int) -> tuple[int, int, int, int]:
        pos = roi.pos()
        size = roi.size()
        x0 = max(0, min(width - 1, int(np.floor(pos.x()))))
        y0 = max(0, min(height - 1, int(np.floor(pos.y()))))
        x1 = max(x0 + 1, min(width, int(np.ceil(pos.x() + size.x()))))
        y1 = max(y0 + 1, min(height, int(np.ceil(pos.y() + size.y()))))
        return x0, y0, x1, y1

    def create_spectrum_roi(self, bounds: tuple[int, int, int, int] | None = None) -> None:
        """Create a draggable ROI whose mean spectrum is shown below the image."""

        if self.scan is None or self.scan.transmission is None:
            return
        # QAction.triggered carries a checked(bool) argument.  It must not be
        # mistaken for the optional programmatic ROI bounds.
        if isinstance(bounds, bool):
            bounds = None
        height, width = self.scan.transmission.shape[1:]
        roi_width = max(4, min(20, width // 8))
        roi_height = max(4, min(20, height // 8))
        colors = ["#42d77d", "#ff8a65", "#64b5f6", "#ce93d8", "#ffd54f", "#4dd0e1"]
        index = len(self._spectrum_rois)
        if bounds is None:
            x, y, w, h = ((width - roi_width) / 2 + index * 4,
                          (height - roi_height) / 2 + index * 4,
                          roi_width, roi_height)
        else:
            x, y, w, h = bounds
        roi = pg.RectROI(
            [x, y],
            [w, h],
            pen=pg.mkPen(colors[index % len(colors)], width=2),
            movable=True,
            resizable=True,
        )
        roi.setZValue(100)
        roi.show()
        self._spectrum_rois.append(roi)
        self.image_view.getView().addItem(roi, ignoreBounds=True)
        roi.sigRegionChanged.connect(self._spectrum_roi_changed)
        self._spectrum_curves.append(
            self.spectrum_plot.plot(
                pen=pg.mkPen(colors[index % len(colors)], width=2),
                name=f"ROI {index + 1}",
            )
        )
        self.image_view.getView().setMouseEnabled(x=False, y=False)
        self.clear_roi_action.setEnabled(True)
        self._update_spectrum()
        self._persist_scan_session()

    def clear_spectrum_roi(self) -> None:
        for roi in self._spectrum_rois:
            self.image_view.getView().removeItem(roi)
            roi.deleteLater()
        for curve in self._spectrum_curves:
            self.spectrum_plot.removeItem(curve)
        self._spectrum_rois.clear()
        self._spectrum_curves.clear()
        self.image_view.getView().setMouseEnabled(x=True, y=True)
        self.clear_roi_action.setEnabled(False)
        self._update_spectrum()
        self._persist_scan_session()

    def _spectrum_roi_changed(self) -> None:
        self._update_spectrum()
        self._persist_scan_session()

    def open_image_window(self) -> None:
        if self.scan is None:
            return
        viewer = ImageWindow(self.scan, self._current_frame)
        self._image_windows.append(viewer)
        viewer.destroyed.connect(lambda: self._remove_image_window(viewer))
        viewer.show()
        self._rebuild_window_menu()

    def _remove_image_window(self, viewer: ImageWindow) -> None:
        if viewer in self._image_windows:
            self._image_windows.remove(viewer)
        if self._closing:
            return
        self._rebuild_window_menu()

    def open_od_window(self) -> None:
        if self.scan is None:
            return
        viewer = ODWindow(self.scan, self._od_roi_bounds)
        self._od_windows.append(viewer)
        viewer.roiChanged.connect(self._set_od_roi)
        viewer.odComputed.connect(self._set_od_result)
        viewer.destroyed.connect(lambda: self._child_window_closed("od", viewer))
        viewer.show()
        self._rebuild_window_menu()

    def open_registration_window(self) -> None:
        if self.scan is None:
            return
        use_od = self._od_result is not None
        data_stack = self._od_result.optical_density if use_od else self.scan.transmission  # type: ignore[attr-defined]
        viewer = RegistrationWindow(self.scan, data_stack, "OD" if use_od else "Transmission")
        self._registration_windows.append(viewer)
        # Applying the registration also (re)builds the pre-map with default
        # ranges, so the downstream maps are always in sync with the alignment.
        viewer.resultReady.connect(lambda result: self._set_registration_result(result, auto_premap=True))
        viewer.destroyed.connect(lambda: self._registration_window_closed(viewer))
        viewer.show()
        self._rebuild_window_menu()

    def _remove_registration_window(self, viewer: RegistrationWindow) -> None:
        if viewer in self._registration_windows:
            self._registration_windows.remove(viewer)
        self._rebuild_window_menu()

    def _child_window_closed(self, kind: str, viewer: object) -> None:
        if self._closing:
            return
        if kind == "od":
            self._remove_od_window(viewer)  # type: ignore[arg-type]
        elif kind == "premap":
            self._remove_premap_window(viewer)  # type: ignore[arg-type]
        elif kind == "fitting" and viewer in self._fitting_windows:
            self._fitting_windows.remove(viewer)  # type: ignore[arg-type]
            self._rebuild_window_menu()
        self._persist_scan_session()

    def _registration_window_closed(self, viewer: RegistrationWindow) -> None:
        # Closing after a successful Run must not discard the calculated
        # registration merely because Apply was not clicked.
        if self._closing:
            return
        if getattr(viewer, "result", None) is not None:
            self._set_registration_result(viewer.result, auto_premap=True)
        self._remove_registration_window(viewer)
        self._persist_scan_session()

    def _set_registration_result(self, result: object, auto_premap: bool = False) -> None:
        self._registered_stack = result.registered  # type: ignore[attr-defined]
        self._registration_config = {
            "reference_index": int(result.reference_index),  # type: ignore[attr-defined]
            "reference_mode": str(result.reference_mode),  # type: ignore[attr-defined]
            "upsample_factor": int(result.upsample_factor),  # type: ignore[attr-defined]
        }
        registered_label = "Registered OD" if self._od_result is not None else "Registered transmission"
        if not self.layer_list.findItems(registered_label, Qt.MatchFlag.MatchExactly):
            self.layer_list.addItem(registered_label)
        # Move the active layer to the registered layer after registration.
        registered_row = self.layer_list.findItems(registered_label, Qt.MatchFlag.MatchExactly)[0]
        self.layer_list.setCurrentItem(registered_row)
        # Applying a registration rebuilds the pre-map from the aligned stack
        # with default energy ranges.  The Pre-map window is only needed when
        # the user wants to change those ranges (it re-creates the maps then).
        if auto_premap:
            self._auto_create_premap()
        self._persist_scan_session()
        self._show_frame(self._current_frame)

    def _auto_create_premap(self) -> None:
        """Create net-absorption / peak / pre-edge maps from the registered
        stack with default energy ranges (pre-edge = first fifth, post-edge =
        remainder)."""
        if self.scan is None or self._registered_stack is None:
            return
        from ..processing.premap import net_absorption_map, peak_map, pre_edge_subtract
        energies = self.scan.energies_eV
        source_stack = self._registered_stack
        source_label = "OD" if self._od_result is not None else "Transmission"
        low, high = float(energies[0]), float(energies[-1])
        split = low + 0.2 * (high - low)
        pre_edge_range = (low, split)
        post_edge_range = (split, high)
        try:
            self._set_premap_maps({
                "net": net_absorption_map(source_stack, energies, pre_edge_range, post_edge_range),
                "peak": peak_map(source_stack, energies, post_edge_range),
                "presub_stack": pre_edge_subtract(source_stack, energies, pre_edge_range),
                "pre_edge_range": list(pre_edge_range),
                "post_edge_range": list(post_edge_range),
                "source": source_label,
                "denoised_stack": None,
            })
        except (ValueError, KeyError):
            # Degenerate energy axes (e.g. a single frame) simply skip the map.
            pass

    def open_premap_window(self) -> None:
        if self.scan is None:
            return
        if self._od_result is not None:
            stack = self._od_result.optical_density  # type: ignore[attr-defined]
            source = "OD"
        else:
            stack = self.scan.transmission
            source = "Transmission"
        viewer = PreMapWindow(self.scan, stack, source)
        self._premap_windows.append(viewer)
        viewer.mapsReady.connect(self._set_premap_maps)
        viewer.destroyed.connect(lambda: self._child_window_closed("premap", viewer))
        viewer.show()
        self._rebuild_window_menu()

    def _remove_premap_window(self, viewer: PreMapWindow) -> None:
        if viewer in self._premap_windows:
            self._premap_windows.remove(viewer)
        self._rebuild_window_menu()

    def _set_premap_maps(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        source = str(payload.get("source", "Transmission"))
        self._premap_maps = {
            f"Net absorption map ({source})": np.asarray(payload["net"]),
            f"Peak map ({source})": np.asarray(payload["peak"]),
        }
        self._premap_config = {
            "pre_edge_range": list(payload.get("pre_edge_range", [])),
            "post_edge_range": list(payload.get("post_edge_range", [])),
            "source": source,
        }
        if "multivariate" in payload:
            self._premap_config["multivariate"] = payload["multivariate"]
        self._analysis_stack = np.asarray(payload["denoised_stack"]) if payload.get("denoised_stack") is not None else None
        if self._analysis_stack is not None:
            self._premap_config["denoised_stack"] = self._analysis_stack.tolist()
        for name in self._premap_maps:
            if not self.layer_list.findItems(name, Qt.MatchFlag.MatchExactly):
                self.layer_list.addItem(name)
        # Pre-edge-subtracted OD is a 3-D (energy, y, x) stack, so it becomes a
        # per-frame grayscale layer (like OD), not a flat pre-map.  The array is
        # kept in memory only; the session stores the pre-edge range and the
        # stack is rebuilt on restore, avoiding a huge JSON payload.
        if payload.get("presub_stack") is not None:
            self._presub_stack = np.asarray(payload["presub_stack"])
            self._presub_layer_name = f"OD - pre-edge ({source})"
            if not self.layer_list.findItems(self._presub_layer_name, Qt.MatchFlag.MatchExactly):
                self.layer_list.addItem(self._presub_layer_name)
        self._persist_scan_session()

    def open_segmentation_window(self) -> None:
        if self.scan is None or self._od_result is None:
            return
        layer_name = self.layer_list.currentItem().text() if self.layer_list.currentItem() else ""
        feature = self._premap_maps.get(layer_name)
        source_label = layer_name if feature is not None else "OD"
        # The energy slider in the window selects a frame from a 3-D stack layer.
        od_stack = self._analysis_stack if self._analysis_stack is not None else self._od_result.optical_density  # type: ignore[attr-defined]
        # Offer a pre-edge-subtracted OD stack alongside raw OD: clustering per
        # chemical composition (the original tool's behaviour) uses OD with the
        # non-resonant baseline removed, not raw OD.  Prefer the stack Pre-map
        # already built (shared with the main "OD - pre-edge" layer); otherwise
        # derive one on the fly from the first few frames as the pre-edge.
        energies = self.scan.energies_eV
        if self._presub_stack is not None:
            presub_stack = self._presub_stack
        else:
            from ..processing.premap import pre_edge_subtract
            n = max(1, min(3, len(energies)))
            pre_range = (float(np.min(energies[:n])), float(np.max(energies[:n])))
            presub_stack = pre_edge_subtract(od_stack, energies, pre_range)
        layer_data = {"OD": od_stack, "OD - pre-edge": presub_stack, **self._premap_maps}
        if feature is None:
            feature = od_stack[self._current_frame]
        saved_groups = {}
        if isinstance(self._segmentation_config, dict):
            saved_groups = self._segmentation_config.get("saved_groups", {}) or {}
        viewer = SegmentationWindow(
            self.scan, feature, od_stack, source_label, layer_data,
            od_frame=self._current_frame, saved_groups=saved_groups,
            saved_defaults=self._segmentation_config,
        )
        self._segmentation_windows.append(viewer)
        viewer.resultReady.connect(self._set_segmentation_result)
        viewer.destroyed.connect(lambda: self._remove_segmentation_window(viewer))
        viewer.show()
        self._rebuild_window_menu()

    def _pre_edge_range(self) -> tuple[float, float]:
        """Pre-edge energy range for OD − pre-edge inputs.  Reuse the Pre-map
        setting when available, else default to the first few frames."""
        energies = self.scan.energies_eV  # type: ignore[union-attr]
        if isinstance(self._premap_config, dict):
            pr = self._premap_config.get("pre_edge_range")
            if isinstance(pr, (list, tuple)) and len(pr) == 2:
                return (float(pr[0]), float(pr[1]))
        n = max(1, min(3, len(energies)))
        return (float(np.min(energies[:n])), float(np.max(energies[:n])))

    def open_fitting_window(self) -> None:
        if self.scan is None or self._od_result is None:
            return
        groups = {}
        if isinstance(self._segmentation_config, dict):
            groups = self._segmentation_config.get("saved_groups", {}) or {}
        viewer = FittingWindow(self.scan.energies_eV, self._analysis_stack if self._analysis_stack is not None else self._od_result.optical_density, groups, pre_edge_range=self._pre_edge_range(), input_source="Low-rank reconstructed OD" if self._analysis_stack is not None else "OD")  # type: ignore[attr-defined]
        self._fitting_windows.append(viewer); viewer.resultReady.connect(self._set_fitting_result); viewer.destroyed.connect(lambda: self._child_window_closed("fitting", viewer)); viewer.show(); self._rebuild_window_menu()

    def _set_fitting_result(self, result: object) -> None:
        from ..report_data import result_to_record
        self._fitting_result = result
        self._fitting_config = result_to_record(result, "fitting")
        name = "PTEE RGBY map"
        if not self.layer_list.findItems(name, Qt.MatchFlag.MatchExactly): self.layer_list.addItem(name)
        items = self.layer_list.findItems(name, Qt.MatchFlag.MatchExactly)
        if items:
            self.layer_list.setCurrentItem(items[0])
            self.layer_list.setCurrentRow(self.layer_list.row(items[0]))
        self._show_frame(self._current_frame)
        self._persist_scan_session()

    def open_multivariate_window(self) -> None:
        if self.scan is None:
            return
        stack = self._analysis_stack
        if stack is None:
            stack = self._od_result.optical_density if self._od_result is not None else self.scan.transmission  # type: ignore[attr-defined]
        viewer = MultivariateWindow(self.scan.energies_eV, stack, input_source="Low-rank reconstructed OD" if self._analysis_stack is not None else ("OD" if self._od_result is not None else "Transmission"))
        self._multivariate_windows.append(viewer)
        viewer.resultReady.connect(self._set_multivariate_result)
        viewer.destroyed.connect(lambda: self._multivariate_windows.remove(viewer) if viewer in self._multivariate_windows else None)
        viewer.show()

    def _report_payload(self) -> dict:
        from ..report_data import build_report_payload
        return build_report_payload(self)

    def open_report_window(self) -> None:
        if self.scan is None:
            self.statusBar().showMessage("Load a scan before creating a report")
            return
        viewer = ReportWindow(self._report_payload(), self)
        self._report_windows.append(viewer)
        viewer.destroyed.connect(lambda: self._report_windows.remove(viewer) if viewer in self._report_windows else None)
        viewer.show()

    def _set_multivariate_result(self, result: object) -> None:
        self._multivariate_result = result
        # A ClusterResult (PCA + k-means) is saved as a distinct cluster-map
        # layer; the PC->RGB composite methods keep the SVD/PCA RGB layer name.
        is_cluster = getattr(result, "n_clusters", None) is not None
        name = "PCA cluster map" if is_cluster else "SVD/PCA RGB map"
        other = "SVD/PCA RGB map" if is_cluster else "PCA cluster map"
        for stale in self.layer_list.findItems(other, Qt.MatchFlag.MatchExactly):
            self.layer_list.takeItem(self.layer_list.row(stale))
        if not self.layer_list.findItems(name, Qt.MatchFlag.MatchExactly):
            self.layer_list.addItem(name)
        item = self.layer_list.findItems(name, Qt.MatchFlag.MatchExactly)[0]
        self.layer_list.setCurrentItem(item)
        self._persist_scan_session()

    def _remove_segmentation_window(self, viewer: SegmentationWindow) -> None:
        if viewer in self._segmentation_windows:
            self._segmentation_windows.remove(viewer)
        if self._closing:
            return
        self._rebuild_window_menu()

    def _set_segmentation_result(self, result: object) -> None:
        self._segmentation_result = result
        layer_name = self.layer_list.currentItem().text() if self.layer_list.currentItem() else "OD"
        previous_groups = {}
        if isinstance(self._segmentation_config, dict):
            previous_groups = dict(self._segmentation_config.get("saved_groups", {}))
        # Do not resurrect the legacy auto-generated placeholder.
        previous_groups.pop("Segmentation 1", None)
        self._segmentation_config = {
            "source": getattr(result, "analysis_parameters", {}).get("source", layer_name if layer_name in self._premap_maps else "OD"),
            "threshold": list(result.threshold),  # type: ignore[attr-defined]
            "minimum_area": int(result.minimum_area),  # type: ignore[attr-defined]
        }
        saved_profiles = getattr(result, "saved_profiles", None)
        saved_areas = getattr(result, "saved_areas", None)
        saved_cluster_ids = getattr(result, "saved_cluster_ids", None)
        if saved_profiles is not None and saved_areas is not None and saved_cluster_ids is not None:
            label = str(getattr(result, "saved_label", None) or "")
            if not label:
                return
            previous_groups[label] = {
                "profiles": np.asarray(saved_profiles).tolist(),
                "areas": np.asarray(saved_areas).astype(int).tolist(),
                "cluster_ids": np.asarray(saved_cluster_ids).astype(int).tolist(),
                "labels": np.asarray(result.labels).astype(int).tolist(),
                "threshold": list(result.threshold),
                "minimum_area": int(result.minimum_area),
                "analysis_parameters": dict(getattr(result, "analysis_parameters", {})),
            }
            self._segmentation_config["saved_groups"] = previous_groups
        self._persist_scan_session()

    def _map_palette_changed(self, index: int) -> None:
        layer = self.layer_list.currentItem().text() if self.layer_list.currentItem() else ""
        if layer not in self._premap_maps:
            return
        palette = ["Jet", "Fire", "Grayscale"][index]
        self._premap_display.setdefault(layer, {"palette": "Jet", "levels": None})["palette"] = palette
        self.image_view.setColorMap(_map_colormap(palette))
        _hide_gradient_ticks(self.image_view)
        self._persist_scan_session()

    def _map_levels_changed(self) -> None:
        if self._updating_map_display:
            return
        layer = self.layer_list.currentItem().text() if self.layer_list.currentItem() else ""
        if layer not in self._premap_maps:
            return
        levels = self.image_view.getLevels()
        if levels is not None:
            self._premap_display.setdefault(layer, {"palette": "Jet", "levels": None})["levels"] = [
                float(value) for value in levels
            ]
            self._persist_scan_session()

    def _reset_map_levels(self) -> None:
        layer = self.layer_list.currentItem().text() if self.layer_list.currentItem() else ""
        image = self._premap_maps.get(layer)
        if image is None:
            return
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return
        levels = [float(finite.min()), float(finite.max())]
        self._updating_map_display = True
        self.image_view.setLevels(*levels)
        self._updating_map_display = False
        self._premap_display.setdefault(layer, {"palette": "Jet", "levels": None})["levels"] = levels
        self._persist_scan_session()

    def _remove_od_window(self, viewer: ODWindow) -> None:
        if viewer in self._od_windows:
            self._od_windows.remove(viewer)
        self._rebuild_window_menu()

    def _set_od_result(self, result: object) -> None:
        self._od_result = result
        if not self.layer_list.findItems("Optical density (OD)", Qt.MatchFlag.MatchExactly):
            self.layer_list.addItem("Optical density (OD)")
        self.layer_list.setEnabled(True)
        od_row = self.layer_list.findItems("Optical density (OD)", Qt.MatchFlag.MatchExactly)[0]
        self.layer_list.setCurrentItem(od_row)
        self.segmentation_action.setEnabled(True)
        self.fitting_action.setEnabled(True)
        self.multivariate_action.setEnabled(True)
        self._persist_scan_session()

    def _set_od_roi(self, bounds: object) -> None:
        if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
            self._od_roi_bounds = [int(v) for v in bounds]
            self._persist_scan_session()

    def _reset_analysis_state(self) -> None:
        for viewer in list(self._report_windows):
            try:
                viewer.close()
            except RuntimeError:
                pass
        self._report_windows.clear()
        for roi in self._spectrum_rois:
            self.image_view.getView().removeItem(roi)
            roi.deleteLater()
        for curve in self._spectrum_curves:
            self.spectrum_plot.removeItem(curve)
        self._od_result = None
        self._od_roi_bounds = None
        self._registered_stack = None
        self._registration_config = None
        self._premap_maps = {}
        self._analysis_stack = None
        self._presub_stack = None
        self._presub_layer_name = None
        self._premap_config = None
        self._premap_display = {}
        self._segmentation_result = None
        self._segmentation_config = None
        self._fitting_result = None
        self._fitting_config = None
        self._multivariate_result = None
        self.layer_list.clear()
        self.layer_list.addItem("Transmission (raw)")
        self.layer_list.setCurrentRow(0)
        self._spectrum_rois.clear()
        self._spectrum_curves.clear()
        self.clear_roi_action.setEnabled(False)

    def _read_session_index(self) -> dict[str, dict]:
        # Analysis sessions are dataset-local sidecars.  Do not merge home or
        # project fallback indexes, which can resurrect stale layer state.
        return {}

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Persist the final dataset state before the main window exits.

        ``_closing`` is set first so the child windows' ``destroyed`` callbacks
        below become no-ops (they must not rebuild menus or re-persist while the
        window is being torn down).  Every step is guarded so a failure during
        shutdown can neither crash the exit nor abort the one persist that has
        already safely written the session.
        """
        self._closing = True
        try:
            self._persist_scan_session()
        except Exception:
            pass
        for windows in (self._image_windows, self._od_windows, self._registration_windows, self._premap_windows, self._segmentation_windows, self._fitting_windows, self._multivariate_windows, self._report_windows):
            for viewer in list(windows):
                try:
                    viewer.close()
                except Exception:
                    pass
        event.accept()
        event.accept()

    def _session_state(self) -> dict:
        current_layer_item = self.layer_list.currentItem()
        state: dict = {
            "scan": str(self.scan.header.path) if self.scan is not None else None,
            "frame": self._current_frame,
            # The active layer is restored by name.  Layers are rebuilt by
            # re-running each analysis, so their positional index is not stable
            # across sessions (a missing upstream step shifts every row); the
            # name identifies the layer regardless of order.  ``layer`` (index)
            # is kept only so older readers still work.
            "layer": self.layer_list.currentRow(),
            "layer_name": current_layer_item.text() if current_layer_item is not None else None,
            "spectrum_rois": [],
            "normalization": {
                "mode": self.normalization_combo.currentIndex() if self.normalization_combo else 0,
                "e1": self.norm_e1.value() if self.norm_e1 else None,
                "e2": self.norm_e2.value() if self.norm_e2 else None,
            },
            "measurement_name": self.measurement_name_edit.text().strip()
            if hasattr(self, "measurement_name_edit") else "",
        }
        if self.scan is not None and self.scan.transmission is not None:
            h, w = self.scan.transmission.shape[1:]
            state["spectrum_rois"] = [
                [x0, y0, x1 - x0, y1 - y0]
                for x0, y0, x1, y1 in (self._roi_bounds(r, w, h) for r in self._spectrum_rois)
            ]
        if self._od_result is not None:
            state["od"] = {
                "i0": np.asarray(self._od_result.i0).tolist(),  # type: ignore[attr-defined]
                "roi": self._od_roi_bounds,
            }
        if self._registration_config is not None:
            state["registration"] = self._registration_config
        if self._premap_config is not None:
            state["premap"] = self._premap_config
            state["premap_display"] = self._premap_display
        if self._segmentation_config is not None:
            state["segmentation"] = self._segmentation_config
        if getattr(self, "_fitting_config", None) is not None:
            state["fitting"] = self._fitting_config
        if self._multivariate_result is not None:
            from ..report_data import result_to_record
            state["multivariate"] = result_to_record(self._multivariate_result, "multivariate")
        return state

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        """Write ``text`` to ``path`` so an interrupted write cannot corrupt an
        existing good file.  The content is written to a sibling temp file and
        then atomically renamed over the target; if the process is killed
        mid-write (e.g. a crash while quitting) only the discarded temp file is
        affected and the previous file stays intact.  This is what keeps the
        recent-scan list and per-scan ROI/analysis state from being lost when
        the app dies during shutdown.
        """
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def _persist_scan_session(self) -> None:
        if self.scan is None or self._restoring_session:
            return
        payload = json.dumps(self._session_state(), ensure_ascii=False, indent=2)
        # Preserve analysis sections already present in the sidecar if a
        # transient UI callback fires while restoration is still rebuilding
        # them.  This prevents a partially restored startup state from
        # erasing OD/maps/profiles.
        try:
            sidecar = self.scan.header.path.with_suffix(".muaxis.json")
            previous = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(previous, dict):
                current = json.loads(payload)
                for key in ("od", "registration", "premap", "premap_display", "segmentation", "fitting"):
                    if key not in current and key in previous:
                        current[key] = previous[key]
                payload = json.dumps(current, ensure_ascii=False, indent=2)
        except (OSError, ValueError, TypeError):
            pass
        # Analysis state belongs to the dataset, not to the user's home
        # directory.  A sidecar next to the HDR travels with the measurement
        # and is loaded preferentially when the scan is reopened.
        try:
            self._atomic_write_text(self.scan.header.path.with_suffix(".muaxis.json"), payload)
        except OSError:
            self.statusBar().showMessage("Session could not be saved next to the HDR")

    def save_session_dialog(self) -> None:
        if self.scan is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save muNEXAFS session", str(self.scan.header.path.with_suffix(".muaxis.json")),
            "muNEXAFS session (*.muaxis.json);;JSON (*.json)",
        )
        if not path:
            return
        try:
            self._atomic_write_text(Path(path), json.dumps(self._session_state(), ensure_ascii=False, indent=2))
            self._session_path = Path(path)
            self.statusBar().showMessage(f"Session saved: {path}")
        except OSError as exc:
            QMessageBox.critical(self, "Cannot save session", str(exc))

    def _restore_scan_session(self, path: Path) -> None:
        state = self._session_index.get(str(path))
        try:
            sidecar = path.with_suffix(".muaxis.json")
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))
            # The sidecar is written on every change and is therefore newer
            # than a home-directory index entry when both are present.
            if isinstance(loaded, dict):
                state = loaded
        except (OSError, ValueError):
            if not isinstance(state, dict):
                state = None
        if not isinstance(state, dict) or self.scan is None or self.scan.transmission is None:
            return
        norm = state.get("normalization")
        if isinstance(norm, dict) and self.normalization_combo is not None:
            mode = norm.get("mode")
            if isinstance(mode, int) and 0 <= mode < self.normalization_combo.count():
                self.normalization_combo.setCurrentIndex(mode)
            for key, spin in (("e1", self.norm_e1), ("e2", self.norm_e2)):
                value = norm.get(key)
                if spin is not None and isinstance(value, (int, float)):
                    spin.setValue(float(value))
        for bounds in state.get("spectrum_rois", []):
            if isinstance(bounds, list) and len(bounds) == 4:
                self.create_spectrum_roi(tuple(int(v) for v in bounds))
        od_state = state.get("od")
        if isinstance(od_state, dict) and isinstance(od_state.get("i0"), list):
            from ..processing.absorbance import compute_optical_density
            i0 = np.asarray(od_state["i0"], dtype=float)
            if i0.shape == (len(self.scan.frame_paths),):
                roi = od_state.get("roi")
                if isinstance(roi, list) and len(roi) == 4:
                    self._od_roi_bounds = [int(v) for v in roi]
                self._set_od_result(compute_optical_density(self.scan.transmission, i0))
        registration = state.get("registration")
        if isinstance(registration, dict) and self.scan.transmission is not None:
            try:
                from ..processing.registration import register_translation_stack
                source = self._od_result.optical_density if self._od_result is not None else self.scan.transmission
                result = register_translation_stack(
                    source,
                    reference_index=int(registration.get("reference_index", 0)),
                    upsample_factor=int(registration.get("upsample_factor", 10)),
                    reference_mode=str(registration.get("reference_mode", "fixed")),
                )
                self._set_registration_result(result)
            except (TypeError, ValueError):
                pass
        premap = state.get("premap")
        if isinstance(premap, dict) and self.scan.transmission is not None:
            try:
                from ..processing.premap import net_absorption_map, peak_map, pre_edge_subtract
                source = self._od_result.optical_density if self._od_result is not None else self.scan.transmission
                if isinstance(premap.get("denoised_stack"), list):
                    source = np.asarray(premap["denoised_stack"], dtype=float)
                source_label = str(premap.get("source", "Transmission"))
                pre_edge_range = tuple(float(v) for v in premap["pre_edge_range"])
                post_edge_range = tuple(float(v) for v in premap["post_edge_range"])
                self._set_premap_maps({
                    "net": net_absorption_map(source, self.scan.energies_eV, pre_edge_range, post_edge_range),
                    "peak": peak_map(source, self.scan.energies_eV, post_edge_range),
                    "presub_stack": pre_edge_subtract(source, self.scan.energies_eV, pre_edge_range),
                    "pre_edge_range": list(pre_edge_range), "post_edge_range": list(post_edge_range),
                    "source": source_label,
                    "denoised_stack": source if isinstance(premap.get("denoised_stack"), list) else None,
                    **({"multivariate": premap["multivariate"]} if "multivariate" in premap else {}),
                })
            except (KeyError, TypeError, ValueError):
                pass
        saved_display = state.get("premap_display")
        if isinstance(saved_display, dict):
            self._premap_display = saved_display
        segmentation = state.get("segmentation")
        if isinstance(segmentation, dict) and self._od_result is not None:
            # Saved profiles are authoritative. Re-clustering frame zero here
            # used to invent a label image unrelated to the original run.
            self._segmentation_config = dict(segmentation)
            groups = {str(k): v for k, v in (segmentation.get("saved_groups") or {}).items()
                      if str(k) != "Segmentation 1" and isinstance(v, dict)}
            self._segmentation_config["saved_groups"] = groups
            if groups:
                from types import SimpleNamespace
                label, group = next(reversed(groups.items()))
                self._segmentation_result = SimpleNamespace(
                    labels=np.asarray(group["labels"], dtype=np.int32) if "labels" in group else None,
                    saved_label=label, saved_profiles=np.asarray(group.get("profiles", []), dtype=float),
                    saved_areas=np.asarray(group.get("areas", []), dtype=np.int64),
                    saved_cluster_ids=np.asarray(group.get("cluster_ids", []), dtype=np.int32),
                    analysis_parameters=group.get("analysis_parameters", {}),
                )
        fitting = state.get("fitting")
        if isinstance(fitting, dict) and isinstance(fitting.get("rgb_y"), list):
            try:
                from ..report_data import result_from_record
                self._set_fitting_result(result_from_record(fitting, "fitting"))
            except (TypeError, ValueError):
                pass
        multivariate = state.get("multivariate")
        if isinstance(multivariate, dict) and isinstance(multivariate.get("rgb_map"), list):
            try:
                from ..report_data import result_from_record
                self._set_multivariate_result(result_from_record(multivariate, "multivariate"))
            except (TypeError, ValueError):
                pass
        frame = state.get("frame", 0)
        if isinstance(frame, int) and 0 <= frame < len(self.scan.frame_paths):
            self._current_frame = frame
            self._update_frame_readout(frame)
        # Prefer the layer name: it survives a reordered or partially restored
        # layer list.  Fall back to the positional index only for sidecars
        # written before the name was recorded.
        layer_name = state.get("layer_name")
        matched = (
            self.layer_list.findItems(layer_name, Qt.MatchFlag.MatchExactly)
            if isinstance(layer_name, str) and layer_name
            else []
        )
        if matched:
            self.layer_list.setCurrentRow(self.layer_list.row(matched[0]))
        else:
            layer = state.get("layer", 0)
            if isinstance(layer, int) and 0 <= layer < self.layer_list.count():
                self.layer_list.setCurrentRow(layer)

    def _rebuild_window_menu(self) -> None:
        try:
            self.window_menu.clear()
        except RuntimeError:
            # Child windows may emit ``destroyed`` while the main window is
            # already being torn down.
            return
        try:
            self.window_menu.addAction(self.new_image_action)
        except RuntimeError:
            return
        self.window_menu.addSeparator()
        for index, viewer in enumerate(self._image_windows, 1):
            action = QAction(f"Image window {index}", self)
            action.triggered.connect(viewer.raise_)
            action.triggered.connect(viewer.activateWindow)
            self.window_menu.addAction(action)
        for index, viewer in enumerate(self._od_windows, 1):
            action = QAction(f"OD window {index}", self)
            action.triggered.connect(viewer.raise_)
            action.triggered.connect(viewer.activateWindow)
            self.window_menu.addAction(action)
        for index, viewer in enumerate(self._registration_windows, 1):
            action = QAction(f"Registration window {index}", self)
            action.triggered.connect(viewer.raise_)
            action.triggered.connect(viewer.activateWindow)
            self.window_menu.addAction(action)
        for index, viewer in enumerate(self._premap_windows, 1):
            action = QAction(f"Pre-map window {index}", self)
            action.triggered.connect(viewer.raise_)
            action.triggered.connect(viewer.activateWindow)
            self.window_menu.addAction(action)
        for index, viewer in enumerate(self._segmentation_windows, 1):
            action = QAction(f"Segmentation window {index}", self)
            action.triggered.connect(viewer.raise_)
            action.triggered.connect(viewer.activateWindow)
            self.window_menu.addAction(action)

    def _load_recent_scans(self) -> None:
        try:
            paths = json.loads(self._recent_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            paths = []
        if not isinstance(paths, list):
            paths = []
        for value in paths:
            saved_name = ""
            if isinstance(value, dict):
                saved_name = str(value.get("name", "") or "")
                value = value.get("path")
            path = Path(value) if isinstance(value, str) else Path()
            if path.suffix.lower() == ".hdr":
                self._insert_recent_item(path, saved_name)
        self._sort_recent_scans()

    def _save_recent_scans(self) -> None:
        paths = [
            {
                "path": self.recent_list.item(i).data(Qt.ItemDataRole.UserRole),
                "name": self.recent_list.item(i).data(Qt.ItemDataRole.UserRole + 1) or "",
            }
            for i in range(self.recent_list.count())
        ]
        try:
            self._atomic_write_text(self._recent_path, json.dumps(paths, ensure_ascii=False, indent=2))
        except OSError:
            self.statusBar().showMessage(f"Could not save recent scan list: {self._recent_path}")

    def _insert_recent_item(self, path: Path, saved_name: str | None = None) -> None:
        for index in range(self.recent_list.count() - 1, -1, -1):
            item = self.recent_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == str(path):
                self.recent_list.takeItem(index)
        from PySide6.QtWidgets import QListWidgetItem

        name = saved_name or ""
        if not name:
            try:
                state = json.loads(path.with_suffix(".muaxis.json").read_text(encoding="utf-8"))
                name = str(state.get("measurement_name", "") or "") if isinstance(state, dict) else ""
            except (OSError, ValueError):
                state = self._session_index.get(str(path), {})
                name = state.get("measurement_name", "") if isinstance(state, dict) else ""
        item = QListWidgetItem()
        self.recent_list.insertItem(0, item)
        item.setData(Qt.ItemDataRole.UserRole, str(path))
        item.setData(Qt.ItemDataRole.UserRole + 1, name)
        item.setToolTip(str(path))
        self._set_recent_item_text(item)
        while self.recent_list.count() > 20:
            self.recent_list.takeItem(self.recent_list.count() - 1)

    def _add_recent_scan(self, path: Path) -> None:
        self._insert_recent_item(path)
        # Opening an existing scan must not turn the list into a recency list.
        # Reapply the user-selected filename/name order, then restore the
        # current-item highlight to the scan that was opened.
        self._sort_recent_scans()
        for index in range(self.recent_list.count()):
            item = self.recent_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == str(path):
                self.recent_list.setCurrentItem(item)
                break
        self._save_recent_scans()

    def _set_recent_item_text(self, item: object) -> None:
        path = Path(item.data(Qt.ItemDataRole.UserRole))  # type: ignore[attr-defined]
        name = item.data(Qt.ItemDataRole.UserRole + 1) or ""  # type: ignore[attr-defined]
        item.setText(f"{path.name} — {name}" if name else path.name)  # type: ignore[attr-defined]

    def _sort_recent_scans(self) -> None:
        items = [self.recent_list.takeItem(0) for _ in range(self.recent_list.count())]
        mode = self.recent_sort_combo.currentIndex()
        by_name = mode in {2, 3}
        reverse = mode in {1, 3}
        items.sort(key=lambda item: str(item.data(Qt.ItemDataRole.UserRole + 1) or "").casefold()
                   if by_name else Path(item.data(Qt.ItemDataRole.UserRole)).name.casefold(), reverse=reverse)
        for item in items:
            self.recent_list.addItem(item)

    def _load_measurement_name(self, path: Path) -> None:
        try:
            state = json.loads(path.with_suffix(".muaxis.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = self._session_index.get(str(path), {})
        name = state.get("measurement_name", "") if isinstance(state, dict) else ""
        self.measurement_name_edit.setText(str(name))

    def _save_measurement_name(self) -> None:
        if self.scan is None:
            return
        self._persist_scan_session()
        path = str(self.scan.header.path)
        for index in range(self.recent_list.count()):
            item = self.recent_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                item.setData(Qt.ItemDataRole.UserRole + 1, self.measurement_name_edit.text().strip())
                self._set_recent_item_text(item)
                break
        self._save_recent_scans()
        self._sort_recent_scans()

    def _recent_item_activated(self, item: object) -> None:
        path = Path(item.data(Qt.ItemDataRole.UserRole))  # type: ignore[attr-defined]
        if path.is_file():
            self.load_scan(path)
        else:
            self.statusBar().showMessage(f"File not found: {path}")


def run_gui(initial_path: str | Path | None = None) -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("muNEXAFS")
    app.setApplicationDisplayName("muNEXAFS")
    window = MainWindow(initial_path)
    window.show()
    return app.exec()


def _jet_colormap() -> pg.ColorMap:
    """Return a dependency-free MATLAB-style jet palette."""
    positions = np.array([0.0, 0.125, 0.375, 0.625, 0.875, 1.0])
    colors = np.array([
        [0, 0, 128, 255], [0, 0, 255, 255], [0, 255, 255, 255],
        [255, 255, 0, 255], [255, 0, 0, 255], [128, 0, 0, 255],
    ], dtype=np.ubyte)
    return pg.ColorMap(positions, colors)


def _gray_colormap() -> pg.ColorMap:
    positions = np.array([0.0, 1.0])
    colors = np.array([[0, 0, 0, 255], [255, 255, 255, 255]], dtype=np.ubyte)
    return pg.ColorMap(positions, colors)


def _fire_colormap() -> pg.ColorMap:
    # ImageJ Fire-like black → red → orange → yellow → white ramp.
    positions = np.array([0.0, 0.30, 0.55, 0.78, 1.0])
    colors = np.array([
        [0, 0, 0, 255], [150, 0, 0, 255], [255, 60, 0, 255],
        [255, 220, 0, 255], [255, 255, 255, 255],
    ], dtype=np.ubyte)
    return pg.ColorMap(positions, colors)


def _map_colormap(name: str) -> pg.ColorMap:
    return {"Jet": _jet_colormap(), "Fire": _fire_colormap(), "Grayscale": _gray_colormap()}.get(
        name, _jet_colormap()
    )


def _hide_gradient_ticks(image_view: pg.ImageView) -> None:
    """Hide editable gradient-stop triangles while keeping level bars."""
    for tick in image_view.ui.histogram.item.gradient.ticks:
        tick.hide()
