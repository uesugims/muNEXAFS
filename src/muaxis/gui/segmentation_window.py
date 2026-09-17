"""Segmentation and cluster-profile window."""

from __future__ import annotations
from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QSlider, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from ..io.stxm import ScanStack
from ..processing.segmentation import SegmentationResult, segment_clusters


class _NumericItem(QTableWidgetItem):
    def __lt__(self, other: QTableWidgetItem) -> bool:
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return super().__lt__(other)


class SegmentationWindow(QMainWindow):
    resultReady = Signal(object)

    def __init__(self, scan: ScanStack, feature_map: np.ndarray, od_stack: np.ndarray,
                 source_label: str, layer_data: dict[str, np.ndarray] | None = None,
                 od_frame: int = 0, saved_groups: dict[str, dict] | None = None,
                 saved_defaults: dict | None = None) -> None:
        super().__init__()
        self.scan, self.od_stack = scan, od_stack
        self.layer_data = layer_data or {source_label: feature_map}
        self.saved_groups = dict(saved_groups or {})
        self.saved_defaults = dict(saved_defaults or {})
        self.source_label = source_label
        self.result: SegmentationResult | None = None
        self._profiles_by_id: dict[int, np.ndarray] = {}
        self._areas_by_id: dict[int, int] = {}
        self._reviewing_saved = False
        self.setWindowTitle(f"muNEXAFS — Segmentation ({source_label})"); self.resize(1100, 760)
        central = QWidget(self); root = QVBoxLayout(central)

        review = QHBoxLayout(); review.addWidget(QLabel("Saved label"))
        self.saved_label_combo = QComboBox(); self.saved_label_combo.addItem("New segmentation", None)
        for label in self.saved_groups:
            self.saved_label_combo.addItem(label, label)
        review.addWidget(self.saved_label_combo, 1); root.addLayout(review)

        top = QHBoxLayout(); top.addWidget(QLabel("Input layer")); self.layer_combo = QComboBox(); self.layer_combo.addItems(list(self.layer_data) + ["Cluster labels"]); self.layer_combo.setCurrentText(source_label); self.layer_combo.currentTextChanged.connect(self._layer_changed); top.addWidget(self.layer_combo)
        # For the "OD" layer the clustering input is a single energy frame; a
        # slider selects which one (the original tool worked the same way).
        top.addWidget(QLabel("OD energy")); self.od_slider = QSlider(Qt.Orientation.Horizontal); self.od_slider.setRange(0, self.od_stack.shape[0] - 1); self.od_slider.setValue(int(np.clip(od_frame, 0, self.od_stack.shape[0] - 1))); self.od_slider.valueChanged.connect(self._od_frame_changed); top.addWidget(self.od_slider, 1); self.od_energy_label = QLabel(); top.addWidget(self.od_energy_label)
        self.feature_map = self._feature_for_layer(source_label)
        # Thresholds are set by dragging the histogram (below) or by the two
        # value boxes next to "Log Y", not by sliders.  State is held here.
        self._low_thr, self._high_thr = 0.0, 1.0
        self._updating_region, self._updating_edits = False, False
        top.addStretch(1); self.run_button = QPushButton("Run"); self.run_button.clicked.connect(self.run_segmentation); top.addWidget(self.run_button); self.reset_button = QPushButton("Reset"); self.reset_button.clicked.connect(self._reset_for_new); top.addWidget(self.reset_button); root.addLayout(top)
        self.low_label = QLabel(); self.high_label = QLabel(); self.area_label = QLabel(); root.addWidget(self.low_label); root.addWidget(self.high_label); root.addWidget(self.area_label)

        display_row = QHBoxLayout()
        self.input_view = pg.ImageView(view=pg.PlotItem()); self.input_view.ui.roiBtn.hide(); self.input_view.ui.menuBtn.hide(); self.input_view.ui.histogram.hide(); display_row.addWidget(self.input_view, 3)
        histogram_panel = QVBoxLayout()
        histogram_options = QHBoxLayout()
        histogram_options.addWidget(QLabel("Lower")); self.low_edit = QLineEdit(); self.low_edit.setMaximumWidth(96); self.low_edit.setToolTip("Lower threshold value (type and press Enter)"); histogram_options.addWidget(self.low_edit)
        histogram_options.addWidget(QLabel("Upper")); self.high_edit = QLineEdit(); self.high_edit.setMaximumWidth(96); self.high_edit.setToolTip("Upper threshold value (type and press Enter)"); histogram_options.addWidget(self.high_edit)
        histogram_options.addStretch()
        self.histogram_log_y = QCheckBox("Log Y")
        self.histogram_log_y.setToolTip("Display the histogram pixel-count axis on a logarithmic scale")
        self.histogram_log_y.toggled.connect(self._set_histogram_log_scale)
        histogram_options.addWidget(self.histogram_log_y)
        histogram_panel.addLayout(histogram_options)
        self.histogram = pg.PlotWidget(title="Input histogram (drag to set thresholds)"); self.histogram.setLabel("bottom", "Value"); self.histogram.setLabel("left", "Pixels"); histogram_panel.addWidget(self.histogram, 1); display_row.addLayout(histogram_panel, 2); root.addLayout(display_row, 3)
        # Dragging on the histogram moves whichever threshold is nearer the
        # cursor; the value boxes stay in sync.
        self.histogram.scene().sigMouseClicked.connect(self._histogram_clicked)
        self.histogram.scene().sigMouseMoved.connect(self._histogram_moved)
        self.low_edit.editingFinished.connect(self._edits_changed)
        self.high_edit.editingFinished.connect(self._edits_changed)
        self._overlay = pg.ImageItem(); self._overlay.setZValue(10); self.input_view.getView().addItem(self._overlay)

        bottom_row = QHBoxLayout()
        self.profile_plot = pg.PlotWidget(title="Cluster mean OD profiles"); self.profile_plot.setLabel("bottom", "Energy", units="eV"); self.profile_plot.setLabel("left", "Mean OD"); bottom_row.addWidget(self.profile_plot, 3)
        self.table = QTableWidget(0, 3); self.table.setHorizontalHeaderLabels(["Cluster", "Area", "Good"]); self.table.setColumnWidth(0, 52); self.table.setColumnWidth(1, 62); self.table.setColumnWidth(2, 42); self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows); self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection); self.table.setSortingEnabled(True); self.table.itemSelectionChanged.connect(self._plot_selected); self.table.itemChanged.connect(self._table_item_changed); bottom_row.addWidget(self.table, 1); root.addLayout(bottom_row, 2)

        controls = QHBoxLayout(); controls.addWidget(QLabel("Minimum size")); self.min_area_spin = QSpinBox(); self.min_area_spin.setRange(1, self.feature_map.size); self.min_area_spin.setValue(10); self.min_area_spin.valueChanged.connect(self._minimum_size_changed); controls.addWidget(self.min_area_spin); controls.addWidget(QLabel("Profile normalization")); self.norm_combo = QComboBox(); self.norm_combo.addItems(["Min–max", "None", "Two energy values"]); self.norm_combo.currentIndexChanged.connect(self._normalization_changed); controls.addWidget(self.norm_combo); controls.addWidget(QLabel("E1")); self.norm_e1 = QDoubleSpinBox(); controls.addWidget(self.norm_e1); controls.addWidget(QLabel("E2")); self.norm_e2 = QDoubleSpinBox(); controls.addWidget(self.norm_e2); self.exclude_bad = QCheckBox("Exclude bad data"); self.exclude_bad.toggled.connect(self._quality_filter_changed); controls.addWidget(self.exclude_bad); controls.addWidget(QLabel("Label")); self.label_edit = QLineEdit(); self.label_edit.setPlaceholderText("segmentation label"); self.label_edit.setMaximumWidth(180); controls.addWidget(self.label_edit); self.save_button = QPushButton("Save"); self.save_button.setEnabled(False); self.save_button.clicked.connect(self.save_result); controls.addWidget(self.save_button); root.addLayout(controls)
        self.status = QLabel("Threshold preview; profiles use OD"); root.addWidget(self.status)
        self.setCentralWidget(central)
        self.norm_e1.valueChanged.connect(self._plot_selected); self.norm_e2.valueChanged.connect(self._plot_selected)
        self._set_feature_map(self.feature_map)
        self._update_od_slider_visibility(source_label)
        lo, hi = float(self.scan.energies_eV.min()), float(self.scan.energies_eV.max())
        for spin, value in ((self.norm_e1, lo), (self.norm_e2, hi)):
            spin.setRange(lo, hi); spin.setValue(value)
        self._normalization_changed(self.norm_combo.currentIndex())
        self.saved_label_combo.currentIndexChanged.connect(self._saved_label_changed)

    def _is_stack_layer(self, name: str) -> bool:
        """A layer whose value is a 3-D (energy, y, x) stack, so the energy
        slider picks the frame.  ``OD`` and ``OD - pre-edge`` are such layers;
        pre-map layers are fixed 2-D maps."""
        return name in self.layer_data and np.asarray(self.layer_data[name]).ndim == 3

    def _feature_for_layer(self, name: str) -> np.ndarray:
        """Input map for a layer name.  For a stack layer the energy slider
        selects the frame; a 2-D map is returned as-is."""
        if self._is_stack_layer(name):
            return np.asarray(self.layer_data[name])[self.od_slider.value()]
        return self.layer_data[name]

    def _update_od_slider_visibility(self, name: str) -> None:
        is_stack = self._is_stack_layer(name)
        self.od_slider.setVisible(is_stack); self.od_energy_label.setVisible(is_stack)
        if is_stack:
            self.od_energy_label.setText(f"{self.scan.energies_eV[self.od_slider.value()]:.4g} eV")

    def _od_frame_changed(self) -> None:
        name = self.layer_combo.currentText()
        if not self._is_stack_layer(name):
            return
        self.od_energy_label.setText(f"{self.scan.energies_eV[self.od_slider.value()]:.4g} eV")
        # Changing the frame changes the clustering input, so drop the stale
        # result and preview the newly selected energy.
        self._clear_result_view()
        self._set_feature_map(self._feature_for_layer(name))

    def _minimum_size_changed(self) -> None:
        if self.result is not None:
            count = sum(area >= self.min_area_spin.value() for area in self._areas_by_id.values())
            # The table is sortable, so its visual row is not necessarily the
            # result-array index.  Resolve the cluster id from column 0 before
            # hiding rows; otherwise changing the sort order hides the wrong
            # clusters and also makes the profile selection appear incorrect.
            minimum = self.min_area_spin.value()
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 0)
                cluster_id = int(item.text()) if item is not None else -1
                area = self._areas_by_id.get(cluster_id, 0)
                self.table.setRowHidden(row, int(area) < minimum)
            self.status.setText(f"{count} clusters above minimum size")
            self._plot_selected()

    def _table_item_changed(self, item: QTableWidgetItem) -> None:
        # Quality checks affect visibility and plotting.  Other cell updates
        # do not need to trigger an expensive full redraw during Run.
        if item.column() == 2:
            self._quality_filter_changed()

    def _quality_filter_changed(self) -> None:
        if self.result is not None:
            for row in range(self.table.rowCount()):
                good = self.table.item(row, 2)
                cluster = self.table.item(row, 0)
                cluster_id = int(cluster.text()) if cluster is not None else -1
                too_small = self._areas_by_id.get(cluster_id, 0) < self.min_area_spin.value()
                bad = self.exclude_bad.isChecked() and (good is None or good.checkState() != Qt.CheckState.Checked)
                hidden = too_small or bad
                self.table.setRowHidden(row, hidden)
        self._plot_selected()

    def _set_feature_map(self, image: np.ndarray) -> None:
        self.feature_map = image; finite = image[np.isfinite(image)]; self.data_low = float(np.nanmin(finite)) if finite.size else 0.0; self.data_high = float(np.nanmax(finite)) if finite.size else 1.0
        # A new feature map starts with the thresholds spanning its full range.
        self._low_thr, self._high_thr = self.data_low, self.data_high
        self.input_view.setImage(image, autoLevels=True, autoRange=True); self._draw_histogram(); self._threshold_changed()

    def _normalization_changed(self, index: int) -> None:
        enabled = index == 2
        self.norm_e1.setEnabled(enabled)
        self.norm_e2.setEnabled(enabled)
        self._plot_selected()

    def _select_new_mode(self) -> None:
        self.saved_label_combo.blockSignals(True)
        self.saved_label_combo.setCurrentIndex(0)
        self.saved_label_combo.blockSignals(False)
        self._reviewing_saved = False

    def _reset_for_new(self) -> None:
        self._select_new_mode()
        self.label_edit.clear()
        self._clear_result_view()
        self.status.setText("Threshold preview; profiles use OD")

    def _layer_changed(self, name: str) -> None:
        if name == "Cluster labels" and self.result is not None:
            self._update_od_slider_visibility(name)
            self._overlay.clear(); self.input_view.setImage(self._cluster_rgba(), autoLevels=False, autoRange=True); self.histogram.clear(); return
        if name in self.layer_data:
            self._select_new_mode()
            self.source_label = name
            # A new feature layer starts a fresh segmentation session.  Do
            # not carry over the previous table's hidden rows, selections, or
            # slider positions, which can otherwise produce an empty second
            # Run even though the new layer contains valid pixels.
            self._clear_result_view()
            self._update_od_slider_visibility(name)
            # ``_set_feature_map`` resets the thresholds to the new layer's full
            # value range.
            self._set_feature_map(self._feature_for_layer(name))

    def _saved_label_changed(self, index: int) -> None:
        label = self.saved_label_combo.itemData(index)
        if label is None:
            if self._reviewing_saved:
                self._reset_for_new()
            return
        group = self.saved_groups.get(str(label))
        if isinstance(group, dict):
            self._load_saved_group(str(label), group)

    def _set_threshold_sliders(self, threshold: tuple[float, float]) -> None:
        """Set both thresholds to explicit values (used when restoring a saved
        segmentation).  Named for backward compatibility; there are no sliders."""
        low, high = sorted(map(float, threshold))
        self._low_thr, self._high_thr = low, high
        self._threshold_changed()

    def _populate_table(self, cluster_ids: list[int], areas: list[int], *, saved: bool) -> None:
        self.table.setSortingEnabled(False)
        self.table.blockSignals(True)
        try:
            self.table.clearContents()
            self.table.setRowCount(len(cluster_ids))
            for row, (cluster_id, area) in enumerate(zip(cluster_ids, areas, strict=True)):
                self.table.setItem(row, 0, _NumericItem(str(cluster_id)))
                self.table.setItem(row, 1, _NumericItem(str(int(area))))
                good = QTableWidgetItem()
                good.setFlags(good.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                good.setCheckState(Qt.CheckState.Checked if saved else Qt.CheckState.Unchecked)
                self.table.setItem(row, 2, good)
                self.table.setRowHidden(row, False)
            self.table.clearSelection()
        finally:
            self.table.blockSignals(False)
        self.table.setSortingEnabled(True)

    def _load_saved_group(self, label: str, group: dict) -> None:
        """Restore a saved segmentation as an interactive, read-only review."""
        parameters = dict(group.get("analysis_parameters") or {})
        source = str(parameters.get("source") or group.get("source")
                     or self.saved_defaults.get("source") or self.source_label)
        if source not in self.layer_data:
            source = self.source_label if self.source_label in self.layer_data else next(iter(self.layer_data))
        frame = parameters.get("frame_index")
        if isinstance(frame, (int, float)) and self._is_stack_layer(source):
            self.od_slider.blockSignals(True)
            self.od_slider.setValue(int(np.clip(frame, 0, self.od_stack.shape[0] - 1)))
            self.od_slider.blockSignals(False)

        self._clear_result_view()
        self._reviewing_saved = True
        self.source_label = source
        self.layer_combo.blockSignals(True); self.layer_combo.setCurrentText(source); self.layer_combo.blockSignals(False)
        self._update_od_slider_visibility(source)
        self._set_feature_map(self._feature_for_layer(source))

        threshold_value = group.get("threshold", parameters.get(
            "threshold", self.saved_defaults.get("threshold", self._thresholds())))
        try:
            threshold = tuple(sorted((float(threshold_value[0]), float(threshold_value[1]))))
        except (TypeError, ValueError, IndexError):
            threshold = self._thresholds()
        self._set_threshold_sliders(threshold)
        minimum = int(group.get("minimum_area", parameters.get(
            "minimum_area_at_save_pixels", parameters.get(
                "minimum_area_pixels", self.saved_defaults.get("minimum_area", 1)))))
        self.min_area_spin.blockSignals(True)
        self.min_area_spin.setValue(max(1, minimum))
        self.min_area_spin.blockSignals(False)

        profiles = np.asarray(group.get("profiles", []), dtype=np.float64)
        if profiles.ndim == 1 and profiles.size:
            profiles = profiles[None, :]
        ids = np.asarray(group.get("cluster_ids", []), dtype=np.int32).reshape(-1)
        areas = np.asarray(group.get("areas", []), dtype=np.int64).reshape(-1)
        count = min(len(profiles) if profiles.ndim == 2 else 0, len(ids), len(areas))
        profiles, ids, areas = profiles[:count], ids[:count], areas[:count]
        valid = np.ones(count, dtype=bool)
        if count:
            valid = np.asarray([profile.size == len(self.scan.energies_eV) for profile in profiles])
        profiles, ids, areas = profiles[valid], ids[valid], areas[valid]

        labels = np.asarray(group.get("labels", []), dtype=np.int32)
        if labels.shape != self.feature_map.shape:
            labels = np.zeros(self.feature_map.shape, dtype=np.int32)
        self._profiles_by_id = {int(cluster_id): profile.copy()
                                for cluster_id, profile in zip(ids, profiles, strict=True)}
        self._areas_by_id = {int(cluster_id): int(area)
                             for cluster_id, area in zip(ids, areas, strict=True)}
        self.result = SegmentationResult(
            labels, areas.copy(), profiles.copy(), threshold, max(1, minimum),
            profiles.copy(), areas.copy(), ids.copy(), label, parameters,
        )
        self.exclude_bad.blockSignals(True); self.exclude_bad.setChecked(False); self.exclude_bad.blockSignals(False)
        self._populate_table(ids.astype(int).tolist(), areas.astype(int).tolist(), saved=True)
        self.label_edit.setText(label)
        self.save_button.setEnabled(False)
        self._show_cluster_overlay()
        self._plot_selected()
        suffix = "" if np.any(labels) else " — profiles only (label image was not saved)"
        self.status.setText(f"Reviewing saved segmentation: {label} ({len(ids)} profiles){suffix}")

    def _thresholds(self) -> tuple[float, float]:
        lo = min(self._low_thr, self._high_thr); hi = max(self._low_thr, self._high_thr)
        return float(np.clip(lo, self.data_low, self.data_high)), float(np.clip(hi, self.data_low, self.data_high))

    def _draw_histogram(self) -> None:
        self.histogram.clear(); values = self.feature_map[np.isfinite(self.feature_map)]
        if values.size:
            counts, edges = np.histogram(values, bins=128); self.histogram.plot((edges[:-1] + edges[1:]) / 2, counts, fillLevel=0, brush=(80, 140, 220, 80), pen=pg.mkPen("#5b8ff9"))
        self.hist_region = pg.LinearRegionItem(self._thresholds(), movable=False, brush=pg.mkBrush(255, 80, 80, 70)); self.histogram.addItem(self.hist_region)

    def _histogram_clicked(self, event: object) -> None:
        try:
            self._move_nearer_threshold(event.scenePos())
        except AttributeError:
            pass

    def _histogram_moved(self, scene_pos: object) -> None:
        if QApplication.mouseButtons() & Qt.MouseButton.LeftButton:
            self._move_nearer_threshold(scene_pos)

    def _move_nearer_threshold(self, scene_pos: object) -> None:
        """Move whichever threshold is nearer the cursor to the cursor value."""
        plot = self.histogram.getPlotItem()
        if not plot.sceneBoundingRect().contains(scene_pos):
            return
        value = float(np.clip(plot.vb.mapSceneToView(scene_pos).x(), self.data_low, self.data_high))
        low, high = self._thresholds()
        if abs(value - low) <= abs(value - high):
            self._low_thr = value
        else:
            self._high_thr = value
        self._threshold_changed()

    def _edits_changed(self) -> None:
        if self._updating_edits:
            return
        try:
            low, high = float(self.low_edit.text()), float(self.high_edit.text())
        except (TypeError, ValueError):
            self._threshold_changed()  # revert boxes to the current values
            return
        self._low_thr, self._high_thr = low, high
        self._threshold_changed()

    def _set_histogram_log_scale(self, enabled: bool) -> None:
        """Switch only the histogram count axis between linear and log scale."""
        self.histogram.setLogMode(x=False, y=enabled)

    def _threshold_changed(self) -> None:
        low, high = self._thresholds(); span = self.data_high - self.data_low; _px = (lambda v: int(round(255 * (v - self.data_low) / span))) if span else (lambda v: 0); self.low_label.setText(f"Lower threshold: {low:.6g}  (pixel value {_px(low)}/255)"); self.high_label.setText(f"Upper threshold: {high:.6g}  (pixel value {_px(high)}/255)")
        self._updating_edits = True; self.low_edit.setText(f"{low:.6g}"); self.high_edit.setText(f"{high:.6g}"); self._updating_edits = False
        if getattr(self, "hist_region", None) is not None:
            self.hist_region.setRegion((low, high))
        mask = np.isfinite(self.feature_map) & (self.feature_map >= low) & (self.feature_map <= high); rgba = np.zeros((*mask.shape, 4), dtype=np.ubyte); rgba[mask] = (255, 0, 0, 100); self._overlay.setImage(rgba, autoLevels=False); selected = int(mask.sum()); dx = float(np.median(np.diff(self.scan.header.x_um))) if self.scan.header.x_um is not None and len(self.scan.header.x_um) > 1 else 1.0; dy = float(np.median(np.diff(self.scan.header.y_um))) if self.scan.header.y_um is not None and len(self.scan.header.y_um) > 1 else 1.0; self.area_label.setText(f"Selected area: {selected} pixels / {abs(selected * dx * dy):.6g} µm² / {selected / self.feature_map.size * 100:.2f}%")

    def _cluster_rgba(self) -> np.ndarray:
        labels = self.result.labels; rgba = np.zeros((*labels.shape, 4), dtype=np.ubyte); palette = [(255, 80, 80), (80, 180, 255), (100, 220, 100), (255, 190, 60), (200, 100, 255), (50, 220, 200), (255, 100, 180), (160, 220, 80)]
        selected = self._selected_cluster_ids()
        visible = self._visible_cluster_ids()
        for cluster_id in range(1, int(labels.max()) + 1):
            if cluster_id not in visible:
                continue
            alpha = 255 if not selected or cluster_id in selected else 150
            rgba[labels == cluster_id] = (*palette[(cluster_id - 1) % len(palette)], alpha)
        return rgba

    def _selected_cluster_ids(self) -> set[int]:
        ids = set()
        for index in self.table.selectionModel().selectedRows():
            item = self.table.item(index.row(), 0)
            if item is not None:
                ids.add(int(item.text()))
        return ids

    def _visible_cluster_ids(self) -> set[int]:
        """Cluster ids that pass the minimum-size and quality filters."""
        visible = set()
        for row in range(self.table.rowCount()):
            if self.table.isRowHidden(row):
                continue
            good = self.table.item(row, 2)
            if self.exclude_bad.isChecked() and good is not None and good.checkState() != Qt.CheckState.Checked:
                continue
            item = self.table.item(row, 0)
            if item is not None:
                visible.add(int(item.text()))
        return visible

    @staticmethod
    def _dim_color(color: str) -> QColor:
        c = QColor(color)
        return QColor(int(c.red() * 0.32), int(c.green() * 0.32), int(c.blue() * 0.32))

    def _show_cluster_overlay(self) -> None:
        self._overlay.setImage(self._cluster_rgba(), autoLevels=False)

    def _clear_result_view(self) -> None:
        """Discard the previous run before calculating a new segmentation."""
        self.result = None
        self._profiles_by_id = {}
        self._areas_by_id = {}
        self.profile_plot.clear()
        self._overlay.clear()
        self.table.setSortingEnabled(False)
        self.table.blockSignals(True)
        try:
            self.table.clearContents()
            self.table.setRowCount(0)
            self.table.clearSelection()
        finally:
            self.table.blockSignals(False)
        self.save_button.setEnabled(False)

    def run_segmentation(self) -> None:
        self._select_new_mode()
        self._clear_result_view()
        # A fresh set of clusters carries no quality judgements yet: every new
        # "Good" box is unchecked.  If the "Exclude bad data" filter were left
        # enabled from a previous run it would treat every new cluster as bad
        # and hide it, leaving the overlay and profile plot empty on the second
        # and later runs.  Reset the filter so a new Run always shows results.
        self.exclude_bad.blockSignals(True); self.exclude_bad.setChecked(False); self.exclude_bad.blockSignals(False)
        self.result = segment_clusters(self.feature_map, self.od_stack, threshold=self._thresholds(), minimum_area=self.min_area_spin.value())
        self.result = replace(self.result, analysis_parameters={
            "source": self.source_label, "threshold": list(self.result.threshold),
            "minimum_area_pixels": self.result.minimum_area, "connectivity": 4,
            "frame_index": self.od_slider.value() if self._is_stack_layer(self.source_label) else None,
            "energy_eV": float(self.scan.energies_eV[self.od_slider.value()]) if self._is_stack_layer(self.source_label) else None,
            "profile_source": "OD (analysis input)",
        })
        ids = list(range(1, len(self.result.areas) + 1))
        self._profiles_by_id = {cluster_id: self.result.mean_profiles[cluster_id - 1]
                                for cluster_id in ids}
        self._areas_by_id = {cluster_id: int(self.result.areas[cluster_id - 1])
                             for cluster_id in ids}
        self._populate_table(ids, self.result.areas.astype(int).tolist(), saved=False)
        self._show_cluster_overlay(); self.status.setText(f"{int(np.sum(self.result.areas >= self.min_area_spin.value()))} clusters above minimum size"); self.save_button.setEnabled(True); self._plot_selected()

    def _plot_selected(self) -> None:
        self.profile_plot.clear()
        if self.result is None: return
        # Keep the standalone cluster-label view in sync as well as the
        # translucent overlay used for an input image.  Previously a
        # selection change only refreshed the overlay, so the "Cluster
        # labels" layer appeared not to react to selection.
        if self.layer_combo.currentText() == "Cluster labels":
            self.input_view.setImage(self._cluster_rgba(), autoLevels=False, autoRange=False)
        else:
            self._show_cluster_overlay()
        palette = ["#ff5050", "#50b4ff", "#64dc64", "#ffbe3c", "#c864ff", "#32dcca", "#ff64b4", "#a0dc50"]
        selected_ids = self._selected_cluster_ids()
        visible_ids = self._visible_cluster_ids()
        rows = [row for row in range(self.table.rowCount()) if not self.table.isRowHidden(row) and self.table.item(row, 0) is not None and int(self.table.item(row, 0).text()) in visible_ids]
        # Always draw all acceptable profiles.  Selection changes only their
        # emphasis; it must not remove the context profiles from the plot.
        rows.sort(key=lambda row: int(self.table.item(row, 0).text()) in selected_ids)
        selected_values = []
        for index in rows:
            cluster_id = int(self.table.item(index, 0).text())
            raw_values = self._profiles_by_id.get(cluster_id)
            if raw_values is None:
                continue
            values = raw_values.copy()
            if self.norm_combo.currentIndex() == 0:
                lo, hi = np.nanmin(values), np.nanmax(values); values = (values - lo) / (hi - lo) if hi != lo else np.zeros_like(values)
            elif self.norm_combo.currentIndex() == 2:
                a = np.interp(self.norm_e1.value(), self.scan.energies_eV, values); b = np.interp(self.norm_e2.value(), self.scan.energies_eV, values); values = (values - a) / (b - a) if b != a else np.zeros_like(values)
            # Table sorting changes row indices; compare the stable cluster
            # identifier rather than the transient row number.
            selected = cluster_id in selected_ids
            if selected:
                selected_values.append(raw_values.copy())
            base_color = "#ffffff" if cluster_id == 0 else palette[(cluster_id - 1) % len(palette)]
            color = base_color if not selected_ids or selected else self._dim_color(base_color)
            width = 4 if selected else (2 if not selected_ids else 1)
            self.profile_plot.plot(self.scan.energies_eV, values, pen=pg.mkPen(color, width=width), name=f"Cluster {cluster_id}")
        if len(selected_values) > 1:
            average = np.nanmean(selected_values, axis=0)
            if self.norm_combo.currentIndex() == 0:
                lo, hi = np.nanmin(average), np.nanmax(average); average = (average - lo) / (hi - lo) if hi != lo else np.zeros_like(average)
            elif self.norm_combo.currentIndex() == 2:
                a = np.interp(self.norm_e1.value(), self.scan.energies_eV, average); b = np.interp(self.norm_e2.value(), self.scan.energies_eV, average); average = (average - a) / (b - a) if b != a else np.zeros_like(average)
            self.profile_plot.plot(self.scan.energies_eV, average, pen=pg.mkPen("#ffffff", width=6), name="Cluster 0 average")

    def save_result(self) -> None:
        if self.result is None:
            return
        visible = sorted(self._visible_cluster_ids())
        ids = [cluster_id for cluster_id in visible if cluster_id in self._profiles_by_id and cluster_id != 0]
        profiles = [self._profiles_by_id[cluster_id] for cluster_id in ids]
        areas = [self._areas_by_id[cluster_id] for cluster_id in ids]
        selected = sorted(self._selected_cluster_ids() & set(visible))
        if len(selected) > 1:
            profiles.append(np.nanmean([self._profiles_by_id[i] for i in selected], axis=0))
            areas.append(int(sum(self._areas_by_id[i] for i in selected)))
            ids.append(0)
        label = self.label_edit.text().strip() or f"Segmentation {self._next_label_number()}"
        self._save_count = getattr(self, "_save_count", 0) + 1
        parameters = dict(self.result.analysis_parameters)
        parameters.update({"minimum_area_at_save_pixels": self.min_area_spin.value(),
                           "exclude_bad_data": self.exclude_bad.isChecked(),
                           "saved_cluster_count": len(visible), "average_cluster_ids": selected if len(selected) > 1 else [],
                           "average_weighting": "Equal weight per cluster" if len(selected) > 1 else None})
        saved_labels = np.where(np.isin(self.result.labels, visible), self.result.labels, 0).astype(np.int32)
        saved = SegmentationResult(saved_labels, self.result.areas, self.result.mean_profiles, self.result.threshold, self.result.minimum_area, np.asarray(profiles, dtype=np.float64), np.asarray(areas, dtype=np.int64), np.asarray(ids, dtype=np.int32), label, parameters)
        self.saved_groups[label] = {
            "profiles": np.asarray(profiles, dtype=np.float64).tolist(),
            "areas": [int(area) for area in areas],
            "cluster_ids": [int(cluster_id) for cluster_id in ids],
            "labels": saved_labels.astype(int).tolist(),
            "threshold": list(self.result.threshold),
            "minimum_area": int(self.result.minimum_area),
            "analysis_parameters": parameters,
        }
        if self.saved_label_combo.findData(label) < 0:
            self.saved_label_combo.addItem(label, label)
        self.resultReady.emit(saved)
        self.status.setText(f"Saved segmentation result: {label}")

    def _next_label_number(self) -> int:
        """Return a stable default label number for repeated saves."""
        return getattr(self, "_save_count", 0) + 1
