"""PCA + clustering (and SVD/PCA linear-combination) analysis window.

The default workflow is **PCA + k-means clustering** — the standard automated
STXM phase-classification workflow (Lerotic 2004 / MANTiS): PCA of the stack,
k-means on the scores, giving a coloured cluster map, a PC-space scatter, and
per-cluster mean OD spectra.  The earlier PC->RGB linear-combination composite
(PCA / SVD / low-rank) is retained as an alternative method and as a display
toggle within the clustering view.
"""
from __future__ import annotations
import threading
from dataclasses import replace
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QProgressBar, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)
from ..processing.multivariate import (
    AnalysisCancelled, ClusterResult, MultivariateResult, cluster_stack,
    decompose_stack, linear_combination_map, mcr_als_reconstruct, nmf_reconstruct,
    nnls_reconstruct,
)

CLUSTER_METHOD = "PCA + clustering"
SCATTER_MAX_POINTS = 20000


class MultivariateSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("muNEXAFS — Multivariate analysis settings")
        self.resize(420, 300)
        root = QVBoxLayout(self)

        common_box = QGroupBox("Common settings")
        common = QFormLayout(common_box)
        self.components = QSpinBox(); self.components.setRange(1, 50); self.components.setValue(5)
        common.addRow("Number of components", self.components)
        self.iterations = QSpinBox(); self.iterations.setRange(1, 100); self.iterations.setValue(1)
        common.addRow("Linear-combination iterations", self.iterations)
        root.addWidget(common_box)

        mcr_box = QGroupBox("MCR-ALS settings")
        mcr = QFormLayout(mcr_box)
        self.nonnegative = QCheckBox(); self.nonnegative.setChecked(True)
        mcr.addRow("Nonnegative constraint", self.nonnegative)
        self.closure = QCheckBox()
        mcr.addRow("Closure constraint", self.closure)
        self.max_iter = QSpinBox(); self.max_iter.setRange(1, 1000); self.max_iter.setValue(50)
        mcr.addRow("Maximum iterations", self.max_iter)
        self.tolerance = QDoubleSpinBox(); self.tolerance.setDecimals(8); self.tolerance.setRange(1e-8, 1); self.tolerance.setValue(1e-5)
        mcr.addRow("Convergence tolerance", self.tolerance)
        root.addWidget(mcr_box)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        root.addWidget(buttons)


class _Worker(QObject):
    finished = Signal(object)
    cancelled = Signal()
    failed = Signal(str)
    done = Signal()
    progress = Signal(int)
    status = Signal(str)

    def __init__(self, stack, components, method, coefficients, iterations, n_clusters,
                 use_nmf, use_nnls, use_mcr, mcr_settings, cancel_event):
        super().__init__()
        self.stack = stack; self.components = components; self.method = method
        self.coefficients = coefficients; self.iterations = iterations
        self.n_clusters = n_clusters; self.use_nmf = use_nmf; self.use_nnls = use_nnls
        self.use_mcr = use_mcr; self.mcr_settings = mcr_settings; self.cancel_event = cancel_event

    def _check_cancelled(self):
        if self.cancel_event.is_set():
            raise AnalysisCancelled("Analysis stopped")

    def _report(self, label, start, end):
        def report(fraction):
            self._check_cancelled()
            self.progress.emit(start + int((end - start) * fraction))
            self.status.emit(f"{label}: {fraction:.0%}")
        return report

    @Slot()
    def run(self):
        try:
            self._check_cancelled()
            self.progress.emit(0)
            if self.method == CLUSTER_METHOD:
                self.status.emit("PCA + clustering…")
                result = cluster_stack(
                    self.stack, self.components, self.n_clusters,
                    cancel_check=self.cancel_event.is_set,
                    progress_callback=self._report("PCA + k-means", 0, 100),
                )
                self._check_cancelled()
                self.finished.emit(result)
                return
            if self.use_mcr:
                max_iter, tolerance, nonnegative, closure = self.mcr_settings
                recon, scores, spectra, count = mcr_als_reconstruct(
                    self.stack, self.components, max_iter, tolerance,
                    nonnegative, closure, self.cancel_event.is_set,
                    self._report("MCR-ALS (initialization / ALS)", 0, 90),
                )
                method = "MCR-ALS"
            elif self.use_nmf or self.use_nnls:
                recon, scores, spectra = nmf_reconstruct(
                    self.stack, self.components,
                    cancel_check=self.cancel_event.is_set,
                    progress_callback=self._report("NMF (negative OD clipped to 0)", 0, 60 if self.use_nnls else 90),
                )
                self._check_cancelled()
                method = "NMF"
                if self.use_nnls:
                    recon, scores = nnls_reconstruct(
                        self.stack, spectra.T, cancel_check=self.cancel_event.is_set,
                        return_scores=True,
                        progress_callback=self._report("NNLS coefficients", 60, 90),
                    )
                    method = "NNLS"
            else:
                self.status.emit(f"{self.method}: decomposition…")
                result = decompose_stack(self.stack, self.components, self.method)

            self._check_cancelled()
            if self.use_mcr or self.use_nmf or self.use_nnls:
                # Nonnegative factors are not principal components: there is
                # no reason to run a second, unrelated PCA for its metadata.
                valid = np.all(np.isfinite(self.stack), axis=0)
                mean = np.mean(self.stack[:, valid], axis=1)
                result = MultivariateResult(method, spectra, scores, recon, mean, np.empty(0))
            if not np.all(np.isfinite(result.components)) or not np.all(np.isfinite(result.scores)):
                raise ValueError("Non-finite factors produced; analysis was not saved")
            rgb = linear_combination_map(
                result.scores, result.reconstructed.shape[1:], self.coefficients,
                self.iterations, cancel_check=self.cancel_event.is_set,
                progress_callback=self._report("RGB mapping", 90, 100),
            )
            # Invalid registration borders remain transparent.
            rgb[~np.all(np.isfinite(self.stack), axis=0)] = 0
            self._check_cancelled()
            parameters = {
                "method": result.method,
                "components_requested": int(self.components),
                "components_returned": int(len(result.components)),
                "rgb_coefficients": self.coefficients.tolist(),
                "linear_combination_iterations": int(self.iterations),
                "RGB_scaling_percentiles": [1, 99],
            }
            if self.use_mcr:
                parameters.update({"maximum_iterations": max_iter, "iterations_completed": count,
                                   "convergence_tolerance": tolerance, "nonnegative": nonnegative,
                                   "closure": closure})
            elif self.use_nmf or self.use_nnls:
                parameters.update({"NMF_maximum_iterations": 300, "NMF_tolerance": 1e-6,
                                   "NMF_negative_OD_clipped": True,
                                   "NNLS_on_original_OD": bool(self.use_nnls)})
            self.finished.emit(replace(result, rgb_map=rgb, analysis_parameters=parameters))
        except AnalysisCancelled:
            self.cancelled.emit()
        except Exception as exc:
            if self.cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            # All exits, including numerical errors, must stop the QThread.
            self.done.emit()


class MultivariateWindow(QMainWindow):
    resultReady = Signal(object)

    def __init__(self, energies_eV: np.ndarray, stack: np.ndarray, *, input_source: str = "OD") -> None:
        super().__init__()
        self.energies_eV = energies_eV; self.stack = stack; self.result = None
        self.setWindowTitle("muNEXAFS — PCA + clustering"); self.resize(1150, 760); self.setMinimumSize(0, 0)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.input_source = input_source
        self.thread = None
        self.worker = None
        self._cancel_event = threading.Event()
        self._close_pending = False

        root = QVBoxLayout()
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Method"))
        self.method = QComboBox()
        self.method.addItems([CLUSTER_METHOD, "PCA (centered)", "SVD (uncentered)",
                              "Low-rank reconstruction (centered)"])
        self.method.currentTextChanged.connect(self._method_changed)
        controls.addWidget(self.method)
        self.clusters_label = QLabel("Clusters (k)")
        controls.addWidget(self.clusters_label)
        self.clusters = QSpinBox(); self.clusters.setRange(2, 50); self.clusters.setValue(6)
        controls.addWidget(self.clusters)
        self.nmf_check = QCheckBox("NMF"); self.nnls_check = QCheckBox("NNLS"); self.mcr_check = QCheckBox("MCR-ALS")
        controls.addWidget(self.nmf_check); controls.addWidget(self.nnls_check); controls.addWidget(self.mcr_check)
        self.settings_button = QPushButton("Settings"); self.settings_button.clicked.connect(self.show_settings)
        controls.addWidget(self.settings_button)
        self.progress = QProgressBar(); controls.addWidget(self.progress, 1)
        self.run = QPushButton("Run"); self.run.clicked.connect(self.run_analysis); controls.addWidget(self.run)
        self.stop = QPushButton("Stop"); self.stop.setEnabled(False); self.stop.clicked.connect(self.stop_analysis)
        controls.addWidget(self.stop)
        root.addLayout(controls)

        coeff = QHBoxLayout(); self.coefficients = []
        self.coeff_widgets = []
        for channel in ("R", "G", "B"):
            lbl = QLabel(channel); coeff.addWidget(lbl); self.coeff_widgets.append(lbl); row = []
            for pc in range(3):
                spin = QDoubleSpinBox(); spin.setRange(-5, 5); spin.setSingleStep(.1)
                spin.setValue(1.0 if (channel, pc) in (("R", 0), ("G", 1), ("B", 2)) else 0.0)
                spin.setPrefix(f"PC{pc+1} "); spin.valueChanged.connect(self._coeff_changed)
                row.append(spin); coeff.addWidget(spin); self.coeff_widgets.append(spin)
            self.coefficients.append(row)
        # Display + PC-axis selectors (for clustering view)
        self.display_label = QLabel("Display")
        coeff.addWidget(self.display_label)
        self.display_combo = QComboBox(); self.display_combo.addItems(["Cluster map", "RGB composite"])
        self.display_combo.currentTextChanged.connect(self._update_image)
        coeff.addWidget(self.display_combo)
        self.pcx_label = QLabel("scatter PC x/y"); coeff.addWidget(self.pcx_label)
        self.pc_x = QSpinBox(); self.pc_x.setRange(1, 50); self.pc_x.setValue(1); self.pc_x.valueChanged.connect(self._update_scatter)
        self.pc_y = QSpinBox(); self.pc_y.setRange(1, 50); self.pc_y.setValue(2); self.pc_y.valueChanged.connect(self._update_scatter)
        coeff.addWidget(self.pc_x); coeff.addWidget(self.pc_y)
        coeff.addStretch(1)
        root.addLayout(coeff)

        self.profile = pg.PlotWidget(title="Cluster mean spectra"); self.profile.addLegend()
        self.profile.setLabel("bottom", "Energy", units="eV"); root.addWidget(self.profile, 2)

        views = QHBoxLayout()
        self.image = pg.ImageView(view=pg.PlotItem()); self.image.ui.roiBtn.hide(); self.image.ui.menuBtn.hide()
        views.addWidget(self.image, 3)
        self.scatter = pg.PlotWidget(title="PC space (coloured by cluster)")
        self.scatter.setLabel("bottom", "PC1"); self.scatter.setLabel("left", "PC2")
        self.scatter_item = pg.ScatterPlotItem(size=4, pen=None)
        self.scatter.addItem(self.scatter_item)
        views.addWidget(self.scatter, 2)
        root.addLayout(views, 3)

        self.settings = MultivariateSettingsDialog(self)
        self.settings.components.setMaximum(min(50, stack.shape[0]))
        self.settings.components.setValue(min(5, stack.shape[0]))
        self.status = QLabel("Run PCA + clustering for a cluster map, PC-space scatter and cluster mean spectra")
        self.status.setWordWrap(True); root.addWidget(self.status)
        c = QWidget(); c.setLayout(root); self.setCentralWidget(c)
        self.nmf_check.setToolTip("NMF factors nonnegative OD: negative input values are clipped to 0 for this calculation.")
        self.nnls_check.setToolTip("Fit nonnegative coefficients to the original OD using NMF-derived reference spectra.")
        self.mcr_check.setToolTip("Alternating least-squares fitting of spectra and concentration maps to OD.")
        self._method_changed(self.method.currentText())

    # ----- method / view switching ----- #
    def _is_cluster(self):
        return self.method.currentText() == CLUSTER_METHOD

    def _method_changed(self, _text=None):
        cluster = self._is_cluster()
        for w in (self.clusters_label, self.clusters, self.display_label, self.display_combo,
                  self.pcx_label, self.pc_x, self.pc_y, self.scatter):
            w.setVisible(cluster)
        for w in (self.nmf_check, self.nnls_check, self.mcr_check):
            w.setVisible(not cluster)
        self.profile.setTitle("Cluster mean spectra" if cluster else "Component spectra")

    def _coeff_changed(self, *_):
        if self.display_combo.currentText() == "RGB composite" and isinstance(self.result, ClusterResult):
            self._update_image()

    def show_settings(self):
        self.settings.exec()

    # ----- run ----- #
    def run_analysis(self):
        if self.thread is not None:
            return
        self._set_running(True)
        self.progress.setValue(0)
        self.status.setText("Starting analysis…")
        self._cancel_event = threading.Event()
        coeff = np.asarray([[spin.value() for spin in row] for row in self.coefficients], float)
        mcr_settings = (self.settings.max_iter.value(), self.settings.tolerance.value(),
                        self.settings.nonnegative.isChecked(), self.settings.closure.isChecked())
        self.thread = QThread(self)
        self.worker = _Worker(
            self.stack, self.settings.components.value(), self.method.currentText(),
            coeff, self.settings.iterations.value(), self.clusters.value(),
            self.nmf_check.isChecked(), self.nnls_check.isChecked(), self.mcr_check.isChecked(),
            mcr_settings, self._cancel_event,
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._analysis_progress)
        self.worker.status.connect(self._analysis_status)
        self.worker.finished.connect(self._analysis_finished)
        self.worker.cancelled.connect(self._analysis_cancelled)
        self.worker.failed.connect(self._analysis_failed)
        self.worker.done.connect(self.thread.quit, Qt.ConnectionType.DirectConnection)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _set_running(self, running):
        self.run.setEnabled(not running)
        self.stop.setEnabled(running)
        for widget in (self.method, self.clusters, self.nmf_check, self.nnls_check,
                       self.mcr_check, self.settings_button):
            widget.setEnabled(not running)
        for row in self.coefficients:
            for spin in row:
                spin.setEnabled(not running)

    @Slot(int)
    def _analysis_progress(self, value):
        if not self._cancel_event.is_set():
            self.progress.setValue(value)

    @Slot(str)
    def _analysis_status(self, message):
        if not self._cancel_event.is_set():
            self.status.setText(message)

    def stop_analysis(self):
        if self.thread is not None:
            self._cancel_event.set(); self.stop.setEnabled(False); self.status.setText("Stopping analysis…")

    @Slot()
    def _analysis_cancelled(self):
        self.progress.setValue(0)
        self.status.setText("Analysis stopped — previous result retained")

    @Slot(str)
    def _analysis_failed(self, message):
        self.progress.setValue(0)
        self.status.setText(f"Analysis failed: {message}")

    @Slot()
    def _thread_finished(self):
        self.thread = None
        self.worker = None
        self._set_running(False)
        if self._close_pending:
            self.close()

    # ----- display ----- #
    @Slot(object)
    def _analysis_finished(self, result):
        if self._cancel_event.is_set() or self._close_pending:
            self._analysis_cancelled()
            return
        result = replace(result, analysis_parameters={**result.analysis_parameters,
                                                       "input_source": self.input_source})
        self.result = result
        if isinstance(result, ClusterResult):
            self._show_cluster_result(result)
        else:
            self._show_multivariate_result(result)
        self.progress.setValue(100)
        self.resultReady.emit(result)

    def _show_cluster_result(self, result):
        self.profile.clear()
        for c in range(result.n_clusters):
            if int(result.cluster_sizes[c]) == 0:
                continue
            col = tuple(int(255 * v) for v in result.colors[c])
            self.profile.plot(self.energies_eV, result.mean_spectra[c],
                              pen=pg.mkPen(col, width=2),
                              name=f"Cluster {c+1} ({int(result.cluster_sizes[c])} px)")
        self.profile.enableAutoRange()
        self.pc_x.setMaximum(result.scores.shape[1]); self.pc_y.setMaximum(result.scores.shape[1])
        self._update_image()
        self._update_scatter()
        n_used = int(np.sum(result.cluster_sizes > 0))
        self.status.setText(f"PCA + k-means: {n_used} non-empty clusters of {result.n_clusters} "
                            f"({int(result.scores.shape[1])} PCA components)")

    def _show_multivariate_result(self, result):
        self.profile.clear()
        colors = ("r", "g", "b")
        prefix = "Component" if result.method in ("NMF", "NNLS", "MCR-ALS") else "PC"
        for i, row in enumerate(result.components):
            color = colors[i] if i < 3 else pg.intColor(i, len(result.components))
            self.profile.plot(self.energies_eV, row, pen=pg.mkPen(color, width=2), name=f"{prefix} {i+1}")
        self.profile.enableAutoRange()
        self.image.setImage(result.rgb_map, autoLevels=False, autoRange=True, levels=(0.0, 1.0),
                            axes={"t": None, "x": 1, "y": 0, "c": 2})
        note = " (negative OD clipped to 0)" if result.method == "NMF" else ""
        self.status.setText(f"{result.method} profiles and RGB map ready{note}")

    def _update_image(self, *_):
        result = self.result
        if not isinstance(result, ClusterResult):
            return
        if self.display_combo.currentText() == "RGB composite":
            coeff = np.asarray([[spin.value() for spin in row] for row in self.coefficients], float)
            rgb = linear_combination_map(result.scores, result.shape, coeff, self.settings.iterations.value())
            rgb[result.labels < 0] = 0
            image = rgb
        else:
            image = result.rgb_map
        self.image.setImage(image, autoLevels=False, autoRange=True, levels=(0.0, 1.0),
                            axes={"t": None, "x": 1, "y": 0, "c": 2})

    def _update_scatter(self, *_):
        result = self.result
        self.scatter_item.clear()
        if not isinstance(result, ClusterResult):
            return
        xi = self.pc_x.value() - 1; yi = self.pc_y.value() - 1
        if max(xi, yi) >= result.scores.shape[1]:
            return
        labels = result.labels.reshape(-1)
        idx = np.flatnonzero(labels >= 0)
        if idx.size > SCATTER_MAX_POINTS:
            idx = np.random.default_rng(0).choice(idx, SCATTER_MAX_POINTS, replace=False)
        xs = result.scores[idx, xi]; ys = result.scores[idx, yi]; labs = labels[idx]
        brushes = [pg.mkBrush(int(255 * result.colors[c][0]), int(255 * result.colors[c][1]),
                              int(255 * result.colors[c][2]), 150) for c in labs]
        self.scatter_item.setData(x=xs, y=ys, brush=brushes, pen=None, size=4)
        self.scatter.setLabel("bottom", f"PC{xi+1}"); self.scatter.setLabel("left", f"PC{yi+1}")

    def closeEvent(self, event):
        if self.thread is not None:
            self._close_pending = True
            self.stop_analysis()
            event.ignore()
        else:
            event.accept()
