"""RGBY reference-spectrum R² fitting window."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QSizePolicy, QVBoxLayout, QWidget
from scipy.ndimage import gaussian_filter
from ..processing.fitting import SpectralFitResult, spectral_r2_map

class FittingWindow(QMainWindow):
    resultReady = Signal(object)
    colors = ("R", "G", "B", "Y")
    def __init__(self, energies_eV: np.ndarray, od_stack: np.ndarray, profile_groups: dict, pre_edge_range: tuple[float, float] | None = None, *, input_source: str = "OD") -> None:
        super().__init__(); self.setMinimumSize(0, 0); self.setMinimumWidth(0); self.energies_eV, self.od_stack, self.groups = energies_eV, od_stack, profile_groups; self.pre_edge_range = pre_edge_range; self.result = None
        self.setWindowTitle("muNEXAFS — Peak-Targeted Endmember Extraction (PTEE, + R2 RGBY)"); self.resize(1100, 760)
        self.input_source = input_source
        root = QVBoxLayout(); self.labels=[]; self.profiles=[]
        body = QHBoxLayout(); left = QVBoxLayout(); right = QVBoxLayout()
        for color in self.colors:
            channel=QLabel(color); channel.setFixedWidth(12); label=QComboBox(); label.setMinimumWidth(0); label.addItem("None"); label.addItems(list(self.groups)); profile=QComboBox(); profile.setMinimumWidth(0); self.labels.append(label); self.profiles.append(profile); index=len(self.labels)-1; label.currentTextChanged.connect(lambda _, i=index: self._profiles_changed(i)); profile.currentIndexChanged.connect(self._plot_refs); right.addWidget(channel); right.addWidget(label); right.addWidget(profile)
        # Normalization modes.  Options 2-4 first remove the pre-edge baseline
        # (from both pixels and references) so the map reflects chemistry, not
        # thickness; they then differ in how amplitude is handled:
        #   1 Min–max              : shape only (baseline + amplitude normalized)
        #   2 Subtract pre-edge    : baseline removed, amplitude kept (量に敏感)
        #   3 " + Absolute max     : baseline removed, scaled by |max|
        #   4 " + max at energy    : baseline removed, scaled so a chosen energy = 1
        right.addWidget(QLabel("Profile normalization")); self.norm=QComboBox(); self.norm.addItems(["Min–max", "Subtract pre-edge", "Subtract pre-edge + Absolute max", "Subtract pre-edge + max at energy"]); self.norm.currentIndexChanged.connect(self._norm_changed); right.addWidget(self.norm)
        # Energy at which mode 4 normalizes every spectrum to 1, so pixel and
        # reference are made to agree at that energy.  Shown only for mode 4.
        self.norm_energy_label=QLabel("Normalize at"); right.addWidget(self.norm_energy_label); self.norm_energy=QComboBox(); self.norm_energy.addItems([f"{e:.4g} eV" for e in self.energies_eV]); self.norm_energy.setCurrentIndex(len(self.energies_eV)-1); self.norm_energy.currentIndexChanged.connect(self._plot_refs); right.addWidget(self.norm_energy)
        self.norm_energy_label.setVisible(False); self.norm_energy.setVisible(False)
        right.addWidget(QLabel("R² floor")); self.floor=QDoubleSpinBox(); self.floor.setRange(-1,1); self.floor.setSingleStep(.01); self.floor.setValue(.9); right.addWidget(self.floor)
        self.blur_check=QCheckBox("Gaussian blur"); self.blur_check.setChecked(True); right.addWidget(self.blur_check); blur_row=QHBoxLayout(); blur_row.addWidget(QLabel("Radius")); self.blur_radius=QLineEdit("0.7"); self.blur_radius.setMaximumWidth(70); blur_row.addWidget(self.blur_radius); right.addLayout(blur_row)
        self.run=QPushButton("Run"); self.run.clicked.connect(self.run_fit); right.addWidget(self.run); self.save=QPushButton("Save map"); self.save.setEnabled(False); self.save.clicked.connect(self.save_map); right.addWidget(self.save); right.addStretch(1)
        self.plot=pg.PlotWidget(title="Reference profiles (RGBY)"); self.plot.setMinimumSize(0,0); self.plot.setLabel("bottom","Energy",units="eV"); self.plot.setLabel("left","OD"); left.addWidget(self.plot,2)
        self.map=pg.ImageView(view=pg.PlotItem()); self.map.setMinimumSize(0,0); self.map.ui.roiBtn.hide(); self.map.ui.menuBtn.hide(); left.addWidget(self.map,3); self.status=QLabel("Select references and Run"); left.addWidget(self.status); body.addLayout(left,4); body.addLayout(right,1); root.addLayout(body)
        central=QWidget(); central.setMinimumSize(0,0); central.setLayout(root); self.setCentralWidget(central)
        for i in range(4): self._profiles_changed(i)
    def _profiles_changed(self, i):
        self.profiles[i].clear(); group=self.groups.get(self.labels[i].currentText(), {}); ids=group.get("cluster_ids", []) if self.labels[i].currentText() != "None" else []; prof=group.get("profiles", []) if self.labels[i].currentText() != "None" else []
        self.profiles[i].addItems([str(x) for x in ids]); self._plot_refs()
    def _selected_refs(self):
        out=[]
        for label, profile in zip(self.labels,self.profiles):
            g=self.groups.get(label.currentText(),{}); p=g.get("profiles",[]); idx=profile.currentIndex(); out.append(np.asarray(p[idx],float) if label.currentText() != "None" and 0<=idx<len(p) else np.full(len(self.energies_eV),np.nan))
        return np.asarray(out)
    def _norm_changed(self):
        show = self.norm.currentIndex() == 3
        self.norm_energy_label.setVisible(show); self.norm_energy.setVisible(show)
        self._plot_refs()
    def _pre_range(self):
        if self.pre_edge_range is not None:
            return self.pre_edge_range
        n = max(1, min(3, len(self.energies_eV)))
        return (float(np.min(self.energies_eV[:n])), float(np.max(self.energies_eV[:n])))
    def _presub_row(self, row):
        lo, hi = sorted(self._pre_range()); mask = (self.energies_eV >= lo) & (self.energies_eV <= hi)
        if not (mask.any() and np.isfinite(row).any()):
            return np.asarray(row, float)
        pre = np.nanmean(np.where(mask, row, np.nan)); return np.maximum(np.asarray(row, float) - pre, 0.0)
    def _norm_energy_index(self):
        return max(0, min(self.norm_energy.currentIndex(), len(self.energies_eV) - 1))
    @staticmethod
    def _minmax(r):
        r = np.asarray(r, float); lo, hi = np.nanmin(r), np.nanmax(r); return (r - lo) / (hi - lo) if hi != lo else np.zeros_like(r)
    @staticmethod
    def _absmax(r):
        r = np.asarray(r, float); s = np.nanmax(np.abs(r)); return r / s if s else np.zeros_like(r)
    def _norm_at_energy_row(self, row):
        row = np.asarray(row, float); d = row[self._norm_energy_index()]
        return row / d if np.isfinite(d) and d != 0 else np.full_like(row, np.nan)
    def _display_refs(self):
        """Selected references processed exactly as the map will use them, for
        the preview plot."""
        refs = self._selected_refs(); mode = self.norm.currentIndex()
        if mode == 0:
            return [self._minmax(r) if np.isfinite(r).any() else r for r in refs]
        refs = [self._presub_row(r) for r in refs]
        if mode == 2:
            return [self._absmax(r) if np.isfinite(r).any() else r for r in refs]
        if mode == 3:
            return [self._norm_at_energy_row(r) if np.isfinite(r).any() else r for r in refs]
        return refs
    def _prepare(self, data, refs):
        """Return (data, refs, spectral_r2_map normalization) for the mode."""
        mode = self.norm.currentIndex()
        if mode == 0:
            return data, refs, "Min–max"
        from ..processing.premap import pre_edge_subtract
        data = pre_edge_subtract(data, self.energies_eV, self._pre_range())
        refs = np.asarray([self._presub_row(r) for r in refs])
        if mode == 2:
            return data, refs, "Absolute max"
        if mode == 3:
            idx = self._norm_energy_index(); denom = np.where(data[idx] == 0, np.nan, data[idx])
            data = data / denom[None]
            refs = np.asarray([self._norm_at_energy_row(r) for r in refs])
            return data, refs, "None"
        return data, refs, "None"
    def _plot_refs(self):
        self.plot.clear(); refs=self._display_refs(); colors=("#ff4040","#40d060","#4080ff","#ffd020")
        for i,row in enumerate(refs):
            if np.isfinite(row).any():
                self.plot.plot(self.energies_eV,row,pen=pg.mkPen(colors[i],width=2),name=self.colors[i])
    def run_fit(self):
        refs=self._selected_refs()
        if not np.isfinite(refs).any(): self.status.setText("Select at least one reference profile"); return
        raw_refs = refs.copy()
        data, refs, norm = self._prepare(self.od_stack, refs)
        try: radius=max(0.0,float(self.blur_radius.text()))
        except ValueError: radius=2.0
        if self.blur_check.isChecked() and radius > 0: data=gaussian_filter(data,sigma=(0,radius,radius))
        parameters = {
            "input_source": self.input_source,
            "profile_normalization": self.norm.currentText(),
            "r2_floor": self.floor.value(),
            "gaussian_blur": self.blur_check.isChecked(),
            "gaussian_sigma_pixels": radius if self.blur_check.isChecked() else 0.0,
            "channels": {
                color: {"label": label.currentText(), "cluster_id": profile.currentText() or None,
                        "active": bool(np.isfinite(raw_refs[i]).all())}
                for i, (color, label, profile) in enumerate(zip(self.colors, self.labels, self.profiles))
            },
        }
        if self.norm.currentIndex() != 0:
            parameters["pre_edge_range_eV"] = list(self._pre_range())
            parameters["clip_negative_after_pre_edge"] = True
        if self.norm.currentIndex() == 3:
            parameters["normalization_energy_eV"] = float(self.energies_eV[self._norm_energy_index()])
        self.result = replace(
            spectral_r2_map(data, self.energies_eV, refs, normalization=norm, r2_floor=self.floor.value()),
            raw_references=raw_refs, analysis_parameters=parameters,
        )
        self.map.setImage(self.result.rgb_y,autoLevels=False,autoRange=True); self.status.setText("R² RGBY map ready"); self.save.setEnabled(True); self.resultReady.emit(self.result)

    def save_map(self):
        if self.result is not None:
            self.resultReady.emit(self.result)
            self.status.setText("Spectral map saved")
