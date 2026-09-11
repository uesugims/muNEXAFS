"""Preview the actual paginated report and export its figures and spectra."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from PySide6.QtCore import QObject, QPointF, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtPdf import QPdfDocument
from PySide6.QtPdfWidgets import QPdfView
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QVBoxLayout, QWidget,
)

from ..reporting import (
    _safe_name, export_layer_images, export_pdf, export_pptx, export_spectra_bundle,
)


class _ReportWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)
    done = Signal()

    def __init__(self, payload, directory, formats, preview=False):
        super().__init__()
        self.payload = payload
        self.directory = directory
        self.formats = formats
        self.preview = preview

    @Slot()
    def run(self):
        try:
            layers = self.payload.get("layers", [])
            metadata = self.payload.get("metadata", {})
            energies = self.payload.get("energies")
            if self.preview:
                self.progress.emit("Building report preview…")
                path = export_pdf(layers, metadata, self.directory, "preview.pdf", energies=energies)
                self.finished.emit({"preview": path})
                return
            stem = _safe_name(metadata.get("title", "muNEXAFS"))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder = self.directory / f"{stem}_report_{timestamp}"
            suffix = 2
            while folder.exists():
                folder = self.directory / f"{stem}_report_{timestamp}_{suffix}"
                suffix += 1
            folder.mkdir(parents=True)
            jobs = {
                "PPTX": lambda: [export_pptx(layers, metadata, folder, stem + ".pptx", energies=energies)],
                "PDF": lambda: [export_pdf(layers, metadata, folder, stem + ".pdf", energies=energies)],
                "CSV": lambda: export_spectra_bundle(
                    self.payload.get("spectra", {}), energies,
                    self.payload.get("mapping_spectra", {}), folder / "spectra", stem,
                ),
                "Images": lambda: export_layer_images(layers, folder / "layers"),
            }
            paths, failures = [], []
            for name in self.formats:
                self.progress.emit(f"Exporting {name}…")
                try:
                    paths.extend(jobs[name]())
                except Exception as exc:
                    failures.append(f"{name}: {exc}")
            self.finished.emit({"paths": paths, "failures": failures, "directory": folder})
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.done.emit()


class ReportWindow(QMainWindow):
    def __init__(self, payload: dict, parent=None):
        super().__init__(parent)
        self.payload = payload
        scan = payload.get("metadata", {}).get("scan")
        self.output_dir = Path(scan).parent if scan else Path.cwd()
        self._temporary = TemporaryDirectory(prefix="muaxis-report-preview-")
        self.thread = None
        self.worker = None
        self._close_pending = False
        self.exported_paths = []
        self.setWindowTitle("muNEXAFS — Report output")
        self.resize(1100, 820)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        central = QWidget(self)
        root = QVBoxLayout(central)
        header = QHBoxLayout()
        header.addWidget(QLabel("Analysis parameters, step images, profiles and conditions"), 1)
        self.previous = QPushButton("Previous")
        self.next = QPushButton("Next")
        self.page_label = QLabel("0 / 0")
        self.previous.clicked.connect(lambda: self._change_page(-1))
        self.next.clicked.connect(lambda: self._change_page(1))
        self.zoom = QComboBox()
        self.zoom.addItems(["Fit page", "Fit width", "100%"])
        self.zoom.currentIndexChanged.connect(self._zoom_changed)
        for item in (self.previous, self.page_label, self.next, self.zoom):
            header.addWidget(item)
        root.addLayout(header)

        self.document = QPdfDocument(self)
        self.preview = QPdfView(self)
        self.preview.setDocument(self.document)
        self.preview.setPageMode(QPdfView.PageMode.SinglePage)
        self.preview.setZoomMode(QPdfView.ZoomMode.FitInView)
        self.preview.pageNavigator().currentPageChanged.connect(self._page_changed)
        root.addWidget(self.preview, 1)

        warnings = payload.get("warnings", [])
        warning_text = "\n".join(warnings[:3])
        if len(warnings) > 3:
            warning_text += f"\n{len(warnings) - 3} more historical-data notes (see page details)."
        self.warning_label = QLabel(warning_text)
        self.warning_label.setToolTip("\n".join(warnings))
        self.warning_label.setWordWrap(True)
        self.warning_label.setVisible(bool(payload.get("warnings")))
        root.addWidget(self.warning_label)
        options = QHBoxLayout()
        self.pptx = QCheckBox("PPTX")
        self.pdf = QCheckBox("PDF")
        self.csv = QCheckBox("CSV: all spectra + separate mapping spectra")
        self.images = QCheckBox("Layer images folder")
        self._choices = {"PPTX": self.pptx, "PDF": self.pdf, "CSV": self.csv, "Images": self.images}
        for choice in self._choices.values():
            choice.setChecked(True)
            options.addWidget(choice)
        options.addStretch()
        root.addLayout(options)
        buttons = QHBoxLayout()
        self.path_label = QLabel(str(self.output_dir))
        self.path_label.setWordWrap(True)
        self.choose = QPushButton("Output folder…")
        self.choose.clicked.connect(self.choose_folder)
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.export)
        buttons.addWidget(self.path_label, 1)
        buttons.addWidget(self.choose)
        buttons.addWidget(self.export_button)
        root.addLayout(buttons)
        self.status = QLabel("Preparing preview…")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.setCentralWidget(central)
        QTimer.singleShot(0, self._prepare_preview)

    def _prepare_preview(self):
        if not self._close_pending:
            self._start_job(Path(self._temporary.name), (), preview=True)

    def _zoom_changed(self, index):
        if index == 2:
            self.preview.setZoomMode(QPdfView.ZoomMode.Custom)
            self.preview.setZoomFactor(1.0)
        else:
            self.preview.setZoomMode(QPdfView.ZoomMode.FitInView if index == 0 else QPdfView.ZoomMode.FitToWidth)

    def _change_page(self, offset):
        if self.document.pageCount():
            page = max(0, min(self.document.pageCount() - 1, self.preview.pageNavigator().currentPage() + offset))
            self.preview.pageNavigator().jump(page, QPointF())

    def _page_changed(self, page):
        self.page_label.setText(f"{page + 1} / {self.document.pageCount()}")
        self.previous.setEnabled(page > 0)
        self.next.setEnabled(page < self.document.pageCount() - 1)

    def choose_folder(self):
        selected = QFileDialog.getExistingDirectory(self, "Choose output folder", str(self.output_dir))
        if selected:
            self.output_dir = Path(selected)
            self.path_label.setText(str(self.output_dir))

    def export(self):
        if self.thread is not None:
            return
        formats = [name for name, choice in self._choices.items() if choice.isChecked()]
        if not formats:
            self.status.setText("Select at least one output format")
            return
        self._start_job(self.output_dir, formats)

    def _start_job(self, directory, formats, preview=False):
        if self.thread is not None:
            return
        self.export_button.setEnabled(False)
        self.choose.setEnabled(False)
        for choice in self._choices.values():
            choice.setEnabled(False)
        self.thread = QThread(self)
        self.worker = _ReportWorker(self.payload, directory, formats, preview)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.status.setText)
        self.worker.finished.connect(self._completed)
        self.worker.failed.connect(self._failed)
        self.worker.done.connect(self.thread.quit, Qt.ConnectionType.DirectConnection)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._job_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @Slot(object)
    def _completed(self, result):
        if self._close_pending:
            return
        if "preview" in result:
            self.document.close()
            error = self.document.load(str(result["preview"]))
            if error != QPdfDocument.Error.None_:
                self.status.setText(f"Preview could not be opened: {error.name}")
                return
            self._page_changed(0)
            self.status.setText(f"Preview ready: {self.document.pageCount()} pages. This layout is used for PDF/PPTX export.")
        else:
            self.exported_paths = result["paths"]
            count = len(self.exported_paths)
            self.status.setText(f"Exported {count} files to {result['directory']}"
                                + ("\nFailed: " + "; ".join(result["failures"]) if result["failures"] else ""))

    @Slot(str)
    def _failed(self, message):
        self.status.setText(f"Report failed: {message}")

    @Slot()
    def _job_finished(self):
        self.thread = None
        self.worker = None
        self.export_button.setEnabled(True)
        self.choose.setEnabled(True)
        for choice in self._choices.values():
            choice.setEnabled(True)
        if self._close_pending:
            self.close()

    def closeEvent(self, event):
        self._close_pending = True
        if self.thread is not None:
            self.status.setText("Finishing report output before closing…")
            event.ignore()
            return
        self.document.close()
        self._temporary.cleanup()
        event.accept()
