"""Read UVSOR STXM/NEXAFS ``.hdr`` and tab-delimited ``.xim`` files.

This module intentionally has no GUI dependency.  It may be used from batch
scripts, notebooks, tests, and the desktop application alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

import numpy as np
from numpy.typing import NDArray


class StxmFormatError(ValueError):
    """Raised when an STXM dataset is incomplete or structurally invalid."""


@dataclass(frozen=True, slots=True)
class FrameMetadata:
    """Metadata recorded for one energy frame in a UVSOR header."""

    index: int
    energy_eV: float
    acquired_at: str | None
    storage_ring_current: float | None
    zone_plate_destination: float | None
    zone_plate_error: float | None


@dataclass(frozen=True, slots=True)
class ScanHeader:
    """Header metadata needed to identify a scan and its physical axes."""

    path: Path
    label: str | None
    scan_type: str | None
    stack_axis_energy_eV: NDArray[np.float64]
    x_um: NDArray[np.float64] | None
    y_um: NDArray[np.float64] | None
    frames: tuple[FrameMetadata, ...]
    dwell_time_ms: float | None = None


@dataclass(frozen=True, slots=True)
class ScanStack:
    """An STXM image stack plus optional companion normalization data.

    ``transmission`` is ordered as ``(energy, y, x)``.  The reader never
    applies source-current normalization or drift registration: these are
    analysis decisions and must remain explicit in the processing layer.
    """

    header: ScanHeader
    frame_paths: tuple[Path, ...]
    transmission: NDArray[np.float64] | None
    i0: NDArray[np.float64] | None
    shifts_xy: NDArray[np.float64] | None

    @property
    def energies_eV(self) -> NDArray[np.float64]:
        return np.asarray([frame.energy_eV for frame in self.header.frames])

    @property
    def shape(self) -> tuple[int, int, int] | None:
        return None if self.transmission is None else self.transmission.shape


_AXIS_RE = re.compile(
    r'(?P<axis>StackAxis|PAxis|QAxis)\s*=\s*\{.*?Points\s*=\s*\((?P<points>.*?)\);',
    re.DOTALL,
)
_IMAGE_RE = re.compile(r'Image(?P<index>\d+)_0\s*=\s*\{(?P<body>.*?)\};', re.DOTALL)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?")


def _numbers(value: str) -> NDArray[np.float64]:
    return np.asarray([float(item) for item in _NUMBER_RE.findall(value)], dtype=np.float64)


def _field(body: str, name: str) -> str | None:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*(?:\"(?P<quoted>[^\"]*)\"|(?P<plain>[^;}}]+))", body)
    if match is None:
        return None
    return (match.group("quoted") or match.group("plain")).strip()


def _float_field(body: str, name: str) -> float | None:
    value = _field(body, name)
    return None if value is None else float(value)


def _axis_values(text: str, axis_name: str) -> NDArray[np.float64] | None:
    for match in _AXIS_RE.finditer(text):
        if match.group("axis") != axis_name:
            continue
        values = _numbers(match.group("points"))
        if len(values) < 2:
            raise StxmFormatError(f"{axis_name} has no coordinate values")
        # The first value is the format's declared point count, not a coordinate.
        declared_count = int(values[0])
        coordinates = values[1:]
        if len(coordinates) != declared_count:
            raise StxmFormatError(
                f"{axis_name} declares {declared_count} points but contains {len(coordinates)}"
            )
        return coordinates
    return None


def read_header(path: str | Path) -> ScanHeader:
    """Parse a UVSOR STXM header without reading its image payload."""

    header_path = Path(path)
    if header_path.suffix.lower() != ".hdr":
        raise StxmFormatError(f"Expected a .hdr file, got {header_path.name!r}")
    try:
        text = header_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = header_path.read_text(encoding="latin-1")

    label = _field(text, "Label")
    scan_type = _field(text, "Type")
    stack_axis = _axis_values(text, "StackAxis")
    if stack_axis is None:
        raise StxmFormatError("Header has no StackAxis")

    frames: list[FrameMetadata] = []
    for match in _IMAGE_RE.finditer(text):
        body = match.group("body")
        energy = _float_field(body, "Energy")
        if energy is None:
            raise StxmFormatError(f"Image{match.group('index')}_0 has no Energy")
        frames.append(
            FrameMetadata(
                index=int(match.group("index")),
                energy_eV=energy,
                acquired_at=_field(body, "Time"),
                storage_ring_current=_float_field(body, "StorageRingCurrent"),
                zone_plate_destination=_float_field(body, "ZP_dest"),
                zone_plate_error=_float_field(body, "ZP_error"),
            )
        )
    frames.sort(key=lambda frame: frame.index)
    if not frames:
        raise StxmFormatError("Header contains no ImageNNN_0 frame records")
    if len(frames) != len(stack_axis):
        raise StxmFormatError(
            f"StackAxis has {len(stack_axis)} values but header has {len(frames)} image records"
        )

    return ScanHeader(
        path=header_path,
        label=label,
        scan_type=scan_type,
        stack_axis_energy_eV=stack_axis,
        x_um=_axis_values(text, "PAxis"),
        y_um=_axis_values(text, "QAxis"),
        frames=tuple(frames),
        dwell_time_ms=(
            _float_field(text, "DwellTime")
            if _field(text, "DwellTime") is not None
            else (_float_field(text, "Dwell") if _field(text, "Dwell") is not None else None)
        ),
    )


def _frame_path(header: ScanHeader, frame: FrameMetadata) -> Path:
    stem = header.path.stem
    return header.path.parent / f"{stem}_a{frame.index:03d}.xim"


def _read_i0(path: Path, expected_count: int) -> NDArray[np.float64]:
    try:
        values = np.loadtxt(path, delimiter=",", usecols=(0,), ndmin=1, dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise StxmFormatError(f"Could not read I0 data from {path}: {exc}") from exc
    if len(values) != expected_count:
        raise StxmFormatError(f"I0 has {len(values)} rows; expected {expected_count}")
    return values


def _read_shifts(path: Path, expected_count: int) -> NDArray[np.float64]:
    try:
        values = np.loadtxt(path, delimiter=",", ndmin=2, dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise StxmFormatError(f"Could not read drift data from {path}: {exc}") from exc
    if values.shape != (expected_count, 2):
        raise StxmFormatError(f"Drift data has shape {values.shape}; expected ({expected_count}, 2)")
    return values


def _read_image(path: Path) -> NDArray[np.float64]:
    """Read a tab-delimited image, tolerating an empty trailing tab column.

    UVSOR files commonly terminate every row with a tab.  ``loadtxt`` treats
    that as a non-numeric final column, while Xojo's field parser silently
    ignores it.  Only a wholly empty final column is discarded; incomplete
    pixels elsewhere remain a format error.
    """

    try:
        matrix = np.genfromtxt(path, delimiter="\t", ndmin=2, dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise StxmFormatError(f"Could not read image {path}: {exc}") from exc
    if matrix.size == 0:
        raise StxmFormatError(f"Image is empty: {path}")
    if matrix.shape[1] > 1 and np.isnan(matrix[:, -1]).all():
        matrix = matrix[:, :-1]
    if not np.isfinite(matrix).all():
        invalid = int(np.size(matrix) - np.isfinite(matrix).sum())
        raise StxmFormatError(f"Image {path.name} contains {invalid} missing or non-finite pixels")
    return matrix


def read_stxm_scan(
    header_path: str | Path,
    *,
    load_images: bool = True,
    load_i0: bool = True,
    load_shifts: bool = True,
    missing_companion: Literal["ignore", "error"] = "ignore",
) -> ScanStack:
    """Read one UVSOR STXM scan directory.

    Parameters
    ----------
    header_path:
        A ``.hdr`` file.  Its sibling ``<stem>_aNNN.xim`` files are located
        using the numeric image records in the header, not directory ordering.
    load_images:
        Set false for quick metadata-only cataloguing.
    missing_companion:
        Controls whether absent optional ``i0.txt`` / ``drift.txt`` files are
        ignored or reported as an error.
    """

    header = read_header(header_path)
    frame_paths = tuple(_frame_path(header, frame) for frame in header.frames)
    missing = [path.name for path in frame_paths if not path.is_file()]
    if missing:
        preview = ", ".join(missing[:3])
        suffix = "..." if len(missing) > 3 else ""
        raise StxmFormatError(f"Missing {len(missing)} frame image(s): {preview}{suffix}")

    transmission: NDArray[np.float64] | None = None
    if load_images:
        matrices: list[NDArray[np.float64]] = []
        expected_shape: tuple[int, int] | None = None
        for path in frame_paths:
            matrix = _read_image(path)
            if expected_shape is None:
                expected_shape = matrix.shape
            elif matrix.shape != expected_shape:
                raise StxmFormatError(
                    f"Inconsistent image shape in {path.name}: {matrix.shape}, expected {expected_shape}"
                )
            matrices.append(matrix)
        transmission = np.stack(matrices)

    def companion(name: str, loader: object) -> NDArray[np.float64] | None:
        path = header.path.parent / name
        if not path.is_file():
            if missing_companion == "error":
                raise StxmFormatError(f"Required companion file is absent: {path}")
            return None
        return loader(path, len(header.frames))  # type: ignore[operator]

    i0 = companion("i0.txt", _read_i0) if load_i0 else None
    shifts = companion("drift.txt", _read_shifts) if load_shifts else None
    return ScanStack(header, frame_paths, transmission, i0, shifts)
