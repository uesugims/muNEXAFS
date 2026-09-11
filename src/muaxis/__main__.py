"""Command-line entry point used by ``make run``.

The desktop GUI will replace the no-argument message once MainWindow is
implemented.  ``--inspect`` is deliberately useful now: it validates that the
same standalone reader used by the GUI can open an STXM scan.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .io.stxm import StxmFormatError, read_stxm_scan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="muNEXAFS", description="STXM-NEXAFS analysis")
    parser.add_argument("--inspect", type=Path, metavar="SCAN.hdr", help="read and summarize a scan")
    parser.add_argument("--open", type=Path, metavar="SCAN.hdr", help="open a scan in the GUI")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.inspect is None and args.open is None:
        from .gui.main_window import run_gui

        return run_gui()
    if args.inspect is None and args.open is not None:
        from .gui.main_window import run_gui

        return run_gui(args.open)
    try:
        scan = read_stxm_scan(args.inspect)
    except StxmFormatError as exc:
        print(f"Cannot read STXM scan: {exc}")
        return 2

    shape = "not loaded" if scan.shape is None else " × ".join(map(str, scan.shape))
    print(f"Scan: {scan.header.label or scan.header.path.name}")
    print(f"Type: {scan.header.scan_type or 'unknown'}")
    print(f"Frames: {len(scan.header.frames)}")
    print(f"Energy: {scan.energies_eV[0]:g}–{scan.energies_eV[-1]:g} eV")
    print(f"Transmission shape (energy × y × x): {shape}")
    print(f"I0: {'present' if scan.i0 is not None else 'absent'}")
    print(f"Drift: {'present' if scan.shifts_xy is not None else 'absent'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
