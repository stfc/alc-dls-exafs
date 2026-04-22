#!/usr/bin/env python3
"""Pack an existing pipeline_output/ directory tree into a single HDF5 file.

Usage
-----
    python scripts/pack_output_to_hdf5.py pipeline_output/
    python scripts/pack_output_to_hdf5.py pipeline_output/ --hdf5 results.h5
    python scripts/pack_output_to_hdf5.py pipeline_output/ --keep-paths

The directory is expected to contain frame_XXXX/site_XXXX/ sub-directories,
each with a chi.dat file written by a previous pipeline run.  feffNNNN.dat
and files.dat are imported too when --keep-paths is given.

No caching or FEFF execution is involved; this is purely a read-and-pack step.
"""

import argparse
import logging
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack a pipeline_output/ directory tree into a single HDF5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Root directory containing frame_*/site_* sub-directories.",
    )
    parser.add_argument(
        "--hdf5",
        dest="hdf5_path",
        type=Path,
        default=None,
        metavar="FILE",
        help="Destination HDF5 file (default: <output_dir>/results.h5).",
    )
    parser.add_argument(
        "--keep-paths",
        action="store_true",
        default=False,
        help=(
            "Also import per-path feffNNNN.dat files and files.dat "
            "(needed for --plot-include paths / --min-cw-ratio later)."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Show debug-level log messages.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    output_dir: Path = args.output_dir.resolve()
    if not output_dir.exists():
        print(f"Error: directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    hdf5_path: Path = args.hdf5_path or (output_dir / "results.h5")

    frame_dirs = sorted(output_dir.glob("frame_*"))
    if not frame_dirs:
        print(
            f"Error: no frame_* directories found in {output_dir}", file=sys.stderr
        )
        sys.exit(1)

    print(f"Input:       {output_dir}  ({len(frame_dirs)} frame dirs found)")
    print(f"Output HDF5: {hdf5_path}")
    print(f"Store paths: {args.keep_paths}")
    print()

    from larch_cli_wrapper.hdf5_store import ExafsHDF5Store

    store = ExafsHDF5Store.from_existing_output_dir(
        output_dir=output_dir,
        hdf5_path=hdf5_path,
        store_paths=args.keep_paths,
    )
    info = store.info()
    store.close()

    print(
        f"Done.  {info.get('n_sites', '?')} sites imported "
        f"across {info.get('n_frames', '?')} frames."
    )
    print(f"File size: {info.get('file_size_mb', 0):.2f} MB")
    print()
    print("Suggested next step:")
    print(
        f"  larch-cli analyze {hdf5_path} --plot-include average,paths"
        + (" --min-cw-ratio 5.0" if args.keep_paths else "")
    )


if __name__ == "__main__":
    main()
