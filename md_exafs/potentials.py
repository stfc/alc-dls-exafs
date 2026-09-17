"""Precomputation and distribution of FEFF potentials (ADR 0006).

Self-consistent field (SCF) potential calculation accounts for the majority of
compute time in FEFF runs. Precomputing potentials once on a representative
structure (CONTROL 1 1 1 0 0 0) and reusing them for all trajectory snapshots
(CONTROL 0 0 0 1 1 1) reduces compute time by ~80% with minimal loss of accuracy.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ase import Atoms
from pymatgen.core import Structure

from .feff_input import FeffConfig, build_feff_inp, make_potentials_feff_config

logger = logging.getLogger("md_exafs.potentials")

CANONICAL_POTENTIAL_FILES: tuple[str, ...] = (
    "phase.pad",
    "pot.pad",
    "xsect.json",
    "xsph.json",
    "genfmt.json",
    "ff2x.json",
    "geom.json",
    "atoms.json",
    "pot.json",
    "global.json",
    "path.json",
    "libpotph.json",
    "POTENTIALS",
)


def place_file(src: Path, dst: Path, mode: str = "symlink") -> bool:
    """Place a file at dst using symlink, hardlink, or copy.

    Args:
        src: Source file path.
        dst: Destination file path.
        mode: Placement mode: "symlink", "hardlink", or "copy".

    Returns:
        True if successful.
    """
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()

        if mode == "hardlink":
            try:
                os.link(src, dst)
                return True
            except OSError:
                shutil.copy2(src, dst)
                return True
        elif mode == "symlink":
            os.symlink(src.resolve(), dst)
            return True
        else:  # copy
            shutil.copy2(src, dst)
            return True
    except OSError as exc:
        logger.error(f"Failed to place {src.name} -> {dst} (mode={mode}): {exc}")
        return False


class PotentialsManager:
    """Manager for FEFF potential precomputation, verification, and distribution."""

    MANIFEST: tuple[str, ...] = CANONICAL_POTENTIAL_FILES

    @classmethod
    def get_manifest_files(cls, directory: Path) -> list[Path]:
        """Return list of existing manifest files in a directory."""
        directory = Path(directory)
        return [directory / f for f in cls.MANIFEST if (directory / f).exists()]

    @classmethod
    def precompute(
        cls,
        structure: Atoms | Structure,
        absorber_idx: int,
        config: FeffConfig,
        output_dir: Path | str,
        feff_executable: str = "feff",
        timeout: float = 600.0,
    ) -> Path:
        """Precompute potentials for an absorber site on a representative structure.

        Args:
            structure: ASE Atoms or pymatgen Structure.
            absorber_idx: 0-based index of the absorbing atom.
            config: Base FeffConfig.
            output_dir: Directory where potentials will be calculated and stored.
            feff_executable: Name or path of the FEFF executable.
            timeout: Subprocess timeout in seconds.

        Returns:
            Path to the output directory containing the generated potentials.
        """
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        pot_config = make_potentials_feff_config(config)
        build_feff_inp(
            structure,
            config=pot_config,
            absorber_idx=absorber_idx,
            output_path=out_dir / "feff.inp",
        )

        logger.info(f"Running potentials precompute in {out_dir}")
        res = subprocess.run(
            [feff_executable],
            cwd=out_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if res.returncode != 0:
            msg = (
                f"FEFF precompute failed (returncode {res.returncode}):\n"
                f"{res.stderr}\n{res.stdout}"
            )
            raise RuntimeError(msg)

        # Verify critical files
        critical = ("phase.pad", "pot.pad")
        missing = [f for f in critical if not (out_dir / f).exists()]
        if missing:
            msg = f"FEFF potential precomputation did not produce required files: {missing}"
            raise FileNotFoundError(msg)

        return out_dir

    @classmethod
    def link_to_dir(
        cls,
        potentials_dir: Path | str,
        target_dir: Path | str,
        mode: str = "symlink",
    ) -> int:
        """Place all existing potential files from potentials_dir into target_dir.

        Returns:
            Number of files placed.
        """
        pot_dir = Path(potentials_dir).resolve()
        dst_dir = Path(target_dir).resolve()
        dst_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for fname in cls.MANIFEST:
            src = pot_dir / fname
            if src.exists():
                if place_file(src, dst_dir / fname, mode=mode):
                    count += 1
        return count

    @classmethod
    def distribute(
        cls,
        potentials_dir: Path | str,
        target_dirs: list[Path | str],
        mode: str = "symlink",
        n_workers: int = 8,
    ) -> int:
        """Distribute potential files to multiple worker task directories."""
        pot_dir = Path(potentials_dir).resolve()
        targets = [Path(t).resolve() for t in target_dirs]

        if not targets:
            return 0

        def _worker(t: Path) -> int:
            return cls.link_to_dir(pot_dir, t, mode=mode)

        if len(targets) == 1 or n_workers <= 1:
            return sum(_worker(t) for t in targets)

        total_files = 0
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for count in executor.map(_worker, targets):
                total_files += count
        return total_files


__all__ = [
    "CANONICAL_POTENTIAL_FILES",
    "PotentialsManager",
    "place_file",
]
