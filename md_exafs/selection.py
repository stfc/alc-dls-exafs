"""Absorber and trajectory selection utilities.

Implements generalized per-frame absorber resolution (ADR 0008).
"""

from __future__ import annotations


def resolve_frame_absorbers(
    symbols: list[str],
    spec: str | int | list[int] | tuple[int, ...],
) -> list[int]:
    """Resolve absorber indices for a single frame.

    Enforces single-species consistency per frame (ADR 0008).

    Args:
        symbols: List of chemical element symbols for each atom in the frame.
        spec: Absorber specification:
            - str: Chemical symbol (e.g. 'Fe'), comma-separated indices ('0,1,2'),
              or element-relative indices ('Fe:0,1').
            - int: Single atom index.
            - list[int] / tuple[int, ...]: Explicit list of atom indices.

            Absolute indices may be negative, with Python semantics (``-1`` is the
            last atom). Element-relative indices (the part after ``:``) may not.
            Callers that need the stored specification to match the resolved index
            exactly — e.g. provenance-tracking front-ends — should reject negative
            values before calling.

    Returns:
        List of 0-based atom indices for absorbing atoms, sorted and de-duplicated.

    Raises:
        ValueError: If indices are out of bounds, species are mixed,
            or element is not found.
        TypeError: If ``spec`` is not a str, int, list or tuple.
    """
    n_atoms = len(symbols)
    if n_atoms == 0:
        raise ValueError("Cannot resolve absorbers for an empty structure (0 atoms).")

    if isinstance(spec, int):
        idx = spec if spec >= 0 else n_atoms + spec
        if not (0 <= idx < n_atoms):
            raise ValueError(f"Absorber index {spec} out of range [0, {n_atoms - 1}].")
        return [idx]

    if isinstance(spec, list | tuple):
        if not spec:
            raise ValueError("Absorber index list must not be empty.")
        indices: list[int] = []
        for raw_i in spec:
            i = int(raw_i)
            idx = i if i >= 0 else n_atoms + i
            if not (0 <= idx < n_atoms):
                raise ValueError(f"Absorber index {i} out of range [0, {n_atoms - 1}].")
            indices.append(idx)
        # Verify single species consistency
        species = {symbols[i].capitalize() for i in indices}
        if len(species) > 1:
            raise ValueError(
                f"All absorber sites must be the same element; "
                f"found multiple elements: {sorted(species)}."
            )
        return sorted(set(indices))

    if isinstance(spec, str):
        cleaned = spec.strip()
        if not cleaned:
            raise ValueError("Absorber specification must not be empty.")

        # Case 1: element-relative selection, e.g. "Fe:0,1"
        if ":" in cleaned:
            elem_part, idx_part = cleaned.split(":", 1)
            target_elem = elem_part.strip().capitalize()
            matching_sites = [
                i for i, sym in enumerate(symbols) if sym.capitalize() == target_elem
            ]
            if not matching_sites:
                raise ValueError(
                    f"No atoms with element '{target_elem}' found in structure."
                )
            if not idx_part.strip():
                return matching_sites
            rel_indices = [int(p.strip()) for p in idx_part.split(",") if p.strip()]
            resolved: list[int] = []
            for r in rel_indices:
                if not (0 <= r < len(matching_sites)):
                    raise ValueError(
                        f"Relative index {r} out of range for element '{target_elem}' "
                        f"({len(matching_sites)} sites available)."
                    )
                resolved.append(matching_sites[r])
            return sorted(set(resolved))

        # Case 2: digits and commas, e.g. "0, 1, 2" or "4"
        if cleaned.replace(",", "").replace(" ", "").lstrip("-").isdigit():
            raw_indices = [int(p.strip()) for p in cleaned.split(",") if p.strip()]
            return resolve_frame_absorbers(symbols, raw_indices)

        # Case 3: Chemical symbol, e.g. "Fe" or "Cu"
        target_elem = cleaned.capitalize()
        matching_sites = [
            i for i, sym in enumerate(symbols) if sym.capitalize() == target_elem
        ]
        if not matching_sites:
            raise ValueError(
                f"No atoms with element '{target_elem}' found in structure."
            )
        return matching_sites

    raise TypeError(f"Invalid absorber specification type: {type(spec).__name__}")


def resolve_trajectory_absorbers(
    trajectory_symbols: list[list[str]],
    spec: str | int | list[int] | tuple[int, ...],
) -> list[list[int]]:
    """Resolve absorber indices across all frames of a trajectory.

    Evaluates the specification against each frame independently, supporting
    both constant-topology and variable/permuted topologies (ADR 0008).
    Enforces species consistency across all frames.

    Args:
        trajectory_symbols: List of symbol lists, one per frame.
        spec: Absorber specification.

    Returns:
        List of absorber index lists (one index list per trajectory frame).

    Raises:
        ValueError: If any frame has no matching absorbers or species differs
            across frames.
    """
    if not trajectory_symbols:
        raise ValueError("Cannot resolve absorbers for an empty trajectory.")

    resolved_trajectory: list[list[int]] = []
    first_species: str | None = None

    for frame_idx, symbols in enumerate(trajectory_symbols):
        frame_absorbers = resolve_frame_absorbers(symbols, spec)
        if not frame_absorbers:
            raise ValueError(f"Frame {frame_idx} resolved to 0 absorbing sites.")

        frame_species = symbols[frame_absorbers[0]].capitalize()
        if first_species is None:
            first_species = frame_species
        elif frame_species != first_species:
            msg = (
                f"Absorbing element species changed across frames: "
                f"frame 0 is '{first_species}' but frame {frame_idx} is '{frame_species}'."
            )
            raise ValueError(msg)
        resolved_trajectory.append(frame_absorbers)

    return resolved_trajectory


def parse_frame_slice(
    slice_spec: str | int | slice | list[int],
    n_frames: int,
) -> list[int]:
    """Parse a frame slice specification into a list of 0-based frame indices.

    Args:
        slice_spec: Slice specification:
            - int: Single frame index.
            - slice: Python slice object.
            - str: "all", ":", "0:10", "::2", "5:20:2", or "0,2,4".
            - list[int]: Explicit frame indices.
        n_frames: Total number of frames in the trajectory.

    Returns:
        List of valid integer frame indices.
    """
    if n_frames <= 0:
        return []

    if isinstance(slice_spec, int):
        idx = slice_spec if slice_spec >= 0 else n_frames + slice_spec
        if not (0 <= idx < n_frames):
            raise ValueError(
                f"Frame index {slice_spec} out of range [0, {n_frames - 1}]."
            )
        return [idx]

    if isinstance(slice_spec, slice):
        return list(range(n_frames)[slice_spec])

    if isinstance(slice_spec, list):
        out = []
        for i in slice_spec:
            idx = i if i >= 0 else n_frames + i
            if not (0 <= idx < n_frames):
                raise ValueError(f"Frame index {i} out of range [0, {n_frames - 1}].")
            out.append(idx)
        return out

    if isinstance(slice_spec, str):
        cleaned = slice_spec.strip()
        if cleaned in ("all", "*", ":"):
            return list(range(n_frames))

        if ":" in cleaned:
            parts = cleaned.split(":")
            if len(parts) > 3:
                raise ValueError(f"Invalid slice syntax: '{slice_spec}'")
            start = int(parts[0].strip()) if parts[0].strip() else None
            stop = (
                int(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else None
            )
            step = (
                int(parts[2].strip()) if len(parts) > 2 and parts[2].strip() else None
            )
            return list(range(n_frames)[slice(start, stop, step)])

        if "," in cleaned:
            indices = [int(p.strip()) for p in cleaned.split(",") if p.strip()]
            return parse_frame_slice(indices, n_frames)

        # Single integer string
        return parse_frame_slice(int(cleaned), n_frames)

    raise TypeError(f"Invalid slice specification type: {type(slice_spec).__name__}")


__all__ = [
    "resolve_frame_absorbers",
    "resolve_trajectory_absorbers",
    "parse_frame_slice",
]
