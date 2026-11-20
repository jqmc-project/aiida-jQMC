"""Read orbital-tiling provenance from a Hamiltonian produced by jQMC.

The converter remains the sole owner of coset selection, orbital phases,
normalization and supercell construction.  This module only asks jQMC to
deserialize its own output and translates the resulting physical structure
and metadata into AiiDA data types.
"""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the jqmc project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from aiida import orm
from aiida.common.constants import elements
from jqmc._twist_average import is_twist_average_input, load_twist, load_twist_average_input, normalize_twist_weights
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.orbital_tiling import TiledOrbitalSet_data

__all__ = ("inspect_tiled_hamiltonian", "is_tiled_hamiltonian")

_VERSION = re.compile(r"^version\s*=\s*(.+)$", re.MULTILINE)
_COMMIT = re.compile(r"^git commit\s*=\s*(.+)$", re.MULTILINE)
# CODATA 2018, in Angstrom. jQMC stores all structure coordinates in Bohr.
_BOHR_TO_ANGSTROM = 0.529177210903


def _first_hamiltonian(path: str | Path) -> tuple[Hamiltonian_data, dict[str, Any]]:
    """Load one physical Hamiltonian and the sector header through jQMC APIs."""
    if is_twist_average_input(str(path)):
        header = load_twist_average_input(str(path))
        weights = np.asarray(header.weights, dtype=float)
        normalized = normalize_twist_weights(weights)
        sectors = [None if point is None else [float(value) for value in point] for point in header.k_points]
        return load_twist(str(path), 0), {
            "sector_mode": "ensemble",
            "number_of_sectors": int(header.n_twist),
            "supertwists": sectors,
            "sector_weights": weights.tolist(),
            "normalized_sector_weights": normalized.tolist(),
            "bz_weight_sum": float(header.bz_weight_sum),
        }

    hamiltonian = Hamiltonian_data.load_from_hdf5(path)
    orbitals = hamiltonian.wavefunction_data.geminal_data.orb_data_up_spin
    # Only tiled orbitals carry a supertwist.
    sectors = [[float(value) for value in orbitals.supertwist]] if isinstance(orbitals, TiledOrbitalSet_data) else []
    return hamiltonian, {
        "sector_mode": "single",
        "number_of_sectors": 1,
        "supertwists": sectors,
        "sector_weights": [1.0],
        "normalized_sector_weights": [1.0],
        "bz_weight_sum": 1.0,
    }


def is_tiled_hamiltonian(path: str | Path) -> bool:
    """Return whether the determinant uses jQMC's tiled orbital container."""
    hamiltonian, _ = _first_hamiltonian(path)
    orbitals = hamiltonian.wavefunction_data.geminal_data.orb_data_up_spin
    return isinstance(orbitals, TiledOrbitalSet_data)


def _physical_structure(hamiltonian: Hamiltonian_data) -> orm.StructureData:
    structure = hamiltonian.structure_data
    if not structure.pbc_flag:
        raise ValueError("A tiled Hamiltonian must contain a periodic physical supercell.")
    result = orm.StructureData(cell=(np.asarray(structure.cell, dtype=float) * _BOHR_TO_ANGSTROM).tolist())
    for number, position in zip(structure.atomic_numbers, structure.positions, strict=True):
        result.append_atom(
            position=(np.asarray(position, dtype=float) * _BOHR_TO_ANGSTROM).tolist(),
            symbols=elements[int(number)]["symbol"],
        )
    return result


def _build_info(path: str | Path, stdout: str) -> dict[str, str]:
    """Prefer immutable file metadata and use the converter banner for a combined twist input."""
    with h5py.File(path, "r") as handle:
        version = handle.attrs.get("jqmc_version")
        commit = handle.attrs.get("jqmc_git_commit")
    if isinstance(version, bytes):
        version = version.decode()
    if isinstance(commit, bytes):
        commit = commit.decode()
    version_match = _VERSION.search(stdout)
    commit_match = _COMMIT.search(stdout)
    return {
        "version": str(version or (version_match.group(1).strip() if version_match else "unknown")),
        "git_commit": str(commit or (commit_match.group(1).strip() if commit_match else "unknown")),
    }


def inspect_tiled_hamiltonian(
    path: str | Path,
    *,
    stdout: str = "",
    source: dict[str, Any] | None = None,
    requested: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], orm.StructureData]:
    """Return converter-owned tiling metadata and the physical QMC structure.

    Loading and validation are delegated to the jQMC release that parses the
    HDF5.  A non-tiled result is an error because this function is called only
    for a conversion that explicitly requested ``tiling``.
    """
    hamiltonian, sector = _first_hamiltonian(path)
    geminal = hamiltonian.wavefunction_data.geminal_data
    orbitals = geminal.orb_data_up_spin
    if not isinstance(orbitals, TiledOrbitalSet_data):
        raise ValueError("jQMC returned an untiled determinant for a conversion that requested `tiling`.")

    jastrow = hamiltonian.wavefunction_data.jastrow_data
    j3 = None if jastrow is None else jastrow.jastrow_three_body_data
    j3_metadata: dict[str, Any] = {"present": j3 is not None}
    if j3 is not None:
        j3_metadata.update(
            {
                "orbital_type": type(j3.orb_data).__name__,
                "number_of_atoms": int(j3.orb_data.structure_data.natom),
                "coupling_matrix_shape": [int(size) for size in np.shape(j3.j_matrix)],
            }
        )

    metadata: dict[str, Any] = {
        "format": "jqmc-orbital-tiling-v1",
        "jqmc": _build_info(path, stdout),
        "source": source or {},
        "requested": requested or {},
        "tiling_matrix": [[int(value) for value in row] for row in orbitals.tiling_matrix],
        "representation": {
            "geminal_type": type(geminal).__name__,
            "orbital_type": type(orbitals).__name__,
            "primitive_block_backends": [type(block.backend).__name__ for block in orbitals.blocks],
            "number_of_primitive_blocks": len(orbitals.blocks),
            "normalization": orbitals.normalization,
        },
        "physical_system": {
            "number_of_atoms": int(hamiltonian.structure_data.natom),
            "number_of_electrons_up": int(geminal.num_electron_up),
            "number_of_electrons_down": int(geminal.num_electron_dn),
        },
        "j3_basis": j3_metadata,
        **sector,
    }
    return metadata, _physical_structure(hamiltonian)
