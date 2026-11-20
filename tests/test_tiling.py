"""Orbital-tiling integration at the AiiDA/jQMC boundary."""

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

import pathlib
import subprocess

import h5py
import numpy as np
import pytest
from aiida import orm
from aiida.engine import run_get_node
from aiida.plugins import CalculationFactory

from aiida_jqmc.calculations.jqmc import is_twist_average
from aiida_jqmc.tiling import inspect_tiled_hamiltonian

from .conftest import DATA_DIR, assert_complete, which_or_skip

JqmcCalculation = CalculationFactory("jqmc.jqmc")
JqmcToolCalculation = CalculationFactory("jqmc.tool")
TILING_DATA = DATA_DIR / "tiling"
SERIAL_OPTIONS = {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1}, "max_wallclock_seconds": 3600}

pytestmark = pytest.mark.slow


def _converter(executable: str, source: pathlib.Path, output: pathlib.Path, *selection: str, kind: str | None = None):
    command = [executable]
    if source.suffix == ".h5":
        command.extend(["trexio", "convert-to", str(source)])
    else:
        command.extend(["qe", str(source)])
    if kind is not None:
        command.extend(["--kind", kind])
    command.extend(["--tiling", "2 1 1", *selection, "-j1", "0.5", "-j2", "0.7", "-j3", "ao-small", "-o", str(output)])
    return subprocess.run(command, check=True, capture_output=True, text=True)


@pytest.fixture(scope="module")
def tiled_gamma_path(tmp_path_factory):
    output = tmp_path_factory.mktemp("tiling") / "gamma.h5"
    completed = _converter(
        which_or_skip("jqmc-tool"),
        TILING_DATA / "trexio" / "lithium_tiling_sectors.h5",
        output,
        "--supertwist",
        "0 0 0",
    )
    return output, completed.stdout


@pytest.fixture(scope="module")
def tiled_twist_path(tmp_path_factory):
    """The two-sector twist-averaged ensemble of the same tiling."""
    output = tmp_path_factory.mktemp("tiling") / "twists.h5"
    _converter(which_or_skip("jqmc-tool"), TILING_DATA / "trexio" / "lithium_tiling_sectors.h5", output, "--twist-average")
    return output


def test_inspect_tiled_hamiltonian_uses_physical_supercell(tiled_gamma_path):
    path, stdout = tiled_gamma_path
    metadata, structure = inspect_tiled_hamiltonian(path, stdout=stdout)
    assert metadata["tiling_matrix"] == [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert metadata["supertwists"] == [[0.0, 0.0, 0.0]]
    assert metadata["physical_system"] == {
        "number_of_atoms": 4,
        "number_of_electrons_up": 2,
        "number_of_electrons_down": 2,
    }
    assert metadata["j3_basis"]["number_of_atoms"] == 4
    assert metadata["j3_basis"]["orbital_type"] == "AOs_sphe_pgto_data"
    assert metadata["jqmc"]["git_commit"] != "unknown"
    assert len(structure.sites) == 4
    np.testing.assert_allclose(structure.cell_lengths, [7.02, 3.51, 3.51], atol=1.0e-10)


@pytest.mark.parametrize("flag", ["opt_geminal", "opt_lambda_basis_exp", "opt_lambda_basis_coeff"])
def test_tiled_vmc_rejects_slater_optimization(aiida_code_installed, aiida_localhost, tiled_gamma_path, flag):
    code = aiida_code_installed(filepath_executable="/bin/true", computer=aiida_localhost, with_mpi=True)
    hamiltonian = orm.SinglefileData(tiled_gamma_path[0])
    parameters = {
        "control": {"job_type": "vmc"},
        "vmc": {"num_mcmc_steps": 1, "num_opt_steps": 1, flag: True},
    }
    with pytest.raises(ValueError, match="Tiled Slater orbitals are frozen"):
        run_get_node(
            JqmcCalculation,
            code=code,
            hamiltonian=hamiltonian,
            parameters=orm.Dict(parameters),
            metadata={"options": SERIAL_OPTIONS},
        )


def test_tiled_vmc_keeps_j3_optimization_available(aiida_code_installed, aiida_localhost, tiled_gamma_path):
    code = aiida_code_installed(filepath_executable="/bin/true", computer=aiida_localhost, with_mpi=True)
    parameters = {
        "control": {"job_type": "vmc"},
        "vmc": {
            "num_mcmc_steps": 1,
            "num_opt_steps": 1,
            "opt_geminal": False,
            "opt_lambda_basis_exp": False,
            "opt_lambda_basis_coeff": False,
            "opt_J3_param": True,
            "opt_J3_basis_exp": True,
            "opt_J3_basis_coeff": True,
        },
    }
    _, node = run_get_node(
        JqmcCalculation,
        code=code,
        hamiltonian=orm.SinglefileData(tiled_gamma_path[0]),
        parameters=orm.Dict(parameters),
        metadata={"dry_run": True, "store_provenance": False, "options": SERIAL_OPTIONS},
    )
    assert pathlib.Path(node.dry_run_info["folder"], "aiida.toml").exists()


def _conversion_inputs(source: str, data: pathlib.Path):
    if source == "qe":
        return {
            "command": orm.Str("qe"),
            "input_folder": orm.FolderData(tree=data / "tiling" / "qe"),
            "save_path": orm.Str("out/li_tiling.save"),
        }
    filename = "lithium_tiling_sectors_cart.h5" if source == "trexio-cart" else "lithium_tiling_sectors.h5"
    return {
        "command": orm.Str("trexio convert-to"),
        "input_file": orm.SinglefileData(data / "tiling" / "trexio" / filename),
    }


@pytest.mark.parametrize(
    ("source", "selection", "kind", "mode", "sectors"),
    [
        ("qe", {"supertwist": [0.0, 0.0, 0.0]}, "bspline", "single", 1),
        ("qe", {"supertwist": [0.22, 0.13, 0.09]}, "pgto", "single", 1),
        ("qe", {"twist_average": True}, "pgto+pw", "ensemble", 2),
        ("trexio-sphe", {"supertwist": [0.0, 0.0, 0.0]}, None, "single", 1),
        ("trexio-cart", {"supertwist": [0.22, 0.13, 0.09]}, None, "single", 1),
        ("trexio-sphe", {"twist_average": True}, None, "ensemble", 2),
    ],
)
def test_six_source_sector_routes(jqmc_tool_code, data_with_pseudopotentials, source, selection, kind, mode, sectors):
    options = {"-j1": 0.5, "-j2": 0.7, "-j3": "ao-small"}
    if kind is not None:
        options["--kind"] = kind
    typed = {key: orm.Bool(value) if key == "twist_average" else orm.List(list=value) for key, value in selection.items()}
    _, node = run_get_node(
        JqmcToolCalculation,
        code=jqmc_tool_code,
        tiling=orm.List(list=[2, 1, 1]),
        options=orm.Dict(options),
        **typed,
        **_conversion_inputs(source, data_with_pseudopotentials),
    )
    assert node.is_finished_ok, (node.exit_status, node.exit_message)
    metadata = node.outputs.conversion_metadata.get_dict()
    assert metadata["sector_mode"] == mode
    assert metadata["number_of_sectors"] == sectors
    assert metadata["physical_system"]["number_of_atoms"] == len(node.outputs.physical_structure.sites) == 4
    assert metadata["jqmc"]["version"] != "unknown"
    assert metadata["jqmc"]["git_commit"] != "unknown"
    if mode == "ensemble":
        np.testing.assert_allclose(metadata["normalized_sector_weights"], [0.2, 0.8])


def test_qe_pw_representation_is_transmitted(jqmc_tool_code, data_with_pseudopotentials):
    _, node = run_get_node(
        JqmcToolCalculation,
        code=jqmc_tool_code,
        tiling=orm.List(list=[2, 1, 1]),
        supertwist=orm.List(list=[0.0, 0.0, 0.0]),
        options=orm.Dict({"--kind": "pw", "-j2": 0.7}),
        **_conversion_inputs("qe", data_with_pseudopotentials),
    )
    assert node.is_finished_ok, (node.exit_status, node.exit_message)
    backends = node.outputs.conversion_metadata["representation"]["primitive_block_backends"]
    assert set(backends) == {"AOs_plane_wave_complex_data"}


def test_converter_coset_error_is_preserved(jqmc_tool_code):
    _, node = run_get_node(
        JqmcToolCalculation,
        code=jqmc_tool_code,
        command=orm.Str("trexio convert-to"),
        input_file=orm.SinglefileData(TILING_DATA / "trexio" / "lithium_tiling_sectors.h5"),
        tiling=orm.List(list=[3, 1, 1]),
        supertwist=orm.List(list=[0.0, 0.0, 0.0]),
    )
    assert node.exit_status == JqmcToolCalculation.exit_codes.ERROR_TILING_CONVERSION_FAILED.status
    assert any(word in node.exit_message.lower() for word in ("coset", "compatible", "sector", "primitive k points"))


def test_tiled_ensemble_vmc_keeps_the_ensemble(jqmc_tool_code, jqmc_code):
    """A tiled two-sector ensemble optimized by VMC (J3 included, one MPI rank per sector): the run is complete and
    its output Hamiltonian is again a combined ensemble, the input of a fresh MCMC or LRDMC."""
    _, conversion = run_get_node(
        JqmcToolCalculation,
        code=jqmc_tool_code,
        tiling=orm.List(list=[2, 1, 1]),
        twist_average=orm.Bool(True),
        options=orm.Dict({"-j1": 0.5, "-j2": 0.7, "-j3": "ao-small"}),
        **_conversion_inputs("trexio-sphe", DATA_DIR),
    )
    assert conversion.is_finished_ok, (conversion.exit_status, conversion.exit_message)
    vmc = {
        "control": {"job_type": "vmc", "number_of_walkers": 2, "mcmc_seed": 1},
        "vmc": {
            "num_mcmc_steps": 40,
            "num_mcmc_warmup_steps": 10,
            "num_mcmc_bin_blocks": 5,
            "num_opt_steps": 1,
            "opt_J3_param": True,
        },
    }
    options = {**SERIAL_OPTIONS, "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 2}}
    _, node = run_get_node(
        JqmcCalculation,
        code=jqmc_code,
        hamiltonian=conversion.outputs.output_hamiltonian,
        parameters=orm.Dict(vmc),
        metadata={"options": options},
    )
    assert len(assert_complete(node, "vmc")["steps"]) == 1
    assert is_twist_average(node.outputs.output_hamiltonian)


MCMC_SECTION = {"num_mcmc_steps": 40, "num_mcmc_warmup_steps": 10, "num_mcmc_bin_blocks": 5}


@pytest.mark.parametrize(
    ("hamiltonian", "job_type", "section"),
    [
        ("gamma", "mcmc", MCMC_SECTION),
        ("twists", "mcmc", MCMC_SECTION),
        (
            "gamma",
            "lrdmc-bra",
            {
                "num_mcmc_steps": 40,
                "alat": 0.3,
                "num_mcmc_per_measurement": 4,
                "num_gfmc_warmup_steps": 25,
                "num_gfmc_bin_blocks": 10,
                "num_gfmc_collect_steps": 5,
            },
        ),
    ],
)
def test_tiled_sampling_reports_the_tiled_orbitals(request, jqmc_code, options, hamiltonian, job_type, section):
    """MCMC and LRDMC on a tiled Gamma Hamiltonian, and MCMC on the tiled twist-averaged ensemble (2 MPI ranks, one
    per twist), are complete, and the system summary lists the tiled orbitals of both spin channels."""
    path = (
        request.getfixturevalue("tiled_gamma_path")[0]
        if hamiltonian == "gamma"
        else request.getfixturevalue("tiled_twist_path")
    )
    parameters = {"control": {"job_type": job_type, "number_of_walkers": 2, "mcmc_seed": 1}, job_type: section}
    _, node = run_get_node(
        JqmcCalculation,
        code=jqmc_code,
        hamiltonian=orm.SinglefileData(path),
        parameters=orm.Dict(parameters),
        metadata={"options": options},
    )
    out = assert_complete(node, job_type)
    assert (out["energy"] if hamiltonian == "gamma" else out["twist_average"]["energy"]) is not None
    tiled = out["system"]["tiled_orbitals"]
    assert [orbitals["spin"] for orbitals in tiled] == ["up", "down"]
    assert all(orbitals["tiling"] == [[2, 0, 0], [0, 1, 0], [0, 0, 1]] for orbitals in tiled)


def _assert_hdf_equal(left: h5py.Group, right: h5py.Group):
    assert set(left.attrs) == set(right.attrs)
    for key in left.attrs:
        np.testing.assert_equal(left.attrs[key], right.attrs[key])
    assert set(left) == set(right)
    for key in left:
        if isinstance(left[key], h5py.Group):
            _assert_hdf_equal(left[key], right[key])
        else:
            np.testing.assert_equal(left[key][()], right[key][()])


def test_aiida_and_direct_cli_outputs_match_by_hdf_content(jqmc_tool_code, tmp_path):
    source = TILING_DATA / "trexio" / "lithium_tiling_sectors.h5"
    direct = tmp_path / "direct.h5"
    _converter(which_or_skip("jqmc-tool"), source, direct, "--supertwist", "0.22 0.13 0.09")
    _, node = run_get_node(
        JqmcToolCalculation,
        code=jqmc_tool_code,
        command=orm.Str("trexio convert-to"),
        input_file=orm.SinglefileData(source),
        tiling=orm.List(list=[2, 1, 1]),
        supertwist=orm.List(list=[0.22, 0.13, 0.09]),
        options=orm.Dict({"-j1": 0.5, "-j2": 0.7, "-j3": "ao-small"}),
    )
    assert node.is_finished_ok, (node.exit_status, node.exit_message)
    with (
        node.outputs.output_hamiltonian.as_path() as aiida_path,
        h5py.File(aiida_path) as aiida_hdf,
        h5py.File(direct) as direct_hdf,
    ):
        _assert_hdf_equal(aiida_hdf, direct_hdf)
