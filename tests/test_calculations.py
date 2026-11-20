"""Tests for JqmcCalculation: input validation and the generated input files."""

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

import pytest
import toml
from aiida import orm
from aiida.engine import run_get_node
from aiida.plugins import CalculationFactory

JqmcCalculation = CalculationFactory("jqmc.jqmc")

MCMC = {
    "control": {"job_type": "mcmc", "number_of_walkers": 4},
    "mcmc": {"num_mcmc_steps": 60, "num_mcmc_warmup_steps": 30, "num_mcmc_bin_blocks": 10},
}


def _dry_run(code, options, **inputs):
    """Run prepare_for_submission only and return the submit folder."""
    _, node = run_get_node(
        JqmcCalculation,
        code=code,
        metadata={"options": options, "dry_run": True, "store_provenance": False},
        **inputs,
    )
    return node, node.dry_run_info["folder"]


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"mcmc": {}}, "must contain a `control` section"),
        ({"control": {"job_type": "dmc"}}, "`control.job_type` must be one of"),
        ({"control": {"job_type": "vmc"}}, "must contain a `vmc` section"),
        ({"control": {"job_type": "mcmc", "restart": True}, "mcmc": {}}, "are set by the plugin"),
        ({"control": {"job_type": "mcmc", "verbosity": "mpi-low"}, "mcmc": {}}, "`control.verbosity` must be one of"),
    ],
)
def test_validate_parameters(jqmc_code, hamiltonian, options, parameters, message):
    """Structural problems in `parameters` are rejected before submission."""
    with pytest.raises(ValueError, match=message):
        run_get_node(
            JqmcCalculation,
            code=jqmc_code,
            parameters=orm.Dict(parameters),
            hamiltonian=hamiltonian,
            metadata={"options": options},
        )


def test_requires_exactly_one_start(jqmc_code, hamiltonian, options):
    """Neither or both of `hamiltonian` and `parent_folder` is an error."""
    with pytest.raises(ValueError, match="Exactly one of"):
        run_get_node(JqmcCalculation, code=jqmc_code, parameters=orm.Dict(MCMC), metadata={"options": options})


def test_max_time_margin_must_fit(jqmc_code, hamiltonian, options):
    """The default margin (>= 60 s) does not fit into a 30 s wall time."""
    with pytest.raises(ValueError, match="must be larger than max_time_margin"):
        run_get_node(
            JqmcCalculation,
            code=jqmc_code,
            parameters=orm.Dict(MCMC),
            hamiltonian=hamiltonian,
            metadata={"options": {**options, "max_wallclock_seconds": 30}},
        )


def test_max_wallclock_seconds_is_required(jqmc_code, hamiltonian, options):
    """Without a wall time jqmc gets no max_time and would be killed before writing its checkpoint."""
    options = {key: value for key, value in options.items() if key != "max_wallclock_seconds"}
    with pytest.raises(ValueError, match="max_wallclock_seconds` is required"):
        run_get_node(
            JqmcCalculation,
            code=jqmc_code,
            parameters=orm.Dict(MCMC),
            hamiltonian=hamiltonian,
            metadata={"options": options},
        )


@pytest.mark.parametrize(
    ("wall", "margin", "max_time"),
    [(None, None, 3240), (3600, None, 3240), (300, None, 240), (86400, None, 84600), (100, 30, 70)],
)
def test_input_file_new_run(jqmc_code, hamiltonian, options, wall, margin, max_time):
    """A new run copies the Hamiltonian and sets the reserved control keys."""
    if wall is not None:
        options = {**options, "max_wallclock_seconds": wall}
    if margin is not None:
        options = {**options, "max_time_margin": margin}
    _, folder = _dry_run(jqmc_code, options, parameters=orm.Dict(MCMC), hamiltonian=hamiltonian)

    written = toml.load(f"{folder}/aiida.toml")
    assert written["mcmc"] == MCMC["mcmc"]
    assert written["control"]["restart"] is False
    assert written["control"]["restart_chk"] == "restart.h5"
    assert written["control"]["hamiltonian_h5"] == "hamiltonian_data.h5"
    assert written["control"].get("max_time") == max_time
    assert written["control"]["verbosity"] == "high"


def test_input_file_keeps_given_verbosity(jqmc_code, hamiltonian, options):
    """An explicit `control.verbosity` is written as given."""
    parameters = {**MCMC, "control": {**MCMC["control"], "verbosity": "low"}}
    _, folder = _dry_run(jqmc_code, options, parameters=orm.Dict(parameters), hamiltonian=hamiltonian)
    assert toml.load(f"{folder}/aiida.toml")["control"]["verbosity"] == "low"


def test_without_frozen_keys_keeps_mutable_keys():
    """Only the keys jqmc restores from the checkpoint are dropped."""
    from aiida_jqmc.workflows.restart import _without_frozen_keys

    parameters = {
        "control": {"job_type": "mcmc", "number_of_walkers": 4, "mcmc_seed": 1, "verbosity": "low"},
        "mcmc": {"num_mcmc_steps": 10, "Dt": 2.0, "num_mcmc_bin_blocks": 5},
    }
    assert _without_frozen_keys(parameters, "mcmc") == {
        "control": {"job_type": "mcmc", "verbosity": "low"},
        "mcmc": {"num_mcmc_steps": 10, "num_mcmc_bin_blocks": 5},
    }


def test_twist_average_input_uses_a_checkpoint_directory(jqmc_code, options, twist_hamiltonian):
    """A twist-averaged Hamiltonian gets the per-twist checkpoint directory `restart_twists`."""
    _, folder = _dry_run(jqmc_code, options, parameters=orm.Dict(MCMC), hamiltonian=twist_hamiltonian)
    control = toml.load(f"{folder}/aiida.toml")["control"]
    assert control["restart_chk"] == "restart_twists"


def _finished_parent(computer, hamiltonian, options, tmp_path, parameters=MCMC) -> orm.RemoteData:
    """The remote folder of a stored, finished JqmcCalculation started from ``hamiltonian``."""
    from aiida.common.links import LinkType

    node = orm.CalcJobNode(computer=computer, process_type="aiida.calculations:jqmc.jqmc")
    for name, value in {"withmpi": True, **options}.items():
        node.set_option(name, value)
    node.base.links.add_incoming(orm.Dict(parameters).store(), link_type=LinkType.INPUT_CALC, link_label="parameters")
    node.base.links.add_incoming(hamiltonian.store(), link_type=LinkType.INPUT_CALC, link_label="hamiltonian")
    node.store()
    remote = orm.RemoteData(computer=computer, remote_path=str(tmp_path))
    remote.base.links.add_incoming(node, link_type=LinkType.CREATE, link_label="remote_folder")
    return remote.store()


def test_twist_average_restart_copies_the_checkpoint_directory(jqmc_code, options, twist_hamiltonian, tmp_path):
    """jqmc writes into the per-twist directory: a restart copies it, and copies the chain's Hamiltonian (route
    selection), so that no job depends on an earlier folder for it."""
    import json

    parent = _finished_parent(jqmc_code.computer, twist_hamiltonian, options, tmp_path)
    restart = {"control": {"job_type": "mcmc"}, "mcmc": {"num_mcmc_steps": 60}}
    _, folder = _dry_run(jqmc_code, options, parameters=orm.Dict(restart), parent_folder=parent)
    assert toml.load(f"{folder}/aiida.toml")["control"]["restart_chk"] == "restart_twists"
    calcinfo = json.loads(pathlib.Path(folder, ".aiida", "calcinfo.json").read_text())
    source = f"{parent.get_remote_path()}/restart_twists"
    assert [entry[1:] for entry in calcinfo["remote_copy_list"]] == [[source, "restart_twists"]]
    assert calcinfo["remote_symlink_list"] == []
    initial = parent.creator.inputs.hamiltonian
    assert [entry for entry in calcinfo["local_copy_list"] if entry[2] == "hamiltonian_data.h5"] == [
        [initial.uuid, initial.filename, "hamiltonian_data.h5"]
    ]


def test_single_twist_restart_links_the_checkpoint(jqmc_code, hamiltonian, options, tmp_path):
    import json

    parent = _finished_parent(jqmc_code.computer, hamiltonian, options, tmp_path)
    restart = {"control": {"job_type": "mcmc"}, "mcmc": {"num_mcmc_steps": 60}}
    _, folder = _dry_run(jqmc_code, options, parameters=orm.Dict(restart), parent_folder=parent)
    calcinfo = json.loads(pathlib.Path(folder, ".aiida", "calcinfo.json").read_text())
    assert calcinfo["remote_copy_list"] == []
    assert [entry[1:] for entry in calcinfo["remote_symlink_list"]] == [
        [f"{parent.get_remote_path()}/restart.h5", "restart.h5"]
    ]


@pytest.mark.parametrize(
    ("restart", "message"),
    [
        ({"control": {"job_type": "vmc"}, "vmc": {"num_opt_steps": 1}}, "Cannot restart job_type = 'vmc'"),
        ({"control": {"job_type": "mcmc", "number_of_walkers": 8}, "mcmc": {}}, r"number_of_walkers \(8\) differs"),
        ({"control": {"job_type": "mcmc"}, "mcmc": {}}, "must be on the same computer"),
    ],
)
def test_restart_must_keep_the_checkpoint_layout(
    jqmc_code, hamiltonian, options, tmp_path, aiida_computer_local, restart, message
):
    """A restart that changes the job type or the number of walkers, or runs on another computer, is rejected."""
    computer = aiida_computer_local() if "same computer" in message else jqmc_code.computer
    parent = _finished_parent(computer, hamiltonian, options, tmp_path)
    with pytest.raises(ValueError, match=message):
        _dry_run(jqmc_code, options, parameters=orm.Dict(restart), parent_folder=parent)


def test_tiled_vmc_restart_is_checked_against_the_initial_hamiltonian(jqmc_code, options, tmp_path):
    """A VMC restart reads no Hamiltonian of its own: the frozen tiled Slater orbitals come from the chain's start."""
    from .conftest import DATA_DIR

    tiled = orm.SinglefileData(DATA_DIR / "tiling" / "hamiltonian" / "lithium_tiled_211_gamma.h5")
    vmc = {"control": {"job_type": "vmc"}, "vmc": {"num_mcmc_steps": 10, "num_opt_steps": 1}}
    parent = _finished_parent(jqmc_code.computer, tiled, options, tmp_path, parameters=vmc)
    restart = {"control": {"job_type": "vmc"}, "vmc": {"num_opt_steps": 1, "opt_geminal": True}}
    with pytest.raises(ValueError, match="Tiled Slater orbitals are frozen"):
        _dry_run(jqmc_code, options, parameters=orm.Dict(restart), parent_folder=parent)


def test_importing_the_plugin_does_not_initialize_mpi():
    """Importing the plugin (and jqmc through it) must not initialize MPI: the singleton environment Open MPI
    then sets is inherited by every process the AiiDA daemon starts, and a local mpirun inheriting it exits
    silently. jqmc's library modules import ``mpi4py.MPI`` only inside the drivers."""
    import subprocess
    import sys

    from .conftest import DATA_DIR

    code = (
        "import sys\n"
        "import aiida_jqmc.calculations.jqmc, aiida_jqmc.calculations.tool, aiida_jqmc.parsers.jqmc, "
        "aiida_jqmc.parsers.tool, aiida_jqmc.workflows.restart, aiida_jqmc.tiling, aiida_jqmc.statistics\n"
        "print('mpi4py.MPI' in sys.modules)\n"
        f"print(aiida_jqmc.tiling.is_tiled_hamiltonian({str(DATA_DIR / 'H2_ecp_ccpvdz_jsd.h5')!r}))\n"
        "print('mpi4py.MPI' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=600)
    assert out.stdout.strip().splitlines()[-3:] == ["False", "False", "False"]  # nor after the check


def test_is_tiled_hamiltonian_recognizes_tiled_orbitals():
    from aiida_jqmc.tiling import is_tiled_hamiltonian

    from .conftest import DATA_DIR

    assert not is_tiled_hamiltonian(DATA_DIR / "H2_ecp_ccpvdz_jsd.h5")
    assert is_tiled_hamiltonian(DATA_DIR / "tiling" / "hamiltonian" / "lithium_tiled_211_gamma.h5")
    assert is_tiled_hamiltonian(DATA_DIR / "tiling" / "hamiltonian" / "lithium_tiled_211_twist_average.h5")
