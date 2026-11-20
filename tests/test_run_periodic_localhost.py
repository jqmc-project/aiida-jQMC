"""End-to-end periodic runs on localhost: QE conversion, MPC, stress, twist averaging.

The Quantum ESPRESSO outputs (bcc Li, ccECP) are stored in AiiDA as FolderData
and converted by ``jqmc-tool qe``. As in test_run_tools_localhost, each run is
checked for completeness, not for values.
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

import pathlib

import pytest
from aiida import orm
from aiida.engine import run_get_node
from aiida.plugins import CalculationFactory, WorkflowFactory

from aiida_jqmc.calculations.jqmc import is_twist_average
from aiida_jqmc.utils import parameters_from_toml

from .conftest import assert_complete, assert_tool_complete

JqmcCalculation = CalculationFactory("jqmc.jqmc")
JqmcToolCalculation = CalculationFactory("jqmc.tool")
JqmcRestartWorkChain = WorkflowFactory("jqmc.restart")

pytestmark = pytest.mark.slow

NUM_TWISTS = 8

MCMC_TOML = """
[control]
job_type = "mcmc"
mcmc_seed = 34478
number_of_walkers = 2
max_time = 86400
restart = false
restart_chk = "restart.h5"
hamiltonian_h5 = "hamiltonian_data.h5"
verbosity = "high"

[mcmc]
num_mcmc_steps = 40
num_mcmc_per_measurement = 4
num_mcmc_warmup_steps = 10
num_mcmc_bin_blocks = 5
atomic_force = true
compute_stress = true

[coulomb]
mpc = true
mpc_cutoff = 10.0
"""


def _qe(code, data: pathlib.Path, name: str, twist_average: bool = False):
    folder = orm.FolderData(tree=data / "qe" / name)
    _, node = run_get_node(
        JqmcToolCalculation,
        code=code,
        command=orm.Str("qe"),
        input_folder=folder,
        save_path=orm.Str("out"),
        options=orm.Dict({"--kind": "pgto", "-j2": 1.0}),
        **({"twist_average": orm.Bool(True)} if twist_average else {}),
    )
    assert_tool_complete(node)
    return node.outputs.output_hamiltonian


def _mcmc(tmp_path, coulomb: dict | None = None, **changes) -> dict:
    """The tutorial-style MCMC input, read through parameters_from_toml."""
    path = tmp_path / "mcmc.toml"
    path.write_text(MCMC_TOML)
    parameters, removed = parameters_from_toml(path)
    assert sorted(removed) == ["hamiltonian_h5", "max_time", "restart", "restart_chk"]
    parameters["mcmc"].update(changes)
    parameters["coulomb"].update(coulomb or {})
    return parameters


def _run(code, parameters: dict, processes: int, process=JqmcCalculation, **inputs):
    options = {
        "resources": {"num_machines": 1, "num_mpiprocs_per_machine": processes},
        "max_wallclock_seconds": 3600,
        **inputs.pop("options", {}),
    }
    inputs = {"code": code, "parameters": orm.Dict(parameters), "metadata": {"options": options}, **inputs}
    _, node = run_get_node(process, **({"jqmc": inputs} if process is JqmcRestartWorkChain else inputs))
    return node


def test_qe_gamma_mcmc_with_forces_stress_and_mpc(jqmc_code, jqmc_tool_code, data_with_pseudopotentials, tmp_path):
    hamiltonian = _qe(jqmc_tool_code, data_with_pseudopotentials, "Li_gamma")
    assert not is_twist_average(hamiltonian)

    node = _run(jqmc_code, _mcmc(tmp_path), 2, hamiltonian=hamiltonian)
    out = assert_complete(node, "mcmc")
    assert out["mcmc_counter"] == 40
    assert len(out["atomic_forces"]) == 2
    assert out["energy_mpc"] is not None and out["mpc_setup"]
    assert {"sigma_xx", "sigma_xx_err"} <= set(out["stress_tensor"])
    assert {"P_hydro", "P_hydro_err"} <= set(out["pressure"])

    energy = _tool_energy(jqmc_tool_code, node)
    assert energy["energy"] is not None and energy["energy_mpc"] is not None


def _tool_energy(code, node) -> dict:
    _, tool = run_get_node(
        JqmcToolCalculation,
        code=code,
        command=orm.Str("mcmc compute-energy"),
        parent_folders={"mcmc": node.outputs.remote_folder},
        options=orm.Dict({"-b": 5, "-w": 2}),
    )
    return assert_tool_complete(tool)


def test_twist_average_run_restart_and_post_processing(jqmc_code, jqmc_tool_code, data_with_pseudopotentials, tmp_path):
    """Eight twists, one MPI rank each: the BZ average, a restart of the finished run, compute-energy on the twists,
    and an LRDMC run of the same twists."""
    hamiltonian = _qe(jqmc_tool_code, data_with_pseudopotentials, "Li_k_twist", twist_average=True)
    assert is_twist_average(hamiltonian)

    # Every twist sets up its own MPC table (minutes for eight at once on one machine): MPC is
    # run in the Gamma test; its BZ average is checked on a recorded cluster log in test_schemas.
    parameters = _mcmc(tmp_path, coulomb={"mpc": False}, atomic_force=False, compute_stress=False)
    node = _run(jqmc_code, parameters, NUM_TWISTS, hamiltonian=hamiltonian, options={"retrieve_checkpoint": True})
    out = assert_complete(node, "mcmc")
    assert [row["twist"] for row in out["twist_average"]["per_twist"]] == list(range(NUM_TWISTS))
    # the per-twist checkpoint directory is retrieved whole
    checkpoints = node.outputs.retrieved.base.repository.list_object_names("restart_twists")
    assert {f"twist_{i:03d}.h5" for i in range(NUM_TWISTS)} | {"shared_state.h5"} <= set(checkpoints)

    # Every twist is complete: the restart reproduces the result without sampling again.
    again = _run(jqmc_code, parameters, NUM_TWISTS, parent_folder=node.outputs.remote_folder)
    assert assert_complete(again, "mcmc")["twist_average"] == out["twist_average"]

    energy = _tool_energy(jqmc_tool_code, node)
    assert energy["num_twists"] == NUM_TWISTS

    # LRDMC takes the same twist route.
    lrdmc = {
        "control": {"job_type": "lrdmc-bra", "number_of_walkers": 2, "mcmc_seed": 1},
        "lrdmc-bra": {
            "num_mcmc_steps": 40,
            "alat": 0.3,
            "num_mcmc_per_measurement": 4,
            "num_gfmc_warmup_steps": 25,
            "num_gfmc_bin_blocks": 10,
            "num_gfmc_collect_steps": 5,
        },
    }
    out = assert_complete(_run(jqmc_code, lrdmc, NUM_TWISTS, hamiltonian=hamiltonian), "lrdmc-bra")
    assert [row["twist"] for row in out["twist_average"]["per_twist"]] == list(range(NUM_TWISTS))


def test_twist_stopped_on_max_time_then_restarted(jqmc_code, jqmc_tool_code, data_with_pseudopotentials, tmp_path):
    """max_time stops the twists: ERROR_TWIST_STOPPED_EARLY, which the work chain leaves to the caller; a restart
    from that job's folder (through the work chain) reruns the incomplete twists to the end."""
    hamiltonian = _qe(jqmc_tool_code, data_with_pseudopotentials, "Li_k_twist", twist_average=True)
    parameters = _mcmc(tmp_path, coulomb={"mpc": False}, atomic_force=False, compute_stress=False, num_mcmc_steps=4000)
    # max_time = 1 s for each twist.
    workchain = _run(
        jqmc_code,
        parameters,
        NUM_TWISTS,
        process=JqmcRestartWorkChain,
        hamiltonian=hamiltonian,
        options={"max_wallclock_seconds": 61, "max_time_margin": 60},
    )
    assert workchain.exit_status == JqmcRestartWorkChain.exit_codes.ERROR_UNHANDLED_FAILURE.status
    (stopped,) = workchain.called
    assert stopped.exit_status == JqmcCalculation.exit_codes.ERROR_TWIST_STOPPED_EARLY.status
    out = stopped.outputs.output_parameters.get_dict()
    assert out["stopped_by_max_time"]
    assert out["unparsed_numeric_lines"] == []
    assert [row["twist"] for row in out["incomplete_twists"]] == list(range(NUM_TWISTS))
    assert all(row["steps"] < row["requested_steps"] == 4000 for row in out["incomplete_twists"])

    parameters["mcmc"]["num_mcmc_steps"] = 40
    resumed = _run(jqmc_code, parameters, NUM_TWISTS, process=JqmcRestartWorkChain, parent_folder=stopped.outputs.remote_folder)
    out = assert_complete(resumed, "mcmc")
    assert len(resumed.called) == 1
    assert [row["twist"] for row in out["twist_average"]["per_twist"]] == list(range(NUM_TWISTS))
