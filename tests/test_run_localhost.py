"""End-to-end runs of jqmc on localhost (2 MPI processes)."""

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

import os

import pytest
from aiida import orm
from aiida.engine import run_get_node
from aiida.plugins import CalculationFactory

JqmcCalculation = CalculationFactory("jqmc.jqmc")

pytestmark = pytest.mark.slow


def _mcmc(num_mcmc_steps: int, num_mcmc_warmup_steps: int = 30) -> orm.Dict:
    return orm.Dict(
        {
            "control": {"job_type": "mcmc", "number_of_walkers": 4, "mcmc_seed": 1},
            "mcmc": {
                "num_mcmc_steps": num_mcmc_steps,
                "num_mcmc_warmup_steps": num_mcmc_warmup_steps,
                "num_mcmc_bin_blocks": 10,
            },
        }
    )


def test_mcmc_restart_chain(jqmc_code, hamiltonian, options):
    """New run -> restart stopped by max_time -> restart to completion. Counters accumulate."""
    _, first = run_get_node(
        JqmcCalculation,
        code=jqmc_code,
        parameters=_mcmc(60),
        hamiltonian=hamiltonian,
        metadata={"options": {**options, "retrieve_checkpoint": True}},
    )
    assert first.is_finished_ok
    assert first.outputs.output_parameters["mcmc_counter"] == 60
    assert "restart.h5" in first.outputs.retrieved.base.repository.list_object_names()

    _, second = run_get_node(
        JqmcCalculation,
        code=jqmc_code,
        parameters=_mcmc(10**7),
        parent_folder=first.outputs.remote_folder,
        metadata={"options": {**options, "max_wallclock_seconds": 70, "max_time_margin": 60}},
    )
    assert second.is_finished_ok
    out = second.outputs.output_parameters.get_dict()
    assert out["stopped_by_max_time"] is True
    assert 60 < out["mcmc_counter"] < 60 + 10**7

    _, third = run_get_node(
        JqmcCalculation,
        code=jqmc_code,
        parameters=_mcmc(20),
        parent_folder=second.outputs.remote_folder,
        metadata={"options": options},
    )
    assert third.is_finished_ok
    assert third.outputs.output_parameters["mcmc_counter"] == out["mcmc_counter"] + 20

    # The child replaced its symlink with a new file; the parent's checkpoint is intact.
    for node in (first, second, third):
        checkpoint = os.path.join(node.outputs.remote_folder.get_remote_path(), "restart.h5")
        assert os.path.isfile(checkpoint)
        assert not os.path.islink(checkpoint)

    with pytest.raises(ValueError, match="number of MPI processes"):
        run_get_node(
            JqmcCalculation,
            code=jqmc_code,
            parameters=_mcmc(20),
            parent_folder=third.outputs.remote_folder,
            metadata={"options": {**options, "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1}}},
        )


def test_mcmc_stopped_before_warmup_ends(jqmc_code, hamiltonian, options):
    """A run stopped before the warm-up ends has no energy yet, but writes a checkpoint that can be continued."""
    _, first = run_get_node(
        JqmcCalculation,
        code=jqmc_code,
        # The warm-up is far longer than the ~10^4 steps the 1 s budget allows.
        parameters=_mcmc(10**7, num_mcmc_warmup_steps=10**6),
        hamiltonian=hamiltonian,
        metadata={"options": {**options, "max_wallclock_seconds": 61, "max_time_margin": 60}},
    )
    assert first.is_finished_ok, first.exit_message
    out = first.outputs.output_parameters.get_dict()
    assert out["stopped_by_max_time"] is True
    assert out["mcmc_counter"] < 10**6
    assert out["energy"] is None

    _, second = run_get_node(
        JqmcCalculation,
        code=jqmc_code,
        parameters=_mcmc(60, num_mcmc_warmup_steps=0),
        parent_folder=first.outputs.remote_folder,
        metadata={"options": options},
    )
    assert second.is_finished_ok
    assert second.outputs.output_parameters["mcmc_counter"] == out["mcmc_counter"] + 60
    assert second.outputs.output_parameters["energy"] is not None


def test_vmc_then_lrdmc(jqmc_code, hamiltonian, options):
    """VMC (new + restart) produces a Hamiltonian that LRDMC can start from."""
    vmc = {
        "control": {"job_type": "vmc", "number_of_walkers": 4},
        "vmc": {"num_mcmc_steps": 40, "num_mcmc_warmup_steps": 30, "num_mcmc_bin_blocks": 10, "num_opt_steps": 2},
    }
    _, first = run_get_node(
        JqmcCalculation, code=jqmc_code, parameters=orm.Dict(vmc), hamiltonian=hamiltonian, metadata={"options": options}
    )
    assert first.is_finished_ok
    assert first.outputs.output_hamiltonian.filename == "hamiltonian_data_opt_step_2.h5"

    _, second = run_get_node(
        JqmcCalculation,
        code=jqmc_code,
        parameters=orm.Dict(vmc),
        parent_folder=first.outputs.remote_folder,
        metadata={"options": options},
    )
    assert second.is_finished_ok
    assert [s["step"] for s in second.outputs.output_parameters["steps"]] == [3, 4]
    assert second.outputs.output_hamiltonian.filename == "hamiltonian_data_opt_step_4.h5"

    lrdmc = {
        "control": {"job_type": "lrdmc-bra", "number_of_walkers": 4},
        "lrdmc-bra": {
            "num_mcmc_steps": 60,
            "alat": 0.3,
            "num_mcmc_per_measurement": 40,
            "num_gfmc_warmup_steps": 25,
            "num_gfmc_bin_blocks": 10,
            "num_gfmc_collect_steps": 5,
        },
    }
    _, node = run_get_node(
        JqmcCalculation,
        code=jqmc_code,
        parameters=orm.Dict(lrdmc),
        hamiltonian=second.outputs.output_hamiltonian,
        metadata={"options": options},
    )
    assert node.is_finished_ok
    assert node.outputs.output_parameters["gfmc_counter"] == 60


def test_invalid_input_exit_code(jqmc_code, hamiltonian, options):
    """jqmc's own input check (num_gfmc_warmup_steps below its minimum) maps to ERROR_INVALID_INPUT."""
    parameters = {
        "control": {"job_type": "lrdmc-bra"},
        "lrdmc-bra": {"num_mcmc_steps": 60, "num_mcmc_per_measurement": 40, "num_gfmc_warmup_steps": 10},
    }
    _, node = run_get_node(
        JqmcCalculation,
        code=jqmc_code,
        parameters=orm.Dict(parameters),
        hamiltonian=hamiltonian,
        metadata={"options": options},
    )
    assert node.exit_status == JqmcCalculation.exit_codes.ERROR_INVALID_INPUT.status
