"""JqmcRestartWorkChain without a jqmc run: its helpers, and the decisions it takes before or between jobs."""

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

from aiida import orm

from aiida_jqmc.schemas import output_parameters_schema


def _job(steps: list[int], seconds: float, commit: str, breakdown: dict) -> orm.Dict:
    return orm.Dict(
        {
            "steps": [{"step": step, "energy": -1.0 * step} for step in steps],
            "dumped_hamiltonians": [f"hamiltonian_data_opt_step_{step}.h5" for step in steps],
            "total_opt_time_sec": seconds,
            "opt_timing_breakdown": breakdown,
            "opt_step_counter": steps[-1],
            "environment": {"git_commit": commit},
            "warnings": ["w"],
        }
    )


def test_merge_vmc_jobs_combines_per_job_fields():
    """Steps, Hamiltonians, times and warnings of every job; the timing per step is averaged over all steps."""
    from aiida_jqmc.workflows.restart import merge_vmc_jobs

    merged = merge_vmc_jobs(
        job_000=_job([1, 2, 3], 30.0, "a" * 40, {"optimizer": 9.0, "sampling": 1.0}),
        job_001=_job([4], 12.0, "b" * 40, {"optimizer": 11.0, "sampling": 1.0}),
    ).get_dict()
    assert [step["step"] for step in merged["steps"]] == [1, 2, 3, 4]
    assert merged["dumped_hamiltonians"][-1] == "hamiltonian_data_opt_step_4.h5"
    assert len(merged["dumped_hamiltonians"]) == 4
    assert merged["total_opt_time_sec"] == 42.0
    assert merged["opt_timing_breakdown"] == {"optimizer": (3 * 9.0 + 11.0) / 4, "sampling": 1.0}
    assert merged["opt_step_counter"] == 4  # the last job's
    assert merged["num_jobs"] == 2
    assert merged["warnings"][0] == "w"
    assert "different jqmc commits" in merged["warnings"][-1]


def test_num_jobs_is_optional_in_the_vmc_schema():
    """A single VMC job has no `num_jobs`; a merged run has it."""
    schema = output_parameters_schema("vmc")
    assert "num_jobs" in schema["properties"]
    assert "num_jobs" not in schema["required"]


def test_parent_without_a_known_counter_is_rejected(jqmc_code, options, tmp_path):
    """A parent folder that no successful JqmcCalculation wrote has no counter to count the target from: 410."""
    from aiida.engine import run_get_node
    from aiida.plugins import WorkflowFactory

    JqmcRestartWorkChain = WorkflowFactory("jqmc.restart")
    parent = orm.RemoteData(computer=jqmc_code.computer, remote_path=str(tmp_path)).store()
    parameters = orm.Dict({"control": {"job_type": "mcmc"}, "mcmc": {"num_mcmc_steps": 60}})
    _, node = run_get_node(
        JqmcRestartWorkChain,
        jqmc={"code": jqmc_code, "parameters": parameters, "parent_folder": parent, "metadata": {"options": options}},
    )
    assert node.exit_status == JqmcRestartWorkChain.exit_codes.ERROR_PARENT_COUNTER_UNKNOWN.status
    assert not node.called


def test_job_without_progress_ends_the_work_chain(aiida_code_installed, aiida_localhost, tmp_path):
    """A job that ends normally without advancing the counter stops the work chain (411) instead of looping.

    The code replays the recorded log of an MCMC restart stopped by max_time, so every job reports the same counter.
    """
    import stat

    from aiida.engine import run_get_node
    from aiida.plugins import WorkflowFactory

    from .conftest import DATA_DIR

    JqmcRestartWorkChain = WorkflowFactory("jqmc.restart")
    replay = tmp_path / "replay.sh"
    replay.write_text(f'#!/bin/bash\ncat "{DATA_DIR / "mcmc_max_time.out"}"\n')
    replay.chmod(replay.stat().st_mode | stat.S_IEXEC)
    code = aiida_code_installed(filepath_executable=str(replay), computer=aiida_localhost, with_mpi=False)
    options = {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1}, "withmpi": False, "max_wallclock_seconds": 3600}
    parameters = orm.Dict({"control": {"job_type": "mcmc"}, "mcmc": {"num_mcmc_steps": 10**7}})
    _, node = run_get_node(
        JqmcRestartWorkChain,
        jqmc={
            "code": code,
            "parameters": parameters,
            "hamiltonian": orm.SinglefileData(DATA_DIR / "H2_ecp_ccpvdz_jsd.h5"),
            "metadata": {"options": options},
        },
    )
    assert node.exit_status == JqmcRestartWorkChain.exit_codes.ERROR_NO_PROGRESS.status
    first, second = sorted(node.called, key=lambda job: job.ctime)
    assert first.is_finished_ok and second.is_finished_ok
    assert second.outputs.output_parameters["mcmc_counter"] == first.outputs.output_parameters["mcmc_counter"]
