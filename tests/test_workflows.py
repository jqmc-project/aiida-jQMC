"""End-to-end tests of JqmcRestartWorkChain on localhost (2 MPI processes)."""

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

import stat

import pytest
from aiida import orm
from aiida.engine import run_get_node
from aiida.plugins import CalculationFactory, WorkflowFactory

from .conftest import keep_logs_of_failed_jobs, which_or_skip

JqmcCalculation = CalculationFactory("jqmc.jqmc")
JqmcRestartWorkChain = WorkflowFactory("jqmc.restart")

pytestmark = pytest.mark.slow


def _mcmc(num_mcmc_steps: int) -> orm.Dict:
    return orm.Dict(
        {
            "control": {"job_type": "mcmc", "number_of_walkers": 4, "mcmc_seed": 1},
            "mcmc": {"num_mcmc_steps": num_mcmc_steps, "num_mcmc_warmup_steps": 30, "num_mcmc_bin_blocks": 10},
        }
    )


def _jobs(node) -> list:
    """The jqmc jobs a restart work chain ran (not its calcfunctions, e.g. merge_vmc_jobs)."""
    return sorted((n for n in node.called if n.process_label == "JqmcCalculation"), key=lambda n: n.ctime)


def test_mcmc_split_by_max_time(jqmc_code, hamiltonian, options):
    """max_time = 5 s per job: the run is split into several jobs and ends exactly on the target."""
    # The native route runs about 1.5-10 x 10^3 steps/s for this H2 input,
    # depending on the machine: the target needs several 5 s jobs on any of them.
    # max_iterations only guards against a run that never ends; it must not
    # depend on the machine's speed (20 jobs at the slow end).
    target = 150_000
    inputs = {
        "jqmc": {
            "code": jqmc_code,
            "parameters": _mcmc(target),
            "hamiltonian": hamiltonian,
            "metadata": {"options": {**options, "max_wallclock_seconds": 65, "max_time_margin": 60}},
        },
        "max_iterations": orm.Int(200),
    }
    _, node = run_get_node(JqmcRestartWorkChain, **inputs)
    keep_logs_of_failed_jobs(node)
    assert node.is_finished_ok, node.exit_message
    jobs = _jobs(node)
    assert len(jobs) > 1
    assert all(job.is_finished_ok for job in jobs)
    assert [job.outputs.output_parameters["stopped_by_max_time"] for job in jobs[:-1]] == [True] * (len(jobs) - 1)
    assert node.outputs.output_parameters["mcmc_counter"] == target
    # Only the step count changes between jobs.
    requested = [job.inputs.parameters["mcmc"]["num_mcmc_steps"] for job in jobs]
    counters = [0] + [job.outputs.output_parameters["mcmc_counter"] for job in jobs[:-1]]
    assert requested == [target - c for c in counters]

    # Extend the finished calculation by 20 steps.
    _, extended = run_get_node(
        JqmcRestartWorkChain,
        jqmc={
            "code": jqmc_code,
            "parameters": _mcmc(20),
            "parent_folder": node.outputs.remote_folder,
            "metadata": {"options": options},
        },
    )
    assert extended.is_finished_ok
    assert extended.outputs.output_parameters["mcmc_counter"] == target + 20


def _vmc(num_opt_steps: int) -> orm.Dict:
    return orm.Dict(
        {
            "control": {"job_type": "vmc", "number_of_walkers": 4},
            "vmc": {
                "num_mcmc_steps": 40,
                "num_mcmc_warmup_steps": 30,
                "num_mcmc_bin_blocks": 10,
                "num_opt_steps": num_opt_steps,
            },
        }
    )


def test_vmc_split_by_max_time(jqmc_code, hamiltonian, options):
    """max_time = 1 s: the run is split into jobs and ends exactly on the target.

    jqmc checks max_time between optimization steps, from the start of the
    optimization, so how many steps a job does depends on the machine's speed.
    """
    _, node = run_get_node(
        JqmcRestartWorkChain,
        jqmc={
            "code": jqmc_code,
            "parameters": _vmc(3),
            "hamiltonian": hamiltonian,
            "metadata": {"options": {**options, "max_wallclock_seconds": 61, "max_time_margin": 60, "all_hamiltonians": True}},
        },
    )
    keep_logs_of_failed_jobs(node)
    assert node.is_finished_ok, node.exit_message
    jobs = [job for job in _jobs(node) if job.is_finished_ok]
    counters = [job.outputs.output_parameters["opt_step_counter"] for job in jobs]
    # the run returns the steps and the Hamiltonians of every job, not only those of the last one
    assert len(jobs) > 1
    assert [step["step"] for step in node.outputs.output_parameters["steps"]] == [1, 2, 3]
    assert node.outputs.output_parameters["num_jobs"] == len(jobs)
    assert sorted(node.outputs.output_hamiltonians) == ["step_1", "step_2", "step_3"]
    assert counters[-1] == 3
    assert all(a < b for a, b in zip([0, *counters], counters, strict=False))
    assert all(job.outputs.output_parameters["stopped_by_max_time"] for job in jobs[:-1])
    # Each job is asked for what is left.
    assert [job.inputs.parameters["vmc"]["num_opt_steps"] for job in jobs] == [3 - c for c in [0, *counters[:-1]]]
    assert node.outputs.output_hamiltonian.filename == "hamiltonian_data_opt_step_3.h5"


def test_lrdmc_split_by_max_time_keeps_frozen_keys_out_of_continuations(jqmc_code, hamiltonian, options):
    """lrdmc-bra with an explicit E_scf, split by max_time = 5 s.

    jqmc updates E_scf during the run and rejects a restart TOML whose E_scf
    differs from the checkpoint, so the continuation jobs must leave it (and the
    other frozen keys) out.
    """
    # ~10^3 branching steps/s for this input: several 5 s jobs.
    target = 15_000
    parameters = orm.Dict(
        {
            "control": {"job_type": "lrdmc-bra", "number_of_walkers": 4, "mcmc_seed": 1},
            "lrdmc-bra": {
                "num_mcmc_steps": target,
                "alat": 0.3,
                "num_mcmc_per_measurement": 40,
                "E_scf": -1.0,
                "num_gfmc_warmup_steps": 25,
                "num_gfmc_bin_blocks": 10,
                "num_gfmc_collect_steps": 5,
            },
        }
    )
    _, node = run_get_node(
        JqmcRestartWorkChain,
        jqmc={
            "code": jqmc_code,
            "parameters": parameters,
            "hamiltonian": hamiltonian,
            "metadata": {"options": {**options, "max_wallclock_seconds": 65, "max_time_margin": 60}},
        },
        max_iterations=orm.Int(200),  # a runaway guard only, not tied to the machine speed
    )
    keep_logs_of_failed_jobs(node)
    assert node.is_finished_ok, node.exit_message
    jobs = _jobs(node)
    assert len(jobs) > 1
    assert node.outputs.output_parameters["gfmc_counter"] == target
    assert jobs[0].inputs.parameters["lrdmc-bra"]["E_scf"] == -1.0
    for job in jobs[1:]:
        assert "E_scf" not in job.inputs.parameters["lrdmc-bra"]
        assert "alat" not in job.inputs.parameters["lrdmc-bra"]
        assert "num_mcmc_per_measurement" not in job.inputs.parameters["lrdmc-bra"]
        assert "number_of_walkers" not in job.inputs.parameters["control"]


def test_lrdmc_tau_split_by_max_time(jqmc_code, hamiltonian, options):
    """lrdmc-tau split by max_time = 5 s: ends on the target, alat and tau left out of the continuations."""
    target = 15_000
    parameters = orm.Dict(
        {
            "control": {"job_type": "lrdmc-tau", "number_of_walkers": 4, "mcmc_seed": 1},
            "lrdmc-tau": {
                "num_mcmc_steps": target,
                "alat": 0.3,
                "tau": 0.1,
                "num_gfmc_warmup_steps": 25,
                "num_gfmc_bin_blocks": 10,
                "num_gfmc_collect_steps": 5,
            },
        }
    )
    _, node = run_get_node(
        JqmcRestartWorkChain,
        jqmc={
            "code": jqmc_code,
            "parameters": parameters,
            "hamiltonian": hamiltonian,
            "metadata": {"options": {**options, "max_wallclock_seconds": 65, "max_time_margin": 60}},
        },
        max_iterations=orm.Int(200),  # a runaway guard only, not tied to the machine speed
    )
    keep_logs_of_failed_jobs(node)
    assert node.is_finished_ok, node.exit_message
    jobs = _jobs(node)
    assert len(jobs) > 1
    assert node.outputs.output_parameters["gfmc_counter"] == target
    for job in jobs[1:]:
        assert not {"alat", "tau"} & set(job.inputs.parameters["lrdmc-tau"])


@pytest.fixture
def flaky_jqmc_code(aiida_code_installed, aiida_localhost, tmp_path):
    """`jqmc` that is killed (SIGKILL) after 3 s on its first call only, as if the job hit the wall time."""
    jqmc = which_or_skip("jqmc")
    marker = tmp_path / "already_failed"
    wrapper = tmp_path / "flaky_jqmc.sh"
    wrapper.write_text(
        "#!/bin/bash\n"
        f'if [ ! -e "{marker}" ]; then\n'
        f'  "{jqmc}" "$@" & pid=$!; sleep 3; kill -9 $pid; wait $pid; touch "{marker}"; exit 1\n'
        "fi\n"
        f'exec "{jqmc}" "$@"\n'
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    return aiida_code_installed(filepath_executable=str(wrapper), computer=aiida_localhost, with_mpi=True)


def test_interrupted_job_is_resubmitted(flaky_jqmc_code, hamiltonian, options):
    """A job cut off before jqmc ended is resubmitted with the same physics and a larger margin."""
    _, node = run_get_node(
        JqmcRestartWorkChain,
        jqmc={
            "code": flaky_jqmc_code,
            # Long enough (~15 s) to be killed after 3 s.
            "parameters": _mcmc(200_000),
            "hamiltonian": hamiltonian,
            "metadata": {"options": {**options, "max_wallclock_seconds": 600, "max_time_margin": 100}},
        },
    )
    assert node.is_finished_ok, node.exit_message
    first, second = _jobs(node)
    assert first.exit_status == JqmcCalculation.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE.status
    assert second.is_finished_ok
    assert second.inputs.parameters.get_dict() == first.inputs.parameters.get_dict()
    assert "hamiltonian" in second.inputs  # nothing to restart from: start again
    assert second.get_option("max_time_margin") == 200
    assert node.outputs.output_parameters["mcmc_counter"] == 200_000


@pytest.fixture
def crashing_jqmc_code(aiida_code_installed, aiida_localhost, tmp_path):
    """`jqmc` killed (SIGKILL) after 3 s on every call, leaving neither an exception nor a scheduler verdict."""
    jqmc = which_or_skip("jqmc")
    wrapper = tmp_path / "crashing_jqmc.sh"
    wrapper.write_text(f'#!/bin/bash\n"{jqmc}" "$@" & pid=$!; sleep 3; kill -9 $pid; wait $pid; exit 1\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    return aiida_code_installed(filepath_executable=str(wrapper), computer=aiida_localhost, with_mpi=True)


def test_repeated_crash_is_retried_once(crashing_jqmc_code, hamiltonian, options):
    """A crash without an error message (exit 302) is retried once, then the work chain stops with 413."""
    _, node = run_get_node(
        JqmcRestartWorkChain,
        jqmc={
            "code": crashing_jqmc_code,
            "parameters": _mcmc(200_000),
            "hamiltonian": hamiltonian,
            "metadata": {"options": {**options, "max_wallclock_seconds": 600, "max_time_margin": 100}},
        },
    )
    assert node.exit_status == JqmcRestartWorkChain.exit_codes.ERROR_REPEATED_CRASH.status
    jobs = _jobs(node)
    assert len(jobs) == 2
    assert all(job.exit_status == JqmcCalculation.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE.status for job in jobs)


def test_invalid_input_is_not_retried(jqmc_code, hamiltonian, options):
    """Errors that need a decision are not handled: the work chain stops after one job."""
    parameters = {
        "control": {"job_type": "lrdmc-bra"},
        "lrdmc-bra": {"num_mcmc_steps": 60, "num_mcmc_per_measurement": 40, "num_gfmc_warmup_steps": 10},
    }
    _, node = run_get_node(
        JqmcRestartWorkChain,
        jqmc={
            "code": jqmc_code,
            "parameters": orm.Dict(parameters),
            "hamiltonian": hamiltonian,
            "metadata": {"options": options},
        },
    )
    assert node.exit_status == JqmcRestartWorkChain.exit_codes.ERROR_UNHANDLED_FAILURE.status
    assert len(_jobs(node)) == 1
