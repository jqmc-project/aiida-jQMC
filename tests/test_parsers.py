"""Tests for JqmcParser: mapping of jqmc logs onto outputs and exit codes."""

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

import math
import re
import shutil

import pytest
from aiida import orm
from aiida.common.links import LinkType
from aiida.plugins import CalculationFactory, ParserFactory

from .conftest import DATA_DIR

JqmcCalculation = CalculationFactory("jqmc.jqmc")
JqmcParser = ParserFactory("jqmc.jqmc")


@pytest.fixture
def generate_node(aiida_localhost):
    """Create a stored CalcJobNode with a `retrieved` folder holding the given files."""

    def factory(job_type: str, files: dict[str, str], hamiltonian=None, **options) -> orm.CalcJobNode:
        node = orm.CalcJobNode(computer=aiida_localhost, process_type="aiida.calculations:jqmc.jqmc")
        node.set_option("resources", {"num_machines": 1, "num_mpiprocs_per_machine": 2})
        for name, value in options.items():
            node.set_option(name, value)
        node.set_option("output_filename", "aiida.out")
        node.set_option("scheduler_stderr", "_scheduler-stderr.txt")
        parameters = orm.Dict({"control": {"job_type": job_type}, job_type: {}}).store()
        node.base.links.add_incoming(parameters, link_type=LinkType.INPUT_CALC, link_label="parameters")
        if hamiltonian is not None:
            node.base.links.add_incoming(hamiltonian.store(), link_type=LinkType.INPUT_CALC, link_label="hamiltonian")
        node.store()

        retrieved = orm.FolderData()
        for name, content in files.items():
            retrieved.base.repository.put_object_from_bytes(content.encode(), name)
        retrieved.base.links.add_incoming(node, link_type=LinkType.CREATE, link_label="retrieved")
        retrieved.store()
        return node

    return factory


def _log(name: str) -> str:
    return (DATA_DIR / name).read_text()


def _parse(node, retrieved_temporary_folder=None):
    results, calcfunction = JqmcParser.parse_from_node(
        node, store_provenance=False, retrieved_temporary_folder=retrieved_temporary_folder
    )
    return results, calcfunction.exit_status, calcfunction.exit_message


def test_mcmc_finished(generate_node):
    """A completed MCMC run gives exit 0 and the final energy.

    The logs in ``tests/data`` come from real jqmc runs. Printed values change
    between jqmc versions, so only their presence and the counters fixed by the
    input are checked.
    """
    results, status, _ = _parse(generate_node("mcmc", {"aiida.out": _log("mcmc.out")}))
    assert status == 0
    out = results["output_parameters"].get_dict()
    for name in ("energy", "energy_error", "variance", "variance_error"):
        assert out[name] is not None and math.isfinite(out[name]), name
    assert out["mcmc_counter"] == out["mcmc_counter_target"] == 60
    assert out["stopped_by_max_time"] is False


def test_mcmc_stopped_by_max_time_is_success(generate_node):
    """Stopping on max_time is a normal end: jqmc wrote its checkpoint."""
    results, status, _ = _parse(generate_node("mcmc", {"aiida.out": _log("mcmc_max_time.out")}))
    assert status == 0
    out = results["output_parameters"].get_dict()
    assert out["stopped_by_max_time"] is True
    # Restart of the 60-step run with num_mcmc_steps = 10**6 and max_time = 1.
    assert out["mcmc_counter_target"] == 60 + 10**6
    assert 60 < out["mcmc_counter"] < out["mcmc_counter_target"]


def test_stdout_missing(generate_node):
    """No stdout in the retrieved folder."""
    _, status, _ = _parse(generate_node("mcmc", {}))
    assert status == JqmcCalculation.exit_codes.ERROR_OUTPUT_STDOUT_MISSING.status


def test_stdout_incomplete(generate_node):
    """A log cut off mid-run (e.g. killed by the scheduler) without an exception."""
    text = _log("mcmc.out")
    truncated = text[: text.index("End MCMC")]
    results, status, _ = _parse(generate_node("mcmc", {"aiida.out": truncated}))
    assert status == JqmcCalculation.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE.status
    assert results["output_parameters"]["program_ended"] is False


def test_not_started(generate_node):
    """An empty stdout (e.g. mpirun refused to launch) is not an interruption to retry."""
    stderr = "Your job has requested more processes than the ppr for\nthis topology can support:\n"
    _, status, _ = _parse(generate_node("mcmc", {"aiida.out": "", "_scheduler-stderr.txt": stderr}))
    assert status == JqmcCalculation.exit_codes.ERROR_NOT_STARTED.status


def test_stdout_incomplete_keeps_scheduler_verdict(generate_node):
    """When the scheduler already reported e.g. the wall time, the parser does not replace it."""
    text = _log("mcmc.out")
    node = generate_node("mcmc", {"aiida.out": text[: text.index("End MCMC")]})
    node.set_exit_status(JqmcCalculation.exit_codes.ERROR_SCHEDULER_OUT_OF_WALLTIME.status)
    results, status, _ = _parse(node)
    assert status == 0  # the parser returns nothing, so the node keeps 120
    assert results["output_parameters"]["program_ended"] is False


def test_invalid_input(generate_node):
    """jqmc rejects its input: the exception from stderr is reported."""
    node = generate_node(
        "lrdmc-bra",
        {"aiida.out": _log("lrdmc_bra_invalid.out"), "_scheduler-stderr.txt": _log("lrdmc_bra_invalid.err")},
    )
    results, status, message = _parse(node)
    assert status == JqmcCalculation.exit_codes.ERROR_INVALID_INPUT.status
    assert "num_gfmc_warmup_steps (10) must be >= 25" in message
    assert results["output_parameters"]["exception"].startswith("ValueError")


def test_exception_during_run(generate_node):
    """An exception after sampling started is not an input error."""
    text = _log("mcmc.out")
    crashed = text[: text.index("End MCMC")]
    node = generate_node(
        "mcmc", {"aiida.out": crashed, "_scheduler-stderr.txt": "Traceback ...\nFloatingPointError: overflow\n"}
    )
    _, status, message = _parse(node)
    assert status == JqmcCalculation.exit_codes.ERROR_JQMC_EXCEPTION.status
    assert "FloatingPointError: overflow" in message


def test_no_checkpoint(generate_node):
    """The program ended but did not report writing the checkpoint."""
    text = _log("mcmc.out").replace("Dump restart checkpoint file(s) to restart.h5.", "")
    _, status, _ = _parse(generate_node("mcmc", {"aiida.out": text}))
    assert status == JqmcCalculation.exit_codes.ERROR_NO_CHECKPOINT.status


def test_non_finite_energy(generate_node):
    """A NaN energy is a broken result: exit 320, and output_parameters can still be stored (NaN as null)."""
    text, count = re.subn(r"Total Energy: E = \S+ \+- \S+ Ha\.", "Total Energy: E = nan +- nan Ha.", _log("mcmc.out"))
    assert count == 1
    results, status, _ = _parse(generate_node("mcmc", {"aiida.out": text}))
    assert status == JqmcCalculation.exit_codes.ERROR_NON_FINITE_RESULT.status
    out = results["output_parameters"]
    out.store()  # AiiDA refuses to store NaN or infinity
    assert out["energy"] is None and out["energy_error"] is None
    assert {"energy", "energy_error"} <= set(out["non_finite_values"])


def test_program_ended_overrides_scheduler_verdict(generate_node):
    """jqmc ended and wrote its checkpoint: success, even when the scheduler then reported e.g. the wall time."""
    node = generate_node("mcmc", {"aiida.out": _log("mcmc.out")})
    node.set_exit_status(JqmcCalculation.exit_codes.ERROR_SCHEDULER_OUT_OF_WALLTIME.status)
    exit_code = JqmcParser(node).parse()
    assert exit_code is not None and exit_code.status == 0  # explicit, so CalcJob.parse uses it over the scheduler's


def test_parser_jqmc_recorded(generate_node):
    """The jqmc that parsed the log is recorded, with a warning when it is not the jqmc that wrote the log."""
    from jqmc._build_info import build_info

    commit = build_info().git_commit
    text = re.sub(r"git commit = \S+", f"git commit = {commit}", _log("mcmc.out"))
    results, status, _ = _parse(generate_node("mcmc", {"aiida.out": text}))
    out = results["output_parameters"].get_dict()
    assert status == 0
    assert out["parser_jqmc"]["git_commit"] == commit
    assert not any("parsed by jqmc" in warning for warning in out["warnings"])

    other = re.sub(r"git commit = \S+", "git commit = " + "0" * 40, _log("mcmc.out"))
    out = _parse(generate_node("mcmc", {"aiida.out": other}))[0]["output_parameters"].get_dict()
    if commit != "unknown":
        assert any("parsed by jqmc" in warning for warning in out["warnings"])


def test_vmc_output_hamiltonian(generate_node, tmp_path):
    """VMC attaches the last dumped Hamiltonian from the temporary retrieved folder."""
    shutil.copy(DATA_DIR / "H2_ecp_ccpvdz_jsd.h5", tmp_path / "hamiltonian_data_opt_step_3.h5")
    results, status, _ = _parse(generate_node("vmc", {"aiida.out": _log("vmc.out")}), str(tmp_path))
    assert status == 0
    assert results["output_hamiltonian"].filename == "hamiltonian_data_opt_step_3.h5"
    assert [s["step"] for s in results["output_parameters"]["steps"]] == [1, 2, 3]


def test_vmc_all_hamiltonians(generate_node, tmp_path):
    """With `all_hamiltonians`, every dump is stored under its step; the last one is also `output_hamiltonian`."""
    dumps = [f"hamiltonian_data_opt_step_{step}.h5" for step in (1, 2, 3)]
    for name in dumps:
        shutil.copy(DATA_DIR / "H2_ecp_ccpvdz_jsd.h5", tmp_path / name)
    node = generate_node("vmc", {"aiida.out": _log("vmc.out")}, all_hamiltonians=True)
    results, status, _ = _parse(node, str(tmp_path))
    assert status == 0
    assert results["output_parameters"]["dumped_hamiltonians"] == dumps
    stored = results["output_hamiltonians"]
    assert sorted(stored) == ["step_1", "step_2", "step_3"]
    assert stored["step_3"].get_content("rb") == results["output_hamiltonian"].get_content("rb")
    assert [stored[f"step_{step}"].filename for step in (1, 2, 3)] == dumps


def test_vmc_output_hamiltonian_missing(generate_node, tmp_path):
    """The dumped Hamiltonian was not retrieved."""
    _, status, _ = _parse(generate_node("vmc", {"aiida.out": _log("vmc.out")}), str(tmp_path))
    assert status == JqmcCalculation.exit_codes.ERROR_OUTPUT_HAMILTONIAN_MISSING.status


def test_twist_average_finished(generate_node, twist_hamiltonian):
    """A twist run logs no checkpoint line; its BZ-averaged result marks it complete (8 twists)."""
    log = _log("cluster/mcmc_twist_stress_mpc_high.out")
    results, status, _ = _parse(generate_node("mcmc", {"aiida.out": log}, hamiltonian=twist_hamiltonian))
    assert status == 0
    assert len(results["output_parameters"]["twist_average"]["per_twist"]) == 8


def test_twist_average_vmc_finished(generate_node, twist_hamiltonian, tmp_path):
    """A twist VMC has no BZ-averaged energy to report: it is complete without one, and its combined dump is
    the output Hamiltonian."""
    with twist_hamiltonian.as_path() as path:
        shutil.copy(path, tmp_path / "hamiltonian_data_opt_step_1.h5")
    log = _log("localhost/vmc_twist_8_twists_low.out")
    results, status, _ = _parse(generate_node("vmc", {"aiida.out": log}, hamiltonian=twist_hamiltonian), str(tmp_path))
    assert status == 0
    assert results["output_parameters"]["twist_average"] is None
    assert results["output_hamiltonian"].filename == "hamiltonian_data_opt_step_1.h5"


def test_twist_average_stopped_by_max_time(generate_node, twist_hamiltonian):
    """jqmc marks twists stopped on max_time as complete, so such a run is not continued."""
    log = _log("cluster/mcmc_twist_stopped_by_max_time.out")
    _, status, _ = _parse(generate_node("mcmc", {"aiida.out": log}, hamiltonian=twist_hamiltonian))
    assert status == JqmcCalculation.exit_codes.ERROR_TWIST_STOPPED_EARLY.status


def test_twist_average_without_result(generate_node, twist_hamiltonian):
    text = _log("cluster/mcmc_twist_stress_mpc_high.out").replace("BZ-averaged Energy", "BZ averaged energy")
    _, status, _ = _parse(generate_node("mcmc", {"aiida.out": text}, hamiltonian=twist_hamiltonian))
    assert status == JqmcCalculation.exit_codes.ERROR_TWIST_RESULT_MISSING.status
