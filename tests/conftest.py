"""Shared fixtures for aiida-jqmc tests."""

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
import pathlib
import shutil
import sys

import pytest

pytest_plugins = ["aiida.tools.pytest_fixtures"]

DATA_DIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def data_with_pseudopotentials(tmp_path_factory) -> pathlib.Path:
    """A mirror of `DATA_DIR` whose Quantum ESPRESSO saves also hold their pseudopotential.

    The pseudopotentials are third-party files that git does not carry:
    tests/data/pseudopotentials.toml pins them and tests/_test_data_cache.py
    provides them from the host-local cache, downloading a missing one once.
    """
    from ._test_data_cache import build_data_mirror

    return build_data_mirror(tmp_path_factory.mktemp("data"))


def which_or_skip(name: str) -> str:
    """The path of executable `name`; skip the test if it is not on PATH (a failure under ``--fail-on-skip``)."""
    executable = shutil.which(name)
    if executable is None:
        pytest.skip(f"`{name}` is not on PATH")
    return executable


def keep_logs_of_failed_jobs(workchain) -> None:
    """Warn about every job of `workchain` that did not finish ok and keep its logs.

    A work chain recovers from such a job, so the test passes; the warning and,
    when AIIDA_JQMC_KEEP_LOGS names a directory, the job's stdout and scheduler
    stderr there are what is left to find out why it failed.
    """
    import warnings

    keep = os.environ.get("AIIDA_JQMC_KEEP_LOGS")
    for job in workchain.called:
        if job.is_finished_ok:
            continue
        logs = {}
        if "retrieved" in job.outputs:
            for name in (job.get_option("output_filename"), job.get_option("scheduler_stderr")):
                if name and name in job.outputs.retrieved.base.repository.list_object_names():
                    logs[name] = job.outputs.retrieved.base.repository.get_object_content(name)
        tail = "\n".join(logs.get(job.get_option("output_filename"), "").splitlines()[-20:])
        warnings.warn(f"job {job.pk} ({job.uuid}): exit {job.exit_status} {job.exit_message}\n{tail}", stacklevel=2)
        if keep:
            target = pathlib.Path(keep) / job.uuid
            target.mkdir(parents=True, exist_ok=True)
            for name, content in logs.items():
                (target / name).write_text(content)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: runs jqmc on localhost with MPI (needs `jqmc` and `mpirun` on PATH)")


def pytest_addoption(parser):
    parser.addoption(
        "--fail-on-skip",
        action="store_true",
        default=False,
        help="Fail the session when any selected test or test module is skipped or xfailed (CI): a test that did "
        "not run is red. A session that selects no test already fails (pytest exit status 5).",
    )


#: Node ids of the tests and modules that did not run in this session (``--fail-on-skip``).
_NOT_RUN: list[str] = []


def pytest_runtest_logreport(report):
    """Record every test that did not run: a skip in any phase, or an xfail."""
    if report.skipped:
        _NOT_RUN.append(report.nodeid)


def pytest_collectreport(report):
    """Record a collection-time skip (``pytest.skip(allow_module_level=True)``, ``importorskip``): its tests never run."""
    if report.skipped:
        _NOT_RUN.append(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    """``--fail-on-skip``: any skipped or xfailed test or module fails the session, ``--collect-only`` included."""
    if not session.config.getoption("--fail-on-skip") or not _NOT_RUN:
        return
    listing = "\n".join(f"  - {nodeid}" for nodeid in sorted(set(_NOT_RUN)))
    print(f"\nERROR: --fail-on-skip: {len(set(_NOT_RUN))} test(s) or module(s) did not run:\n{listing}", file=sys.stderr)
    if session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture
def hamiltonian():
    """A small JSD Hamiltonian of H2 (ccECP, cc-pVDZ) as SinglefileData."""
    from aiida import orm

    return orm.SinglefileData(DATA_DIR / "H2_ecp_ccpvdz_jsd.h5")


@pytest.fixture
def twist_hamiltonian():
    """A combined twist-averaged Hamiltonian written by jqmc-tool (tiled Li, two supertwists) as SinglefileData."""
    from aiida import orm

    return orm.SinglefileData(DATA_DIR / "tiling" / "hamiltonian" / "lithium_tiled_211_twist_average.h5")


@pytest.fixture
def jqmc_code(aiida_code_installed, aiida_localhost):
    """The `jqmc` executable on PATH registered as an InstalledCode."""
    executable = which_or_skip("jqmc")
    aiida_localhost.set_default_mpiprocs_per_machine(2)
    return aiida_code_installed(filepath_executable=executable, computer=aiida_localhost, with_mpi=True)


@pytest.fixture
def jqmc_tool_code(aiida_code_installed, aiida_localhost):
    """The `jqmc-tool` executable on PATH registered as a serial InstalledCode."""
    executable = which_or_skip("jqmc-tool")
    return aiida_code_installed(filepath_executable=executable, computer=aiida_localhost, with_mpi=False)


def assert_complete(node, job_type: str) -> dict:
    """The run finished, its output matches the schema, and every printed number was extracted."""
    import json

    import jsonschema

    from aiida_jqmc.schemas import output_parameters_schema

    assert node.is_finished_ok, (node.exit_status, node.exit_message)
    out = node.outputs.output_parameters.get_dict()
    schema = output_parameters_schema(job_type)
    jsonschema.validate(json.loads(json.dumps(out)), schema, cls=jsonschema.Draft202012Validator)
    assert out["unparsed_numeric_lines"] == [], out["unparsed_numeric_lines"]
    return out


def assert_tool_complete(node) -> dict:
    """The jqmc-tool run finished and its output_parameters (if any) match the schema."""
    import json

    import jsonschema

    from aiida_jqmc.schemas import tool_commands_with_parameters, tool_output_parameters_schema

    assert node.is_finished_ok, (node.exit_status, node.exit_message)
    command = node.inputs.command.value
    if command not in tool_commands_with_parameters():
        return {}
    out = node.outputs.output_parameters.get_dict()
    schema = tool_output_parameters_schema(command)
    jsonschema.validate(json.loads(json.dumps(out)), schema, cls=jsonschema.Draft202012Validator)
    return out


@pytest.fixture
def options():
    """Scheduler options for 2 MPI processes on one machine, one hour of wall time."""
    return {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 2}, "max_wallclock_seconds": 3600}
