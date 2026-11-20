"""Tests for JqmcToolParser, on jqmc-tool outputs recorded on a cluster.

The inputs of the recorded runs: 120 MPI ranks; MCMC with 8 walkers per rank,
post-processed with ``-b 10``; LRDMC post-processed with ``-b 10``. Printed
values change between versions, so only their presence and the quantities
fixed by those inputs are checked.
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

import math

import pytest
from aiida import orm
from aiida.common.links import LinkType
from aiida.plugins import ParserFactory

from .conftest import DATA_DIR

JqmcToolParser = ParserFactory("jqmc.tool")


def _recorded(name: str) -> dict[str, str]:
    files = {"aiida.out": (DATA_DIR / "tool" / f"{name}.out").read_text()}
    stderr = DATA_DIR / "tool" / f"{name}.err"
    if stderr.exists():
        files["_scheduler-stderr.txt"] = stderr.read_text()
    return files


@pytest.fixture
def generate_node(aiida_localhost):
    """A stored CalcJobNode of `command` with a `retrieved` folder holding the given files."""

    def factory(command: str, files: dict[str, str | bytes], **inputs) -> orm.CalcJobNode:
        node = orm.CalcJobNode(computer=aiida_localhost, process_type="aiida.calculations:jqmc.tool")
        node.set_option("resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1})
        node.set_option("output_filename", "aiida.out")
        node.set_option("scheduler_stderr", "_scheduler-stderr.txt")
        for label, value in {"command": orm.Str(command), **inputs}.items():
            node.base.links.add_incoming(value.store(), link_type=LinkType.INPUT_CALC, link_label=label)
        node.store()
        retrieved = orm.FolderData()
        for name, content in files.items():
            retrieved.base.repository.put_object_from_bytes(content if isinstance(content, bytes) else content.encode(), name)
        retrieved.base.links.add_incoming(node, link_type=LinkType.CREATE, link_label="retrieved")
        retrieved.store()
        return node

    return factory


def _parse(node):
    results, calcfunction = JqmcToolParser.parse_from_node(node, store_provenance=False)
    return results, calcfunction.exit_status, calcfunction.exit_message


@pytest.mark.parametrize(
    ("command", "name", "num_binned_samples"),
    [("mcmc compute-energy", "mcmc_energy", 10 * 120 * 8), ("lrdmc compute-energy", "lrdmc_energy", 10)],
)
def test_energy(generate_node, command, name, num_binned_samples):
    results, status, _ = _parse(generate_node(command, _recorded(name)))
    assert status == 0
    out = results["output_parameters"].get_dict()
    assert math.isfinite(out["energy"]) and out["energy_error"] > 0
    # Neither run had an MPC estimate or twists.
    for name in ("dE_mpc", "dE_mpc_error", "energy_mpc", "energy_mpc_error", "num_twists"):
        assert out[name] is None
    assert out["num_mpi_ranks"] == 120
    assert out["num_binned_samples"] == num_binned_samples


@pytest.mark.parametrize("command", ["trexio convert-to", "qe"])
def test_conversion(generate_node, command):
    name = {"trexio convert-to": "convert", "qe": "qe"}[command]
    files = {**_recorded(name), "hamiltonian_data.h5": b"\x89HDF\r\n\x1a\n"}
    results, status, _ = _parse(generate_node(command, files))
    assert status == 0
    assert results["output_hamiltonian"].filename == "hamiltonian_data.h5"
    assert "output_parameters" not in results


def test_tiled_conversion_without_tiled_orbitals_is_invalid(generate_node, aiida_code_installed, aiida_localhost):
    """A conversion that asked for `tiling` but wrote untiled orbitals is not passed on as a tiled Hamiltonian."""
    files = {**_recorded("convert"), "hamiltonian_data.h5": (DATA_DIR / "H2_ecp_ccpvdz_jsd.h5").read_bytes()}
    code = aiida_code_installed(filepath_executable="/bin/true", computer=aiida_localhost)
    node = generate_node("trexio convert-to", files, tiling=orm.List(list=[2, 1, 1]), code=code)
    results, status, message = _parse(node)
    assert status == 314
    assert "untiled determinant" in message
    assert not {"output_hamiltonian", "conversion_metadata", "physical_structure"} & set(results)


def test_scheduler_verdict_is_kept(generate_node):
    """A post-processing run cut off by the scheduler keeps that verdict (e.g. the wall time)."""
    text = _recorded("mcmc_energy")["aiida.out"]
    node = generate_node("mcmc compute-energy", {"aiida.out": text.splitlines()[0]})
    node.set_exit_status(120)
    _, status, _ = _parse(node)
    assert status == 120


def test_conversion_with_exception_is_not_kept(generate_node):
    """jqmc-tool writes the file in place: after an exception the file is not taken as the result."""
    files = {
        **_recorded("convert"),
        "_scheduler-stderr.txt": "Traceback (most recent call last):\nValueError: broken basis\n",
        "hamiltonian_data.h5": b"\x89HDF\r\n\x1a\n",
    }
    results, status, message = _parse(generate_node("trexio convert-to", files))
    assert status == 303
    assert "broken basis" in message
    assert "output_hamiltonian" not in results


def test_environment_from_stderr(generate_node):
    """The version and git commit jqmc-tool prints on stderr are kept in output_parameters."""
    files = _recorded("mcmc_energy")
    commit = "1" * 40
    # the last banner counts (a recorded stderr may hold one of its own)
    files["_scheduler-stderr.txt"] = files.get("_scheduler-stderr.txt", "") + f"version = 1.2.3\ngit commit = {commit}\n"
    results, status, _ = _parse(generate_node("mcmc compute-energy", files))
    assert status == 0
    out = results["output_parameters"].get_dict()
    assert out["environment"] == {"jqmc_version": "1.2.3", "git_commit": commit}
    assert out["parser_jqmc"]["git_commit"]
    assert out["non_finite_values"] == []


def test_conversion_without_output(generate_node):
    results, status, message = _parse(generate_node("trexio convert-to", _recorded("convert")))
    assert status == 312
    assert "hamiltonian_data.h5 was not written" in message
    assert "output_hamiltonian" not in results


@pytest.mark.parametrize(
    ("command", "name", "message"),
    [
        ("mcmc compute-force", "mcmc_force_disabled", "atomic_force was not enabled"),
        ("lrdmc extrapolate-energy", "extrapolate_one", "needs at least 2 distinct lattice spacings"),
    ],
)
def test_reported_error(generate_node, command, name, message):
    """jqmc-tool's own error message is the exit message, and is kept in output_parameters."""
    results, status, exit_message = _parse(generate_node(command, _recorded(name)))
    assert status == 303
    assert message in exit_message
    assert message in results["output_parameters"]["exception"]


def test_not_started(generate_node):
    _, status, _ = _parse(generate_node("mcmc compute-energy", {"aiida.out": ""}))
    assert status == 305


def test_stdout_missing(generate_node):
    _, status, _ = _parse(generate_node("mcmc compute-energy", {}))
    assert status == 301
