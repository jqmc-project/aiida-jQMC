"""End-to-end runs of every job type and every jqmc-tool subcommand on localhost.

Each run is checked for completeness, not for values: it finishes, its
output_parameters match the closed schema, every printed number is extracted,
and the counters equal what the input asked for.
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

import pytest
from aiida import orm
from aiida.engine import run_get_node
from aiida.plugins import CalculationFactory

from .conftest import DATA_DIR, assert_complete, assert_tool_complete

JqmcCalculation = CalculationFactory("jqmc.jqmc")
JqmcToolCalculation = CalculationFactory("jqmc.tool")

pytestmark = pytest.mark.slow

WALKERS = 4


def _tool(code, command: str, **inputs):
    if "options" in inputs:
        inputs["options"] = orm.Dict(inputs["options"])
    _, node = run_get_node(JqmcToolCalculation, code=code, command=orm.Str(command), **inputs)
    return node


def _jqmc(code, parameters: dict, options: dict, **inputs):
    _, node = run_get_node(JqmcCalculation, code=code, parameters=orm.Dict(parameters), metadata={"options": options}, **inputs)
    return node


def _control(job_type: str) -> dict:
    return {"job_type": job_type, "number_of_walkers": WALKERS, "mcmc_seed": 1, "verbosity": "high"}


def test_trexio_and_hamiltonian_tools(jqmc_tool_code, jqmc_code, options):
    """TREXIO -> Hamiltonian -> JAGP, with the info and xyz tools; the JAGP trial runs."""
    trexio = orm.SinglefileData(DATA_DIR / "H2_ecp_ccpvdz_cart.h5")

    info = _tool(jqmc_tool_code, "trexio show-info", input_file=trexio)
    assert len(assert_tool_complete(info)["atoms"]) == 2

    convert = _tool(jqmc_tool_code, "trexio convert-to", input_file=trexio, options={"-j2": 1.0, "-j3": "mo"})
    assert_tool_complete(convert)
    hamiltonian = convert.outputs.output_hamiltonian

    info = _tool(jqmc_tool_code, "hamiltonian show-info", input_file=hamiltonian)
    assert len(assert_tool_complete(info)["atoms"]) == 2

    xyz = _tool(jqmc_tool_code, "hamiltonian to-xyz", input_file=hamiltonian)
    assert_tool_complete(xyz)
    lines = xyz.outputs.output_structure.get_content().splitlines()
    assert int(lines[0]) == 2 and len(lines) == 4

    jagp = _tool(jqmc_tool_code, "hamiltonian conv-wf", input_file=hamiltonian, options={"-c": "jagp"})
    assert_tool_complete(jagp)

    vmc = {
        "control": _control("vmc"),
        "vmc": {"num_mcmc_steps": 40, "num_mcmc_warmup_steps": 10, "num_mcmc_bin_blocks": 5, "num_opt_steps": 1},
    }
    node = _jqmc(jqmc_code, vmc, options, hamiltonian=jagp.outputs.output_hamiltonian)
    assert len(assert_complete(node, "vmc")["steps"]) == 1


def test_tool_errors_are_reported(jqmc_tool_code):
    """jqmc-tool's own error text reaches the exit message."""
    trexio = orm.SinglefileData(DATA_DIR / "H2_ecp_ccpvdz_cart.h5")
    node = _tool(jqmc_tool_code, "trexio convert-to", input_file=trexio, options={"-j3": "not-a-basis"})
    assert node.exit_status == JqmcToolCalculation.exit_codes.ERROR_TOOL_FAILED.status
    assert "not-a-basis" in node.exit_message


def test_vmc_at_high_verbosity_keeps_every_dump(jqmc_code, jqmc_tool_code, hamiltonian, options):
    """Every optimizer field is filled, every dumped Hamiltonian is stored, and analyze-output plots the run."""
    vmc = {
        "control": _control("vmc"),
        "vmc": {"num_mcmc_steps": 40, "num_mcmc_warmup_steps": 10, "num_mcmc_bin_blocks": 5, "num_opt_steps": 3},
    }
    node = _jqmc(jqmc_code, vmc, {**options, "all_hamiltonians": True}, hamiltonian=hamiltonian)
    out = assert_complete(node, "vmc")
    assert [step["step"] for step in out["steps"]] == [1, 2, 3]
    for step in out["steps"]:
        assert step["asr"] and step["diagnostics"] and step["top_coordinates"] and step["blocks"]
        assert (step["num_mcmc_steps"], step["num_mcmc_warmup_steps"], step["num_mcmc_bin_blocks"]) == (40, 10, 5)
    assert sorted(node.outputs.output_hamiltonians) == ["step_1", "step_2", "step_3"]
    assert node.outputs.output_hamiltonians.step_3.filename == node.outputs.output_hamiltonian.filename

    analysis = _tool(
        jqmc_tool_code, "vmc analyze-output", parent_folders={"vmc": node.outputs.remote_folder}, options={"-s": "vmc.png"}
    )
    assert_tool_complete(analysis)
    assert analysis.outputs.output_graph.filename == "graph.png"

    # the dumped steps (all with the same file name) averaged into one Hamiltonian
    average = _tool(
        jqmc_tool_code,
        "vmc average-wf",
        input_files={k: node.outputs.output_hamiltonians[k] for k in node.outputs.output_hamiltonians},
    )
    assert_tool_complete(average)
    assert average.outputs.output_hamiltonian.filename == "hamiltonian_data.h5"


def test_mcmc_with_forces_and_its_post_processing(jqmc_code, jqmc_tool_code, hamiltonian, options):
    mcmc = {
        "control": _control("mcmc"),
        "mcmc": {"num_mcmc_steps": 80, "num_mcmc_warmup_steps": 10, "num_mcmc_bin_blocks": 5, "atomic_force": True},
    }
    node = _jqmc(jqmc_code, mcmc, options, hamiltonian=hamiltonian)
    out = assert_complete(node, "mcmc")
    assert out["mcmc_counter"] == 80
    assert len(out["atomic_forces"]) == 2

    parents = {"mcmc": node.outputs.remote_folder}
    energy = _tool(jqmc_tool_code, "mcmc compute-energy", parent_folders=parents, options={"-b": 5, "-w": 10})
    assert assert_tool_complete(energy)["energy"] is not None
    force = _tool(jqmc_tool_code, "mcmc compute-force", parent_folders=parents, options={"-b": 5, "-w": 10})
    assert len(assert_tool_complete(force)["atomic_forces"]) == 2


def test_lrdmc_bra_with_forces_and_extrapolation(jqmc_code, jqmc_tool_code, hamiltonian, options):
    """lrdmc-bra at two spacings, post-processed one by one and extrapolated with a saved graph."""
    remote = {}
    for alat in (0.3, 0.2):
        lrdmc = {
            "control": _control("lrdmc-bra"),
            "lrdmc-bra": {
                "num_mcmc_steps": 80,
                "alat": alat,
                "num_mcmc_per_measurement": 40,
                "num_gfmc_warmup_steps": 25,
                "num_gfmc_bin_blocks": 10,
                "num_gfmc_collect_steps": 5,
                "atomic_force": True,
            },
        }
        node = _jqmc(jqmc_code, lrdmc, options, hamiltonian=hamiltonian)
        out = assert_complete(node, "lrdmc-bra")
        assert out["gfmc_counter"] == 80
        assert len(out["atomic_forces"]) == 2
        remote[f"a{round(alat * 100):03d}"] = node.outputs.remote_folder

    one = {"a030": remote["a030"]}
    post = {"-b": 10, "-w": 25, "-c": 5}
    assert assert_tool_complete(_tool(jqmc_tool_code, "lrdmc compute-energy", parent_folders=one, options=post))["energy"]
    force = _tool(jqmc_tool_code, "lrdmc compute-force", parent_folders=one, options=post)
    assert len(assert_tool_complete(force)["atomic_forces"]) == 2

    extrapolation = _tool(
        jqmc_tool_code, "lrdmc extrapolate-energy", parent_folders=remote, options={**post, "-p": 1, "-s": "a.pdf"}
    )
    assert assert_tool_complete(extrapolation)["extrapolated_energy"] is not None
    assert extrapolation.outputs.output_graph.filename == "graph.pdf"


def test_lrdmc_tau(jqmc_code, hamiltonian, options):
    lrdmc = {
        "control": _control("lrdmc-tau"),
        "lrdmc-tau": {
            "num_mcmc_steps": 80,
            "alat": 0.3,
            "tau": 0.1,
            "num_gfmc_warmup_steps": 25,
            "num_gfmc_bin_blocks": 10,
            "num_gfmc_collect_steps": 5,
        },
    }
    out = assert_complete(_jqmc(jqmc_code, lrdmc, options, hamiltonian=hamiltonian), "lrdmc-tau")
    assert out["gfmc_counter"] == 80
