"""The output schemas describe every field extracted from real logs.

The logs come from a cluster (120 MPI ranks) and localhost runs. Every object of
the schemas is closed, so a field that jqmc.output_parser returns but the
schema does not describe fails here.
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

import json

import jsonschema
import pytest
from jqmc.output_parser import parse_jqmc_log, parse_lrdmc_extrapolation, parse_tool_energy, parse_tool_force

from aiida_jqmc.parsers.jqmc import add_plugin_fields
from aiida_jqmc.parsers.tool import add_tool_fields
from aiida_jqmc.schemas import output_parameters_schema, tool_commands_with_parameters, tool_output_parameters_schema

from .conftest import DATA_DIR

LOGS = [
    ("cluster/mcmc_high.out", "mcmc"),
    ("cluster/vmc_high.out", "vmc"),
    ("cluster/vmc_restart_low.out", "vmc"),
    ("cluster/lrdmc_bra_high.out", "lrdmc-bra"),
    ("cluster/lrdmc_bra_max_time_low.out", "lrdmc-bra"),
    ("cluster/lrdmc_tau_high.out", "lrdmc-tau"),
    ("cluster/mcmc_twist_stress_mpc_high.out", "mcmc"),
    ("cluster/mcmc_twist_stopped_by_max_time.out", "mcmc"),
    ("cluster/vmc_bspline_dtn_high.out", "vmc"),
    ("cluster/mcmc_mixed_mpc_high.out", "mcmc"),
    ("cluster/vmc_jagp_low.out", "vmc"),
    ("localhost/mcmc_twist_stopped_8_twists.out", "mcmc"),
    ("localhost/vmc_twist_8_twists_low.out", "vmc"),
    ("localhost/mcmc_pbc_pgto_forces_stress_mpc_high.out", "mcmc"),
    ("localhost/vmc_mo_high.out", "vmc"),
    ("localhost/vmc_sr_high.out", "vmc"),
    ("localhost/vmc_lm_low.out", "vmc"),
    ("localhost/vmc_lm_high.out", "vmc"),
    ("mcmc.out", "mcmc"),
    ("mcmc_max_time.out", "mcmc"),
    ("vmc.out", "vmc"),
    ("lrdmc_bra_invalid.out", "lrdmc-bra"),
]


def _stored(result: dict) -> dict:
    """As stored in an AiiDA Dict (JSON: tuples become lists)."""
    return json.loads(json.dumps(result))


@pytest.mark.parametrize(("name", "job_type"), LOGS)
def test_jqmc_outputs_match_the_schema(name, job_type):
    result = add_plugin_fields(parse_jqmc_log((DATA_DIR / name).read_text(), job_type))
    jsonschema.validate(_stored(result), output_parameters_schema(job_type), cls=jsonschema.Draft202012Validator)
    assert result["unparsed_numeric_lines"] == [], "every printed number is extracted"


def test_twist_stress_and_forces_are_described():
    """A twist-averaged MCMC with forces, stress and MPC (8 twists, 120 ranks) fills those fields."""
    result = parse_jqmc_log((DATA_DIR / "cluster" / "mcmc_twist_stress_mpc_high.out").read_text(), "mcmc")
    assert [row["twist"] for row in result["twist_average"]["per_twist"]] == list(range(8))
    assert result["twist_average"]["energy_mpc"] is not None
    assert {"sigma_xx", "sigma_yz_err"} <= set(result["stress_tensor"])
    assert {"Px", "P_hydro_err"} <= set(result["pressure"])
    assert len(result["atomic_forces"]) == 2


def test_periodic_gto_and_stopped_twist_fields_are_filled():
    """localhost runs: the AO image summary of periodic GTOs; every twist that max_time stopped (8 of 8)."""
    pbc = parse_jqmc_log((DATA_DIR / "localhost" / "mcmc_pbc_pgto_forces_stress_mpc_high.out").read_text(), "mcmc")
    assert pbc["system"]["ao_images"]["num_image_cells"] >= 1
    assert pbc["incomplete_twists"] == []
    stopped = parse_jqmc_log((DATA_DIR / "localhost" / "mcmc_twist_stopped_8_twists.out").read_text(), "mcmc")
    assert [row["twist"] for row in stopped["incomplete_twists"]] == list(range(8))
    assert all(row["steps"] < row["requested_steps"] for row in stopped["incomplete_twists"])


def test_high_verbosity_vmc_fills_the_optimizer_fields():
    """The step fields the schema marks "verbosity high" are filled from a real high-verbosity log."""
    result = parse_jqmc_log((DATA_DIR / "cluster" / "vmc_high.out").read_text(), "vmc")
    for step in result["steps"]:
        assert step["asr"] and step["diagnostics"] and step["top_coordinates"] and step["blocks"]
        assert step["optimization_batch"]["sampler_ranks"] == 120
        assert (step["num_mcmc_steps"], step["num_mcmc_warmup_steps"], step["num_mcmc_bin_blocks"]) == (1000, 100, 10)
    assert result["environment"]["process_count"] == 120
    assert result["environment"]["precision_zones"]


def test_optimizer_routes_fill_their_fields():
    """Plain SR updates without aSR and without D; LM reports its conditioning (at high verbosity also its
    decision); an MO optimization reports the S-Stiefel tangent of its MO blocks."""

    def steps(name):
        return parse_jqmc_log((DATA_DIR / "localhost" / name).read_text(), "vmc")["steps"]

    sr = steps("vmc_sr_high.out")
    assert {update["method"] for step in sr for update in step["parameter_updates"]} == {"SR"}
    assert all(source["D_shape"] is None for step in sr for source in step["derivative_sources"])
    assert all(step["derivative_sources"] and step["lm"] == {} for step in sr)
    assert all(step["lm"]["conditioning"] and "decision" not in step["lm"] for step in steps("vmc_lm_low.out"))
    assert all(step["lm"]["conditioning"] and step["lm"]["decision"] for step in steps("vmc_lm_high.out"))
    mo = steps("vmc_mo_high.out")
    assert any("s_stiefel_tangent" in block for step in mo for block in step["blocks"])


def test_asr_fallbacks_are_described():
    """Every aSR fallback line and the discriminant clipping, inserted into a real aSR log, match the schema."""
    lines = [
        "aSR: selected root is not finite/nonzero (gamma=0.0); falling back to plain SR (gamma=0.1).",
        "aSR: selected reference-connected root gamma=5.0 exceeds |gamma_max|=2.0; "
        "falling back to plain SR with the model's sign (gamma=0.1).",
        "aSR: centered metric contraction S_2=-0.5 is negative; falling back to plain SR (gamma=0.1).",
        "aSR: denom = -2*H_1*S_2 ~ 0 (H_1=0.0, S_2=1.0); no meaningful optimization direction. Returning gamma = 0.0.",
        "aSR: selected gamma produced an invalid model denominator; using gamma = 0.1.",
        "aSR: discriminant is negative (-0.2); setting it to zero.",
    ]
    text = (DATA_DIR / "localhost" / "vmc_mo_high.out").read_text()
    final = next(line for line in text.splitlines() if line.startswith("aSR: final gamma"))
    result = add_plugin_fields(parse_jqmc_log(text.replace(final, "\n".join([*lines, final]), 1), "vmc"))
    asr = result["steps"][0]["asr"]
    assert len(asr["fallbacks"]) == 5 and asr["clipped_discriminant"] == -0.2
    assert result["unparsed_numeric_lines"] == []
    jsonschema.validate(_stored(result), output_parameters_schema("vmc"), cls=jsonschema.Draft202012Validator)


@pytest.mark.parametrize(
    ("command", "name", "parse"),
    [
        ("mcmc compute-energy", "mcmc_energy", parse_tool_energy),
        ("lrdmc compute-energy", "lrdmc_energy", parse_tool_energy),
        ("mcmc compute-force", "mcmc_force_disabled", parse_tool_force),
        ("lrdmc extrapolate-energy", "extrapolate_one", parse_lrdmc_extrapolation),
    ],
)
def test_tool_outputs_match_the_schema(command, name, parse):
    result = add_tool_fields(parse((DATA_DIR / "tool" / f"{name}.out").read_text()), "")
    jsonschema.validate(_stored(result), tool_output_parameters_schema(command), cls=jsonschema.Draft202012Validator)


@pytest.mark.parametrize("job_type", ["mcmc", "vmc", "lrdmc-bra", "lrdmc-tau"])
def test_every_field_is_described(job_type):
    """Every property of the schema carries a description (the meaning, for an agent)."""

    def walk(schema, path):
        for key, value in schema.get("properties", {}).items():
            assert value.get("description") is not None, f"{path}.{key}"
            walk(value, f"{path}.{key}")
            if isinstance(value.get("items"), dict):
                walk(value["items"], f"{path}.{key}[]")

    walk(output_parameters_schema(job_type), job_type)
    for command in tool_commands_with_parameters():
        walk(tool_output_parameters_schema(command), command)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda r: r.update(unknown_field=1),
        lambda r: r["steps"][0].update(unknown_field=1),
        lambda r: r["steps"][0]["diagnostics"]["sr_solve"].update(unknown_field=1),
        lambda r: r.update(energy="-1.0"),
        lambda r: r["steps"][0]["top_coordinates"][0].pop("force_error"),
    ],
)
def test_schema_rejects_undescribed_or_mistyped_fields(corrupt):
    """The schemas are closed and typed, so they catch fields they do not describe."""
    result = _stored(parse_jqmc_log((DATA_DIR / "cluster" / "vmc_high.out").read_text(), "vmc"))
    corrupt(result)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(result, output_parameters_schema("vmc"), cls=jsonschema.Draft202012Validator)
