"""Shared helpers of the examples: the jQMC CLI tutorials, run through AiiDA.

Each ``jqmc-exampleNN/run_aiida.py`` follows the commands of the tutorial of the
same name in jQMC ``examples/`` (of the installed jqmc version), reading its TOML
inputs, scripts and QE files from there (``JQMC_EXAMPLES``). Mean-field steps (PySCF, ``pw.x``) run as
aiida-shell jobs, the jqmc and jqmc-tool steps as JqmcRestartWorkChain and
JqmcToolCalculation, so that the whole chain is in the provenance graph.

The tutorials' published run lengths are substantial. ``SCALE`` (default
``0.02``) shortens the runs read with ``tutorial_input`` to that fraction; set
``JQMC_EXAMPLES_SCALE=1`` for the published sizes. A script says where it
departs from its tutorial (see also ``README.md``).

Computers and codes come from ``examples/config.toml`` (see
``config.example.toml``).
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

import copy
import math
import os
import pathlib

import toml
from aiida import load_profile, orm
from aiida.engine import run_get_node
from aiida.plugins import CalculationFactory, WorkflowFactory

from aiida_jqmc.utils import parameters_from_toml

HERE = pathlib.Path(__file__).resolve().parent
CONFIG = toml.load(HERE / "config.toml")
load_profile(CONFIG.get("profile"))

EXAMPLES = pathlib.Path(os.environ.get("JQMC_EXAMPLES", CONFIG["jqmc_examples"]))
SCALE = float(os.environ.get("JQMC_EXAMPLES_SCALE", CONFIG.get("scale", 0.02)))

JqmcToolCalculation = CalculationFactory("jqmc.tool")
JqmcRestartWorkChain = WorkflowFactory("jqmc.restart")

# Run-length keys and their minimums (jqmc rejects shorter warmups / fewer bins).
_STEP_KEYS = {
    "mcmc": ("num_mcmc_steps",),
    "vmc": ("num_opt_steps", "num_mcmc_steps"),
    "lrdmc-bra": ("num_mcmc_steps",),
    "lrdmc-tau": ("num_mcmc_steps",),
}


def report(label: str, node) -> None:
    status = "ok" if node.is_finished_ok else f"FAILED [{node.exit_status}] {node.exit_message}"
    print(f"{label}: {node.process_label}<{node.pk}> {status}", flush=True)
    if not node.is_finished_ok:
        raise SystemExit(f"{label} failed")


def shortened(parameters: dict, scale: float = SCALE) -> dict:
    """Scale the run length of a tutorial input, keeping warmup and binning valid."""
    parameters = copy.deepcopy(parameters)
    job_type = parameters["control"]["job_type"]
    section = parameters[job_type]
    if scale >= 1:
        return parameters
    for key in _STEP_KEYS[job_type]:
        if key in section:
            section[key] = max(1, math.ceil(section[key] * scale))
    if job_type == "vmc":
        section["num_opt_steps"] = max(2, section.get("num_opt_steps", 2))
        section["num_mcmc_steps"] = max(section.get("num_mcmc_steps", 0), 100)
    elif job_type == "mcmc":
        warmup = section.get("num_mcmc_warmup_steps", 0)
        bins = section.get("num_mcmc_bin_blocks", 1)
        section["num_mcmc_steps"] = max(section["num_mcmc_steps"], warmup + 10 * bins)
    else:
        warmup = section.get("num_gfmc_warmup_steps", 0)
        bins = section.get("num_gfmc_bin_blocks", 1)
        section["num_mcmc_steps"] = max(section["num_mcmc_steps"], warmup + 10 * bins)
    return parameters


def tutorial_input(path: str | pathlib.Path, scale: float = SCALE, **updates) -> dict:
    """``parameters`` from a tutorial TOML, shortened; ``updates`` are {"section.key": value}."""
    parameters, _ = parameters_from_toml(EXAMPLES / path)
    parameters = shortened(parameters, scale)
    for dotted, value in updates.items():
        section, key = dotted.split(".")
        parameters.setdefault(section, {})[key] = value
    return parameters


def _options(kind: str) -> dict:
    return copy.deepcopy(CONFIG["options"][kind])


def code(name: str) -> orm.AbstractCode:
    return orm.load_code(CONFIG["codes"][name])


def shell(label: str, name: str, arguments: str, nodes: dict, outputs: list | None = None, kind: str = "shell"):
    """Run a program (PySCF script, pw.x) as an aiida-shell job."""
    from aiida_shell import launch_shell_job

    results, node = launch_shell_job(
        code(name),
        arguments=arguments,
        nodes=nodes,
        outputs=outputs or [],
        metadata={"options": _options(kind)},
        submit=False,
    )
    report(label, node)
    return results, node


def tool(label: str, command: str, options: dict | None = None, kind: str = "tool", **inputs):
    """Run one jqmc-tool subcommand."""
    builder = JqmcToolCalculation.get_builder()
    builder.code = code(CONFIG["tool_code_for"][kind])
    builder.command = orm.Str(command)
    if options:
        builder.options = orm.Dict(options)
    for key, value in inputs.items():
        setattr(builder, key, value)
    builder.metadata.options = _options(kind)
    _, node = run_get_node(builder)
    report(label, node)
    return node


def jqmc(
    label: str,
    parameters,
    hamiltonian=None,
    parent_folder=None,
    kind: str = "jqmc",
    max_iterations: int = 10,
    code_name: str = "jqmc",
):
    """Run one jqmc calculation to completion (JqmcRestartWorkChain); ``parameters`` is a dict or a Dict node."""
    builder = JqmcRestartWorkChain.get_builder()
    builder.jqmc.code = code(code_name)
    builder.jqmc.parameters = parameters if isinstance(parameters, orm.Dict) else orm.Dict(parameters)
    if hamiltonian is not None:
        builder.jqmc.hamiltonian = hamiltonian
    if parent_folder is not None:
        builder.jqmc.parent_folder = parent_folder
    builder.jqmc.metadata.options = _options(kind)
    builder.max_iterations = orm.Int(max_iterations)
    _, node = run_get_node(builder)
    report(label, node)
    return node


def file(path: str | pathlib.Path, filename: str | None = None) -> orm.SinglefileData:
    path = EXAMPLES / path
    return orm.SinglefileData(path, filename=filename or path.name)


def show(label: str, node) -> None:
    """Print the non-null results of a calculation."""
    if "output_parameters" in node.outputs:
        values = {k: v for k, v in node.outputs.output_parameters.get_dict().items() if v not in (None, [], {})}
        print(f"    {label}: {values}", flush=True)


def continuation(parameters: dict, steps: int) -> dict:
    """Input that adds ``steps`` to a finished calculation: the same input without the keys
    jqmc restores from the checkpoint, and the new number of steps."""
    from aiida_jqmc.workflows.restart import STEP_KEYS, _without_frozen_keys

    job_type = parameters["control"]["job_type"]
    extended = _without_frozen_keys(copy.deepcopy(parameters), job_type)
    extended[job_type][STEP_KEYS[job_type][0]] = steps
    return extended


def sample_to_target(label: str, parameters: dict, target_error: float, hamiltonian=None, extensions: int = 2, **kwargs):
    """A pilot run, then extensions until the energy error bar reaches ``target_error``.

    The decision that jqmc_workflow takes automatically, written out as a caller (an
    agent) takes it with aiida_jqmc.statistics: estimate, extend, look again.
    """
    from aiida_jqmc.statistics import steps_to_target_error

    node = jqmc(f"{label} (pilot)", parameters, hamiltonian=hamiltonian, **kwargs)
    for index in range(extensions):
        more = steps_to_target_error(node, target_error)
        error = node.outputs.output_parameters["energy_error"]
        print(
            f"    {label}: error {error:.2e} after {index} extension(s); {more} more steps for {target_error:.1e}", flush=True
        )
        if more == 0:
            break
        node = jqmc(f"{label} (+{more})", continuation(parameters, more), parent_folder=node.outputs.remote_folder, **kwargs)
    return node


# Target error bars of the tutorials are for their run lengths; shortened runs aim at proportionally
# larger ones (error ~ 1/sqrt(N)).
def target(error: float) -> float:
    return error / math.sqrt(min(SCALE, 1.0))
