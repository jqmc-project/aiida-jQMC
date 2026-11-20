"""Run one jqmc calculation to completion, across as many jobs as needed."""

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

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import BaseRestartWorkChain, ProcessHandlerReport, calcfunction, process_handler, while_
from jqmc.jqmc_miscs import cli_parameters

from aiida_jqmc.calculations.jqmc import JqmcCalculation, checkpoint_name, default_max_time_margin, is_twist_average

__all__ = ("JqmcRestartWorkChain", "merge_vmc_jobs")

# job_type -> (input key holding the number of steps, output key holding the cumulative counter)
STEP_KEYS = {
    "mcmc": ("num_mcmc_steps", "mcmc_counter"),
    "vmc": ("num_opt_steps", "opt_step_counter"),
    "lrdmc-bra": ("num_mcmc_steps", "gfmc_counter"),
    "lrdmc-tau": ("num_mcmc_steps", "gfmc_counter"),
}


def _without_frozen_keys(parameters: dict, job_type: str) -> dict:
    """Drop the keys that jqmc restores from the checkpoint on restart.

    jqmc rejects a frozen key whose value differs from the checkpoint, and some of
    them change during a run (e.g. ``E_scf`` of lrdmc-bra). They were applied by
    the first job of the chain, so the continuation jobs leave them out.
    """
    for section in ("control", job_type):
        policy = cli_parameters.get(f"{section}_restart_policy", {})
        for key in [key for key in parameters.get(section, {}) if policy.get(key) == "frozen"]:
            del parameters[section][key]
    return parameters


def _counter(node: orm.CalcJobNode) -> int | None:
    """Cumulative step counter reported by a finished JqmcCalculation."""
    if "output_parameters" not in node.outputs:
        return None
    job_type = node.inputs.parameters["control"]["job_type"]
    return node.outputs.output_parameters.get(STEP_KEYS[job_type][1])


@calcfunction
def merge_vmc_jobs(**jobs: orm.Dict) -> orm.Dict:
    """The ``output_parameters`` of one VMC optimization run in several jobs (continuations, ``job_000`` first).

    Each job reports only its own steps and timings. The merged dict is that of the last job, except:

    * ``steps``: those of every job (a step done twice keeps its later record);
    * ``dumped_hamiltonians``: those of every job, in order;
    * ``total_opt_time_sec``: the sum over the jobs, and ``opt_timing_breakdown``: per step, averaged over all steps;
    * ``num_jobs``, and ``warnings`` of every job (plus one when the jobs ran different jqmc commits).
    """
    ordered = [jobs[key].get_dict() for key in sorted(jobs)]
    merged = copy.deepcopy(ordered[-1])
    steps: dict = {}
    for parameters in ordered:
        for step in parameters["steps"]:
            steps[step["step"]] = step
    merged["steps"] = [steps[number] for number in sorted(steps)]
    merged["dumped_hamiltonians"] = [name for parameters in ordered for name in parameters["dumped_hamiltonians"]]

    times = [parameters["total_opt_time_sec"] for parameters in ordered]
    merged["total_opt_time_sec"] = sum(times) if all(time is not None for time in times) else None
    counts = [len(parameters["steps"]) for parameters in ordered]
    breakdown: dict = {}
    for count, parameters in zip(counts, ordered, strict=True):
        for part, seconds in parameters["opt_timing_breakdown"].items():
            breakdown[part] = breakdown.get(part, 0.0) + seconds * count
    merged["opt_timing_breakdown"] = {part: total / sum(counts) for part, total in breakdown.items()} if sum(counts) else {}

    warnings = [warning for parameters in ordered for warning in parameters["warnings"]]
    commits = {parameters["environment"]["git_commit"] for parameters in ordered} - {None}
    if len(commits) > 1:
        warnings.append(f"The jobs of this run used different jqmc commits: {sorted(commits)}.")
    merged["warnings"] = list(dict.fromkeys(warnings))
    merged["num_jobs"] = len(ordered)
    return orm.Dict(merged)


class JqmcRestartWorkChain(BaseRestartWorkChain):
    """Run a JqmcCalculation until the requested number of steps is done.

    The ``jqmc.parameters`` input describes the *whole* calculation, exactly as for
    a single uninterrupted jqmc run: ``num_mcmc_steps`` (mcmc, lrdmc) or
    ``num_opt_steps`` (vmc) is the number of steps to perform in this work chain,
    counted on top of the parent checkpoint when ``jqmc.parent_folder`` is given.

    The work chain makes execution decisions only. Every physical or statistical
    parameter is passed to each job unchanged. It

    * continues from the latest checkpoint when a job stops on ``max_time``,
    * resubmits from the last good checkpoint when a job was cut off before
      jqmc ended (wall time, node failure), and then doubles ``max_time_margin``,
    * and stops on anything else (invalid input, exceptions in jqmc, NaN, ...),
      leaving the decision to the caller.

    The outputs are those of the last job: an MCMC or LRDMC continuation restores the whole history of
    measurements from the checkpoint, so its energy and counter cover the whole run. A VMC job reports only
    its own optimization steps, so a VMC run split into several jobs returns ``output_parameters`` with the
    ``steps`` of every job (``merge_vmc_jobs``) and the ``output_hamiltonians`` of every job.
    """

    _process_class = JqmcCalculation

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs and the outline."""
        super().define(spec)
        spec.expose_inputs(JqmcCalculation, namespace="jqmc")
        spec.expose_outputs(JqmcCalculation)

        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )

        spec.exit_code(
            410,
            "ERROR_PARENT_COUNTER_UNKNOWN",
            message="The step counter of the parent checkpoint is unknown: "
            "`jqmc.parent_folder` must come from a successful JqmcCalculation.",
        )
        spec.exit_code(411, "ERROR_NO_PROGRESS", message="A job ended without advancing the step counter.")
        spec.exit_code(
            412,
            "ERROR_TOO_MANY_INTERRUPTED_JOBS",
            message="Jobs were cut off before jqmc ended too many times in a row.",
        )
        spec.exit_code(
            413,
            "ERROR_REPEATED_CRASH",
            message="Jobs ended twice in a row without an error message and without a scheduler verdict "
            "(e.g. a segmentation fault or an MPI abort): resubmitting the same checkpoint would crash again.",
        )

    # Consecutive jobs cut off before jqmc ended (wall time, node failure), before giving up.
    _MAX_CONSECUTIVE_INTERRUPTIONS = 2
    # Consecutive jobs that ended with neither an exception nor a scheduler verdict (exit 302) before giving up: such
    # a crash is usually deterministic (the checkpoint and the random state are the same), so it is retried once.
    _MAX_CONSECUTIVE_UNEXPLAINED = 1

    def setup(self):
        """Work out the target counter from the parent checkpoint and the requested steps."""
        super().setup()
        self.ctx.inputs = AttributeDict(self.exposed_inputs(JqmcCalculation, "jqmc"))

        parameters = self.ctx.inputs.parameters.get_dict()
        job_type = parameters["control"]["job_type"]
        self.ctx.step_key, _ = STEP_KEYS[job_type]
        self.ctx.job_type = job_type

        # A twist-averaged run samples every twist to num_mcmc_steps (one job per twist group); there is
        # no counter to accumulate across jobs: it is done when the run ends with its BZ-averaged result.
        parent = self.ctx.inputs.parent_folder.creator if "parent_folder" in self.ctx.inputs else None
        if parent is not None and parent.process_class is JqmcCalculation:
            self.ctx.twist = checkpoint_name(parent) == JqmcCalculation._TWIST_CHECKPOINT_DIR
        else:
            self.ctx.twist = "hamiltonian" in self.ctx.inputs and is_twist_average(self.ctx.inputs.hamiltonian)

        start = 0
        if "parent_folder" in self.ctx.inputs and not self.ctx.twist:
            known = parent is not None and parent.process_class is JqmcCalculation and parent.is_finished_ok
            start = _counter(parent) if known else None
            if start is None:
                return self.exit_codes.ERROR_PARENT_COUNTER_UNKNOWN
        if "parent_folder" in self.ctx.inputs:
            # jqmc restores these keys from the checkpoint; leave them out from the first job on, as for the others.
            stripped = _without_frozen_keys(copy.deepcopy(parameters), job_type)
            if stripped != parameters:
                self.ctx.inputs.parameters = orm.Dict(stripped)

        self.ctx.counter = start
        self.ctx.target = start + parameters[job_type][self.ctx.step_key]
        self.ctx.restart_folder = self.ctx.inputs.get("parent_folder")
        self.ctx.num_interruptions = 0
        self.ctx.num_unexplained = 0
        self.report(f"{job_type}: counter {start} -> target {self.ctx.target}.")
        return None

    def _restart_from(self, folder: orm.RemoteData | None, counter: int) -> None:
        """Prepare the inputs of the next job: continue from ``folder`` up to the target."""
        parameters = copy.deepcopy(self.ctx.inputs.parameters.get_dict())
        if not self.ctx.twist:
            # A twist-averaged restart skips the completed twists and reruns the others in full.
            parameters[self.ctx.job_type][self.ctx.step_key] = self.ctx.target - counter
        if folder is not None:
            parameters = _without_frozen_keys(parameters, self.ctx.job_type)
            self.ctx.inputs.parent_folder = folder
            self.ctx.inputs.pop("hamiltonian", None)
        self.ctx.inputs.parameters = orm.Dict(parameters)

    @process_handler(priority=600)
    def handle_unfinished(self, node):
        """A job ended normally (e.g. stopped on ``max_time``) before the target was reached."""
        if not node.is_finished_ok:
            return None

        self.ctx.num_interruptions = 0
        self.ctx.num_unexplained = 0
        if self.ctx.twist:
            return None
        counter = _counter(node)
        if counter is None or counter <= self.ctx.counter:
            return ProcessHandlerReport(True, self.exit_codes.ERROR_NO_PROGRESS)
        if counter >= self.ctx.target:
            return None

        self.report(f"{node.process_label}<{node.pk}> stopped at {counter}/{self.ctx.target}: continuing.")
        self.ctx.counter = counter
        self.ctx.restart_folder = node.outputs.remote_folder
        self._restart_from(self.ctx.restart_folder, counter)
        return ProcessHandlerReport(True)

    @process_handler(
        priority=500,
        exit_codes=[
            JqmcCalculation.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE,
            JqmcCalculation.exit_codes.ERROR_SCHEDULER_OUT_OF_WALLTIME,
            JqmcCalculation.exit_codes.ERROR_SCHEDULER_NODE_FAILURE,
        ],
    )
    def handle_interrupted(self, node):
        """A job was cut off before jqmc ended: resubmit from the last good checkpoint."""
        self.ctx.num_interruptions += 1
        if self.ctx.num_interruptions > self._MAX_CONSECUTIVE_INTERRUPTIONS:
            return ProcessHandlerReport(True, self.exit_codes.ERROR_TOO_MANY_INTERRUPTED_JOBS)
        if node.exit_status == JqmcCalculation.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE.status:
            # Neither an exception nor a scheduler verdict: a crash (segfault, MPI abort) rather than a wall time.
            self.ctx.num_unexplained += 1
            if self.ctx.num_unexplained > self._MAX_CONSECUTIVE_UNEXPLAINED:
                return ProcessHandlerReport(True, self.exit_codes.ERROR_REPEATED_CRASH)
        else:
            self.ctx.num_unexplained = 0

        options = self.ctx.inputs.metadata.options
        wall = options.get("max_wallclock_seconds")
        if wall is not None:
            # More headroom between jqmc's max_time and the scheduler wall time.
            margin = options.get("max_time_margin") or default_max_time_margin(wall)
            # Never shrink the margin, and keep at least half of the wall time for jqmc.
            options["max_time_margin"] = max(margin, min(2 * margin, wall // 2))

        self.report(
            f"{node.process_label}<{node.pk}> was cut off ({node.exit_message}): "
            f"resubmitting from {'the start' if self.ctx.restart_folder is None else 'the last checkpoint'}."
        )
        self._restart_from(self.ctx.restart_folder, self.ctx.counter)
        return ProcessHandlerReport(True)

    def results(self):
        """The outputs of the last job; for a VMC run split into several jobs, the steps and the Hamiltonians of all."""
        jobs = [node for node in self.ctx.children if node.is_finished_ok and "output_parameters" in node.outputs]
        if self.ctx.job_type == "vmc" and not self.ctx.twist and len(jobs) > 1:
            self.out(
                "output_parameters",
                merge_vmc_jobs(**{f"job_{i:03d}": node.outputs.output_parameters for i, node in enumerate(jobs)}),
            )
            hamiltonians = {}
            for node in jobs:  # oldest first: a step written again by a later job keeps the later file
                if "output_hamiltonians" in node.outputs:
                    hamiltonians.update(dict(node.outputs.output_hamiltonians))
            for name, hamiltonian in sorted(hamiltonians.items()):
                self.out(f"output_hamiltonians.{name}", hamiltonian)
            last = self.ctx.children[self.ctx.iteration - 1]
            self.report(f"work chain completed after {self.ctx.iteration} iterations")
            for name, output in self.exposed_outputs(last, JqmcCalculation).items():
                if name not in ("output_parameters", "output_hamiltonians"):  # the other outputs of the last job
                    self.out(name, output)
            return None
        return super().results()
