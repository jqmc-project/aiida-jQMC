"""Parser for :class:`aiida_jqmc.calculations.jqmc.JqmcCalculation`.

The log is parsed by ``jqmc.output_parser``. This module only maps its
result onto AiiDA outputs and exit codes. It reports what jqmc printed and does
not judge whether the values are physically sensible.
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
import pathlib

from aiida import orm
from aiida.engine import ExitCode
from aiida.parsers import Parser
from jqmc._build_info import build_info
from jqmc.output_parser import parse_jqmc_log, parse_last_exception

from aiida_jqmc.calculations.jqmc import JqmcCalculation, checkpoint_name

__all__ = ("JqmcParser", "add_plugin_fields", "finite_or_none", "parser_jqmc")


def _non_finite(results: dict) -> bool:
    values = [results.get(key) for key in ("energy", "energy_error", "variance", "variance_error")]
    return any(value is not None and not math.isfinite(value) for value in values)


def finite_or_none(value, path: str, replaced: list[str]):
    """``value`` with every NaN or infinite float replaced by None (AiiDA cannot store them); ``replaced`` collects
    the paths, e.g. ``energy`` or ``steps.3.energy``."""
    if isinstance(value, float) and not math.isfinite(value):
        replaced.append(path)
        return None
    if isinstance(value, dict):
        return {key: finite_or_none(item, f"{path}.{key}" if path else str(key), replaced) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(item, f"{path}.{i}", replaced) for i, item in enumerate(value)]
    return value


def parser_jqmc(results: dict) -> dict:
    """The jqmc whose output_parser read the log (on the AiiDA host), and a warning when it is not the jqmc that ran."""
    info = build_info()
    ran = results["environment"]["git_commit"]
    if ran and info.git_commit != "unknown" and ran != info.git_commit:
        results.setdefault("warnings", []).append(
            f"The log was written by jqmc {ran} but parsed by jqmc {info.git_commit} (the AiiDA host's): "
            "the parser and the restart policy come from the latter."
        )
    return {"jqmc_version": info.version, "git_commit": info.git_commit}


def add_plugin_fields(results: dict) -> dict:
    """What the plugin adds to the dict of ``jqmc.output_parser``: ``parser_jqmc`` (with its warning) and
    ``non_finite_values``, the NaN or infinite values replaced by None so that AiiDA can store the dict."""
    results["parser_jqmc"] = parser_jqmc(results)
    replaced: list[str] = []
    results = finite_or_none(results, "", replaced)
    results["non_finite_values"] = replaced
    return results


class JqmcParser(Parser):
    """Turn the jqmc stdout into ``output_parameters`` (and ``output_hamiltonian`` for VMC)."""

    def parse(self, retrieved_temporary_folder: str | None = None, **kwargs) -> ExitCode | None:
        """Parse the retrieved files."""
        output_filename = self.node.get_option("output_filename")
        try:
            stdout = self.retrieved.base.repository.get_object_content(output_filename)
        except FileNotFoundError:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING

        job_type = self.node.inputs.parameters["control"]["job_type"]
        results = parse_jqmc_log(stdout, job_type)

        try:
            stderr = self.retrieved.base.repository.get_object_content(
                self.node.get_option("scheduler_stderr") or "_scheduler-stderr.txt"
            )
        except FileNotFoundError:
            stderr = ""
        exception = parse_last_exception(stderr) or parse_last_exception(stdout)
        if exception:
            results["exception"] = exception
        non_finite = _non_finite(results)
        results = add_plugin_fields(results)
        self.out("output_parameters", orm.Dict(results))

        if not results["program_ended"]:
            if exception is None:
                # Keep the scheduler's verdict (e.g. ERROR_SCHEDULER_OUT_OF_WALLTIME), which
                # is set on the node before the parser runs; returning None keeps it.
                if self.node.exit_status is not None and self.node.exit_status > 0:
                    return None
                # Nothing at all on stdout: jqmc never ran, so resubmitting will not help.
                if not stdout.strip():
                    return self.exit_codes.ERROR_NOT_STARTED
                return self.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE
            # jqmc validates its input before any sampling starts.
            started = any(results.get(key) is not None for key in ("mcmc_counter", "gfmc_counter", "opt_step_counter"))
            if exception.startswith("ValueError") and not started:
                return self.exit_codes.ERROR_INVALID_INPUT.format(message=exception)
            return self.exit_codes.ERROR_JQMC_EXCEPTION.format(message=exception)

        if checkpoint_name(self.node) == JqmcCalculation._TWIST_CHECKPOINT_DIR:
            # A twist-averaged run writes one checkpoint per twist and does not log it.
            if results["stopped_by_max_time"]:
                return self.exit_codes.ERROR_TWIST_STOPPED_EARLY
            if job_type != "vmc" and results["twist_average"] is None:
                return self.exit_codes.ERROR_TWIST_RESULT_MISSING
        elif results["restart_checkpoint"] != JqmcCalculation._CHECKPOINT_FILE:
            return self.exit_codes.ERROR_NO_CHECKPOINT

        if job_type == "vmc":
            exit_code = self._attach_output_hamiltonian(results, retrieved_temporary_folder)
            if exit_code is not None:
                return exit_code

        if non_finite:
            return self.exit_codes.ERROR_NON_FINITE_RESULT

        # jqmc ended and wrote its checkpoint: that is a success even when the scheduler reported a failure
        # afterwards (e.g. the wall time or an out-of-memory event during MPI/JAX teardown).
        return ExitCode(0)

    def _attach_output_hamiltonian(self, results: dict, retrieved_temporary_folder: str | None) -> ExitCode | None:
        """Store the Hamiltonian of the last optimization step as ``output_hamiltonian``."""
        if not results["dumped_hamiltonians"]:
            return self.exit_codes.ERROR_OUTPUT_HAMILTONIAN_MISSING
        last = results["dumped_hamiltonians"][-1]
        if retrieved_temporary_folder is None:
            return self.exit_codes.ERROR_OUTPUT_HAMILTONIAN_MISSING
        folder = pathlib.Path(retrieved_temporary_folder)
        if not (folder / last).is_file():
            return self.exit_codes.ERROR_OUTPUT_HAMILTONIAN_MISSING
        self.out("output_hamiltonian", orm.SinglefileData(folder / last))
        if self.node.get_option("all_hamiltonians"):
            # A node has one creator link, so the last dump is stored again under its step.
            for name in results["dumped_hamiltonians"]:
                if (folder / name).is_file():
                    step = name.rsplit("_", 1)[1].removesuffix(".h5")
                    self.out(f"output_hamiltonians.step_{step}", orm.SinglefileData(folder / name))
        return None
