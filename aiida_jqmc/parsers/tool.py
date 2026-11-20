"""Parser for :class:`aiida_jqmc.calculations.tool.JqmcToolCalculation`.

The stdout is parsed by ``jqmc.output_parser``. As for jqmc itself, the
parser reports what jqmc-tool printed and does not judge the values.
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

from aiida import orm
from aiida.engine import ExitCode
from aiida.parsers import Parser
from jqmc.output_parser import (
    parse_environment,
    parse_last_exception,
    parse_lrdmc_extrapolation,
    parse_system_info,
    parse_tool_energy,
    parse_tool_force,
)

from aiida_jqmc.calculations.tool import SUBCOMMANDS, JqmcToolCalculation
from aiida_jqmc.parsers.jqmc import add_plugin_fields
from aiida_jqmc.tiling import inspect_tiled_hamiltonian

__all__ = ("JqmcToolParser", "add_tool_fields")


def add_tool_fields(results: dict, stderr: str) -> dict:
    """What the plugin adds to a jqmc-tool result: ``environment`` (the version and git commit jqmc-tool prints on
    stderr for every subcommand), ``warnings`` (the plugin's) and the fields of
    :func:`~aiida_jqmc.parsers.jqmc.add_plugin_fields`."""
    environment = parse_environment(stderr)
    results["environment"] = {key: environment.get(key) for key in ("jqmc_version", "git_commit")}
    results["warnings"] = []
    return add_plugin_fields(results)


# result -> (stdout parser, key that must be found for the run to count as done)
_PARSERS = {
    "energy": (parse_tool_energy, "energy"),
    "force": (parse_tool_force, "atomic_forces"),
    "extrapolation": (parse_lrdmc_extrapolation, "extrapolated_energy"),
    "info": (parse_system_info, "atoms"),
}
# result -> (output file, output port)
_FILES = {
    "hamiltonian": (JqmcToolCalculation._HAMILTONIAN_FILE, "output_hamiltonian"),
    "structure": (JqmcToolCalculation._STRUCTURE_FILE, "output_structure"),
}


def _tiling_requested(node) -> bool:
    return "tiling" in node.inputs


def _source_provenance(node) -> dict:
    source = {"command": node.inputs.command.value, "code_uuid": node.inputs.code.uuid}
    if "input_file" in node.inputs:
        source["input_uuid"] = node.inputs.input_file.uuid
    elif "input_folder" in node.inputs:
        source["input_uuid"] = node.inputs.input_folder.uuid
    elif "parent_folders" in node.inputs:
        source["input_uuids"] = {key: value.uuid for key, value in node.inputs.parent_folders.items()}
    return source


def _requested_selection(node) -> dict:
    options = node.inputs.options.get_dict() if "options" in node.inputs else {}
    tiling = node.inputs.tiling.get_list() if "tiling" in node.inputs else None
    supertwist = node.inputs.supertwist.get_list() if "supertwist" in node.inputs else None
    twist_average = "twist_average" in node.inputs and node.inputs.twist_average.value
    representation = options.get("--kind", options.get("-k"))
    return {
        "tiling": tiling,
        "supertwist": supertwist,
        "twist_average": twist_average,
        "representation": representation,
    }


class JqmcToolParser(Parser):
    """Turn the jqmc-tool stdout into ``output_parameters`` or ``output_hamiltonian``."""

    def parse(self, **kwargs) -> ExitCode | None:
        """Parse the retrieved files."""
        try:
            stdout = self.retrieved.base.repository.get_object_content(self.node.get_option("output_filename"))
        except FileNotFoundError:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING
        try:
            stderr = self.retrieved.base.repository.get_object_content(
                self.node.get_option("scheduler_stderr") or "_scheduler-stderr.txt"
            )
        except FileNotFoundError:
            stderr = ""

        result = SUBCOMMANDS[self.node.inputs.command.value]["result"]
        exception = parse_last_exception(stderr) or parse_last_exception(stdout)
        retrieved = self.retrieved.base.repository.list_object_names()

        graphs = [name for name in retrieved if name.startswith(f"{JqmcToolCalculation._GRAPH_STEM}.")]
        if graphs:
            with self.retrieved.base.repository.open(graphs[0], "rb") as handle:
                self.out("output_graph", orm.SinglefileData(handle, filename=graphs[0]))

        if result in _FILES:
            name, port = _FILES[result]
            if exception is not None:
                # jqmc-tool writes the file in place: after an exception it may be partial or unvalidated.
                return self._failure(stdout, exception, f"{name} was written but jqmc-tool reported an error")
            if name in retrieved:
                with self.retrieved.base.repository.open(name, "rb") as handle:
                    output = orm.SinglefileData(handle, filename=name)
                if result == "hamiltonian" and _tiling_requested(self.node):
                    try:
                        with output.as_path() as path:
                            metadata, structure = inspect_tiled_hamiltonian(
                                path,
                                stdout=f"{stdout}\n{stderr}",
                                source=_source_provenance(self.node),
                                requested=_requested_selection(self.node),
                            )
                    except Exception as exception:
                        # As after an exception: a Hamiltonian that failed the check is not an output.
                        return self.exit_codes.ERROR_TILING_OUTPUT_INVALID.format(message=str(exception))
                    self.out("conversion_metadata", orm.Dict(metadata))
                    self.out("physical_structure", structure)
                self.out(port, output)
                return None
            return self._failure(stdout, exception, f"{name} was not written")
        if result == "analysis":
            # The table repeats the steps of the VMC calculations' output_parameters; the graph is the result.
            if exception is not None or not stdout.strip():
                return self._failure(stdout, exception, "no analysis")
            return None

        parse, required = _PARSERS[result]
        results = parse(stdout)
        if exception:
            results["exception"] = exception
        results = add_tool_fields(results, stderr)
        self.out("output_parameters", orm.Dict(results))
        if results[required] in (None, []):
            if any(path == required or path.startswith(f"{required}.") for path in results["non_finite_values"]):
                return self._failure(stdout, exception, f"{required} is NaN or infinite")
            return self._failure(stdout, exception, f"no {required} in the stdout")
        return None

    def _failure(self, stdout: str, exception: str | None, missing: str) -> ExitCode:
        # Keep a verdict the scheduler set (e.g. the wall time), as JqmcParser does.
        if exception is None and self.node.exit_status is not None and self.node.exit_status > 0:
            return ExitCode(self.node.exit_status, self.node.exit_message)
        if exception is not None:
            if _tiling_requested(self.node):
                return self.exit_codes.ERROR_TILING_CONVERSION_FAILED.format(message=exception)
            return self.exit_codes.ERROR_TOOL_FAILED.format(message=exception)
        if not stdout.strip():
            return self.exit_codes.ERROR_NOT_STARTED
        return self.exit_codes.ERROR_OUTPUT_MISSING.format(message=missing)
