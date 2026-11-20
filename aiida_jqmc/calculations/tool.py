"""CalcJob for one run of the ``jqmc-tool`` command-line program."""

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
import shlex

from aiida import orm
from aiida.common import datastructures
from aiida.engine import CalcJob

from aiida_jqmc.calculations.jqmc import checkpoint_name

__all__ = ("SUBCOMMANDS", "JqmcToolCalculation")

# Subcommand -> what it reads and what it produces.
#   input: "file" (one `input_file`), "checkpoint" / "checkpoints" (the restart.h5 of one /
#          several `parent_folders`), "directory" (one `parent_folders` entry, linked whole,
#          or an `input_folder`, copied), or "logs" (the jqmc stdout of one or more `parent_folders`)
#   result: "hamiltonian" / "structure" (an output file), or the name of the stdout parser
SUBCOMMANDS = {
    "trexio convert-to": {"input": "file", "result": "hamiltonian"},
    "trexio show-info": {"input": "file", "result": "info"},
    "qe": {"input": "directory", "result": "hamiltonian"},
    "hamiltonian conv-wf": {"input": "file", "result": "hamiltonian"},
    "hamiltonian show-info": {"input": "file", "result": "info"},
    "hamiltonian to-xyz": {"input": "file", "result": "structure"},
    "vmc analyze-output": {"input": "logs", "result": "analysis"},
    "vmc average-wf": {"input": "files", "result": "hamiltonian"},
    "mcmc compute-energy": {"input": "checkpoint", "result": "energy"},
    "mcmc compute-force": {"input": "checkpoint", "result": "force"},
    "lrdmc compute-energy": {"input": "checkpoint", "result": "energy"},
    "lrdmc compute-force": {"input": "checkpoint", "result": "force"},
    "lrdmc extrapolate-energy": {"input": "checkpoints", "result": "extrapolation"},
}

# Options the plugin sets itself.
RESERVED_OPTIONS = ("-o", "--output")
# Options that open a window, per subcommand.
INTERACTIVE_OPTIONS = {
    "lrdmc extrapolate-energy": ("-g", "--plot-graph"),
    "vmc analyze-output": ("-p", "--plot_graph"),
}
# Options that save a graph: the plugin names the file and stores it as `output_graph`.
GRAPH_OPTIONS = ("-s", "--save-graph")

# Conversion selectors: given only through their AiiDA ports (tiling,
# supertwist, twist_average), never through ``options``.
_CONVERSION_COMMANDS = ("qe", "trexio convert-to")
_SELECTOR_OPTIONS = ("--tiling", "--supertwist", "--twist-average")
_TILED_QE_KINDS = ("pw", "bspline", "pgto", "pgto+pw")


def _tiling_determinant(entries: list[int]) -> int:
    """Return the exact determinant of a validated 3- or 9-entry matrix."""
    if len(entries) == 3:
        return entries[0] * entries[1] * entries[2]
    a, b, c, d, e, f, g, h, i = entries
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def validate_tiling(value: orm.List, _) -> str | None:
    """Validate only the CLI shape and non-singularity; jQMC owns tiling geometry."""
    entries = value.get_list()
    if len(entries) not in (3, 9):
        return "`tiling` must contain 3 diagonal entries or 9 row-major matrix entries."
    if any(isinstance(item, bool) or not isinstance(item, int) for item in entries):
        return "Every `tiling` entry must be an integer."
    if _tiling_determinant(entries) == 0:
        return "`tiling` must be non-singular."
    return None


def validate_supertwist(value: orm.List, _) -> str | None:
    """Validate the fractional supercell coordinate passed to jQMC."""
    entries = value.get_list()
    if len(entries) != 3:
        return "`supertwist` must contain three fractional supercell coordinates."
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in entries):
        return "Every `supertwist` coordinate must be finite."
    return None


def _option_value(options: dict, names: tuple[str, ...], default=None):
    values = [options[name] for name in names if name in options]
    return values[-1] if values else default


def validate_options(value: orm.Dict, _) -> str | None:
    """Option names are given verbatim, e.g. ``{"-j2": 1.0, "--num_mcmc_warmup_steps": 100}``.

    A list repeats the option, e.g. ``{"-u": ["H=H.upf", "O=O.upf"]}``.
    """
    for key, option in value.get_dict().items():
        if not key.startswith("-"):
            return f"Option {key!r} must be written as on the command line, e.g. '-j2' or '--output'."
        if key in RESERVED_OPTIONS:
            return f"Option {key!r} is set by the plugin and must not be given."
        values = option if isinstance(option, list) else [option]
        if not values or any(item is None or isinstance(item, (dict, list)) for item in values):
            return f"Option {key!r}: give a number, a string, true for a flag, or a list to repeat it."
    return None


def validate_inputs(value: dict, _) -> str | None:
    """The inputs must match what the subcommand reads."""
    command = value["command"].value
    if command not in SUBCOMMANDS:
        return f"`command` must be one of {sorted(SUBCOMMANDS)}, got {command!r}."
    kind = SUBCOMMANDS[command]["input"]
    options = value["options"].get_dict() if "options" in value else {}
    tiling = value.get("tiling")
    supertwist = value.get("supertwist")
    twist_average = value.get("twist_average")

    selectors = {"tiling": tiling, "supertwist": supertwist, "twist_average": twist_average}
    if any(item is not None for item in selectors.values()) and command not in _CONVERSION_COMMANDS:
        return f"`tiling`, `supertwist` and `twist_average` are only valid for {_CONVERSION_COMMANDS}."
    if given := [option for option in _SELECTOR_OPTIONS if option in options]:
        return f"Give {given} through the `tiling`, `supertwist` and `twist_average` inputs, not `options`."
    if supertwist is not None and tiling is None:
        return "`supertwist` requires `tiling`."
    if supertwist is not None and twist_average is not None and twist_average.value:
        return "`supertwist` and `twist_average` are mutually exclusive."
    if tiling is not None:
        j3 = _option_value(options, ("-j3", "--jastrow-3b-basis-set-type"))
        if j3 == "mo":
            return "Tiled conversion does not support `-j3 mo`; use an independent physical-supercell AO J3 basis."
        if command == "trexio convert-to" and _option_value(options, ("--ao-conv-to",)) is not None:
            return "Tiled TREXIO conversion preserves the source angular form; omit `--ao-conv-to`."
        if command == "qe":
            representation = _option_value(options, ("-k", "--kind"), "pgto+pw")
            if representation not in _TILED_QE_KINDS:
                return f"Tiled QE conversion supports {_TILED_QE_KINDS}, got {representation!r}."
    for key in INTERACTIVE_OPTIONS.get(command, ()):
        if key in options:
            return f"Option {key!r} opens a window; use -s/--save-graph to store the graph instead."
    folders = value.get("parent_folders", {})
    if "input_folder" in value and kind != "directory":
        return f"`input_folder` is only read by {sorted(c for c, s in SUBCOMMANDS.items() if s['input'] == 'directory')}."
    files = value.get("input_files", {})
    if files and kind != "files":
        return f"`input_files` are only read by {sorted(c for c, s in SUBCOMMANDS.items() if s['input'] == 'files')}."
    if kind == "files":
        if not files or folders or "input_file" in value:
            return f"{command!r} reads one or more `input_files` and no `input_file` or `parent_folders`."
        return None
    if kind == "file":
        if "input_file" not in value or folders:
            return f"{command!r} reads one `input_file` and no `parent_folders`."
    elif kind == "directory" and "input_folder" in value:
        if folders or "input_file" in value:
            return f"{command!r} reads either one of `parent_folders` or an `input_folder`, not both."
    else:
        if "input_file" in value or not folders:
            return f"{command!r} reads `parent_folders` and no `input_file`."
        if kind in ("checkpoint", "directory") and len(folders) != 1:
            return f"{command!r} reads exactly one of `parent_folders`, got {len(folders)}."
        if "save_path" in value and kind != "directory":
            return f"`save_path` is only read by {sorted(c for c, s in SUBCOMMANDS.items() if s['input'] == 'directory')}."
        code = value.get("code")
        if code is not None and any(folder.computer.uuid != code.computer.uuid for folder in folders.values()):
            return "`parent_folders` must be on the same computer as `code`."
    return None


def _command_line_options(options: dict) -> list[str]:
    arguments = []
    for key, option in options.items():
        for item in option if isinstance(option, list) else [option]:
            if item is True:
                arguments.append(key)
            elif isinstance(item, float) and item.is_integer():
                # jqmc-tool's integer options reject '10.0'; its float options read '10'.
                arguments.extend([key, str(int(item))])
            elif item is not False:
                arguments.extend([key, str(item)])
    return arguments


class JqmcToolCalculation(CalcJob):
    """Run one ``jqmc-tool`` subcommand: a conversion or the post-processing of checkpoints.

    ``command`` selects the subcommand (see ``SUBCOMMANDS``) and ``options``
    holds its options verbatim. The plugin only wires the files:

    * conversions (``trexio convert-to``, ``hamiltonian conv-wf``) read
      ``input_file`` and produce ``output_hamiltonian``;
    * ``qe`` reads a Quantum ESPRESSO output directory: the one entry of
      ``parent_folders`` (e.g. the ``remote_folder`` of a PwCalculation) is
      linked whole (or ``input_folder``, a FolderData, is copied), and
      ``save_path`` points inside it at ``prefix.save`` or
      its parent (``out`` for aiida-quantumespresso); it produces
      ``output_hamiltonian``;
    * post-processing (``compute-energy``, ``compute-force``,
      ``extrapolate-energy``) reads the checkpoint ``restart.h5`` of each
      ``parent_folders`` entry (a JqmcCalculation ``remote_folder``) and reports
      the parsed stdout in ``output_parameters``.
    """

    _OUTPUT_FILE = "aiida.out"
    _HAMILTONIAN_FILE = "hamiltonian_data.h5"
    _CHECKPOINT_FILE = "restart.h5"
    _STRUCTURE_FILE = "structure.xyz"
    _GRAPH_STEM = "graph"

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs and exit codes."""
        super().define(spec)

        spec.input("command", valid_type=orm.Str, help=f"The jqmc-tool subcommand, one of {sorted(SUBCOMMANDS)}.")
        spec.input(
            "options",
            valid_type=orm.Dict,
            required=False,
            validator=validate_options,
            help="Options as on the command line, e.g. {'-j2': 1.0, '-j3': 'mo'}; true for a flag.",
        )
        spec.input("input_file", valid_type=orm.SinglefileData, required=False, help="File read by a conversion.")
        spec.input_namespace(
            "input_files",
            valid_type=orm.SinglefileData,
            dynamic=True,
            required=False,
            help="``vmc average-wf``: the Hamiltonian files of the optimization steps, passed in key order.",
        )
        spec.input_namespace(
            "parent_folders",
            valid_type=orm.RemoteData,
            dynamic=True,
            required=False,
            help="Remote folders: of JqmcCalculations whose checkpoints are post-processed, "
            "or (``qe``) of the Quantum ESPRESSO run.",
        )
        spec.input(
            "input_folder",
            valid_type=orm.FolderData,
            required=False,
            help="``qe`` only, instead of `parent_folders`: a Quantum ESPRESSO output directory stored in AiiDA.",
        )
        spec.input(
            "save_path",
            valid_type=orm.Str,
            required=False,
            help="``qe`` only: path of `prefix.save`, or of its parent, inside the parent folder (default: the folder).",
        )
        spec.input(
            "tiling",
            valid_type=orm.List,
            required=False,
            validator=validate_tiling,
            help="Conversion only: three diagonal integers or nine row-major integers for the non-singular tiling matrix.",
        )
        spec.input(
            "supertwist",
            valid_type=orm.List,
            required=False,
            validator=validate_supertwist,
            help="Tiled conversion only: three finite fractional coordinates selecting one physical-supercell sector.",
        )
        spec.input(
            "twist_average",
            valid_type=orm.Bool,
            required=False,
            help="Conversion only: retain every complete sector and its converter-defined weight.",
        )
        spec.inputs.validator = validate_inputs

        spec.inputs["metadata"]["options"]["withmpi"].default = False
        spec.inputs["metadata"]["options"]["output_filename"].default = cls._OUTPUT_FILE
        spec.inputs["metadata"]["options"]["parser_name"].default = "jqmc.tool"
        spec.inputs["metadata"]["options"]["resources"].default = {"num_machines": 1, "num_mpiprocs_per_machine": 1}

        spec.output("output_parameters", valid_type=orm.Dict, required=False, help="Results parsed from the stdout.")
        spec.output(
            "output_hamiltonian", valid_type=orm.SinglefileData, required=False, help="Conversions: the Hamiltonian written."
        )
        spec.output(
            "conversion_metadata",
            valid_type=orm.Dict,
            required=False,
            help="Tiled conversions: matrix, actual sectors and weights, physical counts and converter build information.",
        )
        spec.output(
            "physical_structure",
            valid_type=orm.StructureData,
            required=False,
            help="Tiled conversions: the physical supercell structure deserialized from the converted Hamiltonian.",
        )
        spec.output("output_structure", valid_type=orm.SinglefileData, required=False, help="`to-xyz`: the xyz file.")
        spec.output("output_graph", valid_type=orm.SinglefileData, required=False, help="The graph saved with -s/--save-graph.")

        spec.exit_code(301, "ERROR_OUTPUT_STDOUT_MISSING", message="The jqmc-tool stdout file was not retrieved.")
        spec.exit_code(303, "ERROR_TOOL_FAILED", message="jqmc-tool failed: {message}")
        spec.exit_code(305, "ERROR_NOT_STARTED", message="jqmc-tool wrote nothing; see the scheduler stderr.")
        spec.exit_code(312, "ERROR_OUTPUT_MISSING", message="jqmc-tool ended without its result: {message}")
        spec.exit_code(313, "ERROR_TILING_CONVERSION_FAILED", message="jQMC rejected the tiled conversion: {message}")
        spec.exit_code(
            314,
            "ERROR_TILING_OUTPUT_INVALID",
            message="The tiled Hamiltonian could not be validated by jQMC: {message}",
        )

    def prepare_for_submission(self, folder):
        """Link the input file or the checkpoints and build the command line."""
        command = self.inputs.command.value
        options = self.inputs.options.get_dict() if "options" in self.inputs else {}

        local_copy_list = []
        remote_symlink_list = []
        positional = []
        retrieve_list = [self._OUTPUT_FILE]
        if SUBCOMMANDS[command]["input"] == "file":
            node = self.inputs.input_file
            local_copy_list.append((node.uuid, node.filename, node.filename))
            positional.append(node.filename)
        elif SUBCOMMANDS[command]["input"] == "files":
            # One subdirectory per key keeps the original file names (jqmc-tool records them as the sources).
            for key in sorted(self.inputs.input_files):
                node = self.inputs.input_files[key]
                folder.get_subfolder(key, create=True)
                local_copy_list.append((node.uuid, node.filename, f"{key}/{node.filename}"))
                positional.append(f"{key}/{node.filename}")
        elif SUBCOMMANDS[command]["input"] == "directory":
            if "input_folder" in self.inputs:
                key = "input_folder"
                local_copy_list.append((self.inputs.input_folder.uuid, "", key))
            else:
                (key,) = self.inputs.parent_folders
                parent = self.inputs.parent_folders[key]
                remote_symlink_list.append((parent.computer.uuid, parent.get_remote_path(), key))
            save_path = self.inputs.save_path.value.strip("/") if "save_path" in self.inputs else ""
            positional.append(f"{key}/{save_path}" if save_path else key)
        elif SUBCOMMANDS[command]["input"] == "logs":
            # The jqmc stdout of each parent, in key order (e.g. a VMC run and its continuations).
            for key in sorted(self.inputs.parent_folders):
                parent = self.inputs.parent_folders[key]
                creator = parent.creator
                name = creator.get_option("output_filename") if creator is not None else "aiida.out"
                folder.get_subfolder(key, create=True)
                remote_symlink_list.append((parent.computer.uuid, f"{parent.get_remote_path()}/{name}", f"{key}/{name}"))
                positional.append(f"{key}/{name}")
        else:
            # One subdirectory per parent, named by its key, holding the parent's checkpoint.
            for key in sorted(self.inputs.parent_folders):
                parent = self.inputs.parent_folders[key]
                # A twist-averaged run leaves a directory of per-twist checkpoints instead.
                creator = parent.creator
                name = checkpoint_name(creator) if creator is not None else self._CHECKPOINT_FILE
                path = f"{key}/{name}"
                folder.get_subfolder(key, create=True)
                remote_symlink_list.append((parent.computer.uuid, f"{parent.get_remote_path()}/{name}", path))
                positional.append(path)
        if SUBCOMMANDS[command]["result"] == "hamiltonian":
            options = {**options, "-o": self._HAMILTONIAN_FILE}
            retrieve_list.append(self._HAMILTONIAN_FILE)
        if SUBCOMMANDS[command]["result"] == "structure":
            options = {**options, "-o": self._STRUCTURE_FILE}
            retrieve_list.append(self._STRUCTURE_FILE)
        for key in GRAPH_OPTIONS:
            if key in options:
                # Keep the requested format (the suffix), with a name the parser knows.
                suffix = pathlib.PurePosixPath(str(options.pop(key))).suffix or ".png"
                options[key] = f"{self._GRAPH_STEM}{suffix}"
                retrieve_list.append(options[key])

        codeinfo = datastructures.CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        selection = []
        if "tiling" in self.inputs:
            selection.extend(["--tiling", " ".join(str(item) for item in self.inputs.tiling.get_list())])
        if "supertwist" in self.inputs:
            selection.extend(["--supertwist", " ".join(str(item) for item in self.inputs.supertwist.get_list())])
        if "twist_average" in self.inputs and self.inputs.twist_average.value:
            selection.append("--twist-average")
        codeinfo.cmdline_params = [*shlex.split(command), *positional, *selection, *_command_line_options(options)]
        codeinfo.stdout_name = self._OUTPUT_FILE
        codeinfo.withmpi = self.node.get_option("withmpi")

        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = local_copy_list
        calcinfo.remote_symlink_list = remote_symlink_list
        calcinfo.retrieve_list = retrieve_list
        return calcinfo
