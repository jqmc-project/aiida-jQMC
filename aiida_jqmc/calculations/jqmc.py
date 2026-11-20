"""CalcJob for one run of the ``jqmc`` command-line program."""

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

import toml
from aiida import orm
from aiida.common import datastructures
from aiida.engine import CalcJob
from jqmc._twist_average import is_twist_average_input
from jqmc.jqmc_miscs import cli_parameters

from aiida_jqmc.tiling import is_tiled_hamiltonian

__all__ = ("JqmcCalculation",)

JOB_TYPES = ("mcmc", "vmc", "lrdmc-bra", "lrdmc-tau")

# ``control`` keys that wire files and wall time together. They are set by the
# plugin and must not appear in the ``parameters`` input.
RESERVED_CONTROL_KEYS = ("restart", "restart_chk", "hamiltonian_h5", "max_time")

# Per-rank log files would hide the rank-0 stdout that the parser reads.
ALLOWED_VERBOSITY = ("low", "high", "devel")


def is_twist_average(hamiltonian: orm.SinglefileData) -> bool:
    """Whether ``hamiltonian`` is a combined twist-averaged input (jqmc then runs one job per twist).

    jqmc's own predicate decides: a file with the combined ``shared``/``twists`` layout; an
    incomplete combined layout is an error rather than a single-twist input.
    """
    with hamiltonian.as_path() as path:
        return is_twist_average_input(str(path))


def checkpoint_name(calc: orm.CalcJobNode) -> str:
    """Checkpoint written by the JqmcCalculation ``calc``: a file, or a directory with one file per twist.

    It follows from the inputs: the Hamiltonian of a new run, or the parent of a restart.
    """
    while "hamiltonian" not in calc.inputs:
        parent = calc.inputs.parent_folder.creator if "parent_folder" in calc.inputs else None
        if parent is None or parent.process_class is not JqmcCalculation:
            return JqmcCalculation._CHECKPOINT_FILE
        calc = parent
    twist = is_twist_average(calc.inputs.hamiltonian)
    return JqmcCalculation._TWIST_CHECKPOINT_DIR if twist else JqmcCalculation._CHECKPOINT_FILE


def _tot_num_mpiprocs(computer: orm.Computer, resources: dict, withmpi: bool) -> int:
    """Total number of MPI processes for the given scheduler resources."""
    if not withmpi:
        return 1
    scheduler = computer.get_scheduler()
    resources = dict(resources or {})
    scheduler.preprocess_resources(resources, computer.get_default_mpiprocs_per_machine())
    return scheduler.create_job_resource(**resources).get_tot_num_mpiprocs()


def _num_walkers(calc: orm.CalcJobNode) -> int:
    """Number of walkers stored in the checkpoint written by ``calc``.

    A restart that omits ``number_of_walkers`` keeps the value of its checkpoint,
    so follow the restart chain up to the job that set it.
    """
    control = calc.inputs.parameters.get_dict()["control"]
    if "number_of_walkers" in control:
        return control["number_of_walkers"]
    parent = calc.inputs.parent_folder.creator if "parent_folder" in calc.inputs else None
    if parent is not None and parent.process_class is JqmcCalculation:
        return _num_walkers(parent)
    return cli_parameters["control"]["number_of_walkers"]


def validate_parameters(value: orm.Dict, _) -> str | None:
    """Check the structure of the jqmc input, not the physics."""
    params = value.get_dict()
    control = params.get("control")
    if not isinstance(control, dict):
        return "`parameters` must contain a `control` section."
    job_type = control.get("job_type")
    if job_type not in JOB_TYPES:
        return f"`control.job_type` must be one of {JOB_TYPES}, got {job_type!r}."
    if not isinstance(params.get(job_type), dict):
        return f"`parameters` must contain a `{job_type}` section for job_type = {job_type!r}."
    reserved = sorted(set(control) & set(RESERVED_CONTROL_KEYS))
    if reserved:
        return f"`control` keys {reserved} are set by the plugin and must not be given."
    verbosity = control.get("verbosity", "high")
    if verbosity not in ALLOWED_VERBOSITY:
        return f"`control.verbosity` must be one of {ALLOWED_VERBOSITY}, got {verbosity!r}."
    return None


def default_max_time_margin(wall: int) -> int:
    """10% of the wall time, clamped to [60, 1800] seconds."""
    return min(max(int(0.1 * wall), 60), 1800)


def _validate_restart(value: dict) -> str | None:
    """The checkpoint stores one group per MPI rank: a restart chain must keep its layout."""
    parent_folder = value["parent_folder"]
    code = value["code"]
    if parent_folder.computer.uuid != code.computer.uuid:
        return "`parent_folder` must be on the same computer as `code`."

    parent = parent_folder.creator
    if parent is None or parent.process_class is not JqmcCalculation:
        # Nothing to compare against, e.g. a RemoteData created by hand.
        return None

    parameters = value["parameters"].get_dict()
    parent_parameters = parent.inputs.parameters.get_dict()
    job_type = parameters["control"]["job_type"]
    parent_job_type = parent_parameters["control"]["job_type"]
    if parent_job_type != job_type:
        return f"Cannot restart job_type = {job_type!r} from a {parent_job_type!r} checkpoint."

    options = value.get("metadata", {}).get("options", {})
    nprocs = _tot_num_mpiprocs(code.computer, options.get("resources"), options.get("withmpi", True))
    parent_nprocs = _tot_num_mpiprocs(parent.computer, parent.get_option("resources"), parent.get_option("withmpi"))
    if nprocs != parent_nprocs:
        return (
            f"The number of MPI processes ({nprocs}) differs from the parent ({parent_nprocs}). "
            "The checkpoint is stored per MPI rank."
        )
    num_walkers = parameters["control"].get("number_of_walkers")
    if num_walkers is not None and num_walkers != _num_walkers(parent):
        return f"number_of_walkers ({num_walkers}) differs from the parent ({_num_walkers(parent)})."
    return None


def _initial_hamiltonian(value: dict) -> orm.SinglefileData | None:
    """Follow an AiiDA restart chain to the Hamiltonian that started it."""
    if "hamiltonian" in value:
        return value["hamiltonian"]
    parent = value["parent_folder"].creator if "parent_folder" in value else None
    while parent is not None and parent.process_class is JqmcCalculation:
        if "hamiltonian" in parent.inputs:
            return parent.inputs.hamiltonian
        parent = parent.inputs.parent_folder.creator if "parent_folder" in parent.inputs else None
    return None


def _validate_tiled_vmc(value: dict) -> str | None:
    parameters = value.get("parameters")
    if parameters is None:
        return None
    parameters = parameters.get_dict()
    if parameters["control"]["job_type"] != "vmc":
        return None
    vmc = parameters["vmc"]
    frozen = ("opt_geminal", "opt_lambda_basis_exp", "opt_lambda_basis_coeff")
    enabled = [name for name in frozen if vmc.get(name) is True]
    if not enabled:
        return None
    hamiltonian = _initial_hamiltonian(value)
    if hamiltonian is None:
        return None
    try:
        with hamiltonian.as_path() as path:
            tiled = is_tiled_hamiltonian(path)
    except (OSError, KeyError, TypeError, ValueError):
        # The main jQMC process remains responsible for diagnosing a malformed
        # Hamiltonian. Do not replace that diagnostic during input validation.
        return None
    if tiled:
        return (
            f"Tiled Slater orbitals are frozen; set {enabled} to false. "
            "J3 matrix, exponent and contraction-coefficient optimization remain independent and supported."
        )
    return None


def validate_inputs(value: dict, _) -> str | None:
    """Check how the run is started and wired, before anything is uploaded."""
    has_hamiltonian = "hamiltonian" in value
    has_parent = "parent_folder" in value
    if has_hamiltonian == has_parent:
        return "Exactly one of `hamiltonian` (new run) or `parent_folder` (restart) must be given."

    options = value.get("metadata", {}).get("options", {})
    wall = options.get("max_wallclock_seconds")
    margin = options.get("max_time_margin")
    if wall is None:
        return (
            "`metadata.options.max_wallclock_seconds` is required: jqmc's `max_time` is set from it, so that jqmc "
            "writes its checkpoint before the scheduler ends the job."
        )
    margin = default_max_time_margin(wall) if margin is None else margin
    if wall <= margin:
        return f"max_wallclock_seconds ({wall}) must be larger than max_time_margin ({margin})."

    if has_parent and "code" in value and "parameters" in value:
        restart_error = _validate_restart(value)
        if restart_error is not None:
            return restart_error
    return _validate_tiled_vmc(value)


class JqmcCalculation(CalcJob):
    """Run ``jqmc <input.toml>`` for one of the job types mcmc, vmc, lrdmc-bra or lrdmc-tau.

    The plugin only wires files and wall time. Every physical or statistical
    parameter is taken verbatim from the ``parameters`` input.

    * New run: give ``hamiltonian`` (``hamiltonian_data.h5``).
    * Restart: give ``parent_folder`` (the ``remote_folder`` of a previous
      JqmcCalculation). The checkpoint is symlinked, ``control.restart`` is set,
      and the step counters stored in the checkpoint keep accumulating, i.e.
      ``num_mcmc_steps`` / ``num_opt_steps`` are *additional* steps. The job type,
      the number of MPI processes and the number of walkers must not change,
      because the checkpoint is stored per MPI rank. Keys that jqmc restores
      from the checkpoint ("frozen on restart") may be omitted.
    * ``control.max_time`` is set to ``max_wallclock_seconds - max_time_margin``
      so that jqmc stops gracefully and writes its checkpoint before the
      scheduler kills the job; ``max_wallclock_seconds`` is therefore required.
    * ``control.verbosity`` defaults to ``"high"``, so that ``output_parameters``
      holds every diagnostic the parser knows (optimizer diagnostics, top
      coordinates, block statistics, environment); give ``"low"`` to override.
    """

    _INPUT_FILE = "aiida.toml"
    _OUTPUT_FILE = "aiida.out"
    _HAMILTONIAN_FILE = "hamiltonian_data.h5"
    _CHECKPOINT_FILE = "restart.h5"
    _TWIST_CHECKPOINT_DIR = "restart_twists"
    _OPT_HAMILTONIAN_GLOB = "hamiltonian_data_opt_step_*.h5"

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs and exit codes."""
        super().define(spec)

        spec.input("parameters", valid_type=orm.Dict, validator=validate_parameters, help="The jqmc TOML input as a dict.")
        spec.input(
            "hamiltonian",
            valid_type=orm.SinglefileData,
            required=False,
            help="`hamiltonian_data.h5` to start a new run from.",
        )
        spec.input(
            "parent_folder",
            valid_type=orm.RemoteData,
            required=False,
            help="Remote folder of a previous JqmcCalculation whose checkpoint is continued.",
        )
        spec.input(
            "metadata.options.max_time_margin",
            valid_type=int,
            required=False,
            help="Seconds reserved between jqmc's `max_time` and the scheduler wall time "
            "(default: 10% of the wall time, clamped to [60, 1800]).",
        )
        spec.input(
            "metadata.options.all_hamiltonians",
            valid_type=bool,
            default=False,
            help="VMC: also store every `hamiltonian_data_opt_step_*.h5` written (`output_hamiltonians`), "
            "so that an earlier step can be chosen; by default only the last one is stored.",
        )
        spec.input(
            "metadata.options.retrieve_checkpoint",
            valid_type=bool,
            default=False,
            help="Also retrieve the checkpoint into the repository. It stays in the remote folder anyway.",
        )
        spec.inputs.validator = validate_inputs

        spec.inputs["metadata"]["options"]["withmpi"].default = True
        spec.inputs["metadata"]["options"]["input_filename"].default = cls._INPUT_FILE
        spec.inputs["metadata"]["options"]["output_filename"].default = cls._OUTPUT_FILE
        spec.inputs["metadata"]["options"]["parser_name"].default = "jqmc.jqmc"

        spec.output("output_parameters", valid_type=orm.Dict, help="Results and diagnostics parsed from the jqmc stdout.")
        spec.output(
            "output_hamiltonian",
            valid_type=orm.SinglefileData,
            required=False,
            help="VMC only: the last `hamiltonian_data_opt_step_*.h5` written.",
        )
        spec.output_namespace(
            "output_hamiltonians",
            valid_type=orm.SinglefileData,
            dynamic=True,
            required=False,
            help="VMC with `all_hamiltonians`: every Hamiltonian written, as `step_<N>`.",
        )
        spec.default_output_node = "output_parameters"

        spec.exit_code(301, "ERROR_OUTPUT_STDOUT_MISSING", message="The jqmc stdout file was not retrieved.")
        spec.exit_code(
            302,
            "ERROR_OUTPUT_STDOUT_INCOMPLETE",
            message="The jqmc stdout is incomplete (the run was killed or crashed before it ended).",
        )
        spec.exit_code(
            305,
            "ERROR_NOT_STARTED",
            message="jqmc wrote nothing: it was not started (e.g. the MPI launch failed); see the scheduler stderr.",
        )
        spec.exit_code(303, "ERROR_INVALID_INPUT", message="jqmc rejected the input: {message}")
        spec.exit_code(304, "ERROR_JQMC_EXCEPTION", message="jqmc raised an exception: {message}")
        spec.exit_code(310, "ERROR_NO_CHECKPOINT", message="jqmc ended but did not report writing its checkpoint.")
        spec.exit_code(311, "ERROR_OUTPUT_HAMILTONIAN_MISSING", message="The VMC output Hamiltonian was not retrieved.")
        spec.exit_code(
            312,
            "ERROR_TWIST_STOPPED_EARLY",
            message="A twist-averaged run stopped on max_time before every twist was complete: restart it from "
            "its remote_folder (the incomplete twists are rerun in full) or rerun with a longer wall time.",
        )
        spec.exit_code(313, "ERROR_TWIST_RESULT_MISSING", message="A twist-averaged run ended without its BZ-averaged result.")
        spec.exit_code(320, "ERROR_NON_FINITE_RESULT", message="The final energy or variance is NaN or infinite.")

    def _max_time(self) -> int | None:
        wall = self.node.get_option("max_wallclock_seconds")
        if wall is None:
            return None
        margin = self.node.get_option("max_time_margin")
        return wall - (default_max_time_margin(wall) if margin is None else margin)

    def prepare_for_submission(self, folder):
        """Write the TOML input and link the Hamiltonian or the parent checkpoint."""
        parameters = copy.deepcopy(self.inputs.parameters.get_dict())
        control = parameters["control"]
        restart = "parent_folder" in self.inputs

        # A twist-averaged run keeps one checkpoint file per twist in a directory.
        checkpoint = checkpoint_name(self.node)
        twist = checkpoint == self._TWIST_CHECKPOINT_DIR

        control["restart"] = restart
        control["restart_chk"] = checkpoint
        control["hamiltonian_h5"] = self._HAMILTONIAN_FILE
        # The parser and its schemas cover every field jqmc prints at "high"; "low" leaves them empty.
        control.setdefault("verbosity", "high")
        max_time = self._max_time()
        if max_time is not None:
            control["max_time"] = max_time

        local_copy_list = []
        remote_symlink_list = []
        remote_copy_list = []
        if restart:
            parent = self.inputs.parent_folder
            source = (parent.computer.uuid, f"{parent.get_remote_path()}/{checkpoint}", checkpoint)
            if twist:
                # jqmc writes into the checkpoint directory: copy it, so the parent's stays as it was.
                remote_copy_list.append(source)
                # jqmc selects the twist-averaged route from the Hamiltonian file, also on restart. Copy the one that
                # started the chain: a link to the parent's (itself a link) would chain every job to every earlier
                # folder, so cleaning one of them would break all later jobs.
                initial = _initial_hamiltonian({"parent_folder": parent})
                if initial is not None:
                    local_copy_list.append((initial.uuid, initial.filename, self._HAMILTONIAN_FILE))
                else:
                    remote_symlink_list.append(
                        (
                            parent.computer.uuid,
                            f"{parent.get_remote_path()}/{self._HAMILTONIAN_FILE}",
                            self._HAMILTONIAN_FILE,
                        )
                    )
            else:
                # jqmc removes and rewrites `restart.h5`, which replaces the symlink and
                # leaves the parent's checkpoint untouched.
                remote_symlink_list.append(source)
        else:
            hamiltonian = self.inputs.hamiltonian
            local_copy_list.append((hamiltonian.uuid, hamiltonian.filename, self._HAMILTONIAN_FILE))

        with folder.open(self._INPUT_FILE, "w") as handle:
            toml.dump(parameters, handle)

        codeinfo = datastructures.CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [self._INPUT_FILE]
        codeinfo.stdout_name = self._OUTPUT_FILE
        codeinfo.withmpi = self.node.get_option("withmpi")

        retrieve_list = [self._OUTPUT_FILE]
        if self.node.get_option("retrieve_checkpoint"):
            retrieve_list.append(checkpoint)

        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = local_copy_list
        calcinfo.remote_symlink_list = remote_symlink_list
        calcinfo.remote_copy_list = remote_copy_list
        calcinfo.retrieve_list = retrieve_list
        if control["job_type"] == "vmc":
            calcinfo.retrieve_temporary_list = [self._OPT_HAMILTONIAN_GLOB]
        return calcinfo
