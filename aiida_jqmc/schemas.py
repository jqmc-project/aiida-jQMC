"""JSON Schemas of ``output_parameters`` of JqmcCalculation and JqmcToolCalculation.

The schemas describe every field that ``jqmc.output_parser`` extracts and
the plugin stores, with its type, unit and meaning, so that a driver (for
example an AI agent) can read the results without reading the log. Every
object is closed (``additionalProperties: false``): the tests validate the
outputs of real logs against these schemas, so a field added in jqmc without
a description here fails them.

A key whose line was not printed is present with value ``None``; the values are
what jqmc printed, with no judgement of whether they are physically sensible.

Use :func:`output_parameters_schema` and :func:`tool_output_parameters_schema`.
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

__all__ = ("output_parameters_schema", "tool_commands_with_parameters", "tool_output_parameters_schema")

_DRAFT = "https://json-schema.org/draft/2020-12/schema"


def _nullable(schema: dict) -> dict:
    types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
    return {**schema, "type": [*types, "null"]}


def _num(description: str, unit: str | None = None, nullable: bool = True) -> dict:
    schema = {"type": "number", "description": description}
    if unit:
        schema["unit"] = unit
    return _nullable(schema) if nullable else schema


def _int(description: str, nullable: bool = True) -> dict:
    schema = {"type": "integer", "description": description}
    return _nullable(schema) if nullable else schema


def _str(description: str, nullable: bool = True) -> dict:
    schema = {"type": "string", "description": description}
    return _nullable(schema) if nullable else schema


def _bool(description: str, nullable: bool = True) -> dict:
    schema = {"type": "boolean", "description": description}
    return _nullable(schema) if nullable else schema


def _obj(description: str, properties: dict, nullable: bool = False, required: bool = True) -> dict:
    schema = {
        "type": "object",
        "description": description,
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = sorted(properties)
    return _nullable(schema) if nullable else schema


def _map(description: str, values: dict) -> dict:
    """An object with free-form keys, e.g. ``{zone: dtype}``."""
    return {"type": "object", "description": description, "additionalProperties": values}


def _list(description: str, items: dict, nullable: bool = False) -> dict:
    schema = {"type": "array", "description": description, "items": items}
    return _nullable(schema) if nullable else schema


def _value_error(name: str, description: str, unit: str) -> dict:
    return {name: _num(description, unit), f"{name}_error": _num(f"Statistical error of {name}.", unit)}


_SHAPE = {"type": "array", "items": {"type": "integer"}}
_MEMORY = _list(
    "Peak memory reported at named stages, in order.",
    {
        **_obj(
            "",
            {
                "stage": _str("Stage name.", False),
                "rss_peak_max_gb": _num("Peak RSS, maximum over ranks.", "GB", False),
                "jax_peak_max_gb": _num("Peak JAX device memory, maximum over ranks (GPU/TPU runs only).", "GB", False),
            },
        ),
        "required": ["rss_peak_max_gb", "stage"],
    },
)
_ANY_VALUE = {"description": "A printed input value: number, boolean, string, list or object."}

# -- Sections shared by every jqmc run -----------------------------------

_FORCES = _list(
    "Atomic forces, one row per atom, as printed in the final output; null when not computed.",
    _obj(
        "",
        {
            "label": _str("Atom label.", False),
            **{f"F{axis}": _num(f"Force component {axis}.", "Ha/bohr", False) for axis in "xyz"},
            **{f"F{axis}_err": _num(f"Statistical error of F{axis}.", "Ha/bohr", False) for axis in "xyz"},
        },
    ),
    nullable=True,
)


def _components(description: str, names: tuple, unit: str) -> dict:
    """``{name: value, name_err: error}`` for each component; partial tables leave components out."""
    properties = {}
    for name in names:
        properties[name] = _num(f"{name}.", unit, False)
        properties[f"{name}_err"] = _num(f"Statistical error of {name}.", unit, False)
    return _obj(description, properties, nullable=True, required=False)


_STRESS = _components(
    "Stress tensor as printed; null when not computed.",
    ("sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy", "sigma_xz", "sigma_yz"),
    "Ha/bohr^3",
)
_PRESSURE = _components(
    "Pressure P = -diag(sigma) as printed; null when not computed.", ("Px", "Py", "Pz", "P_hydro"), "Ha/bohr^3"
)

_ENVIRONMENT = _obj(
    "Software, hardware and precision of the run (header of the log).",
    {
        "jqmc_version": _str("jqmc version (printed once per process)."),
        "git_commit": _str("Git commit of the jqmc build."),
        "library_versions": _map("Versions of jax, jaxlib, numpy and python (verbosity high).", {"type": "string"}),
        "device_info": _str("JAX device summary (verbosity high)."),
        "process_count": _int("JAX process count (verbosity high)."),
        "platform": _str("OS family and architecture, e.g. 'Linux x86_64' (verbosity high)."),
        "xla_flags": _str("XLA_FLAGS of the run (verbosity high)."),
        "jax_environment_variables": _map(
            "JAX-related environment variables reported by jQMC (verbosity high).", {"type": "string"}
        ),
        "execution_backend": _str("Execution backend, e.g. 'XLA + Rust'."),
        "rust_threads": _int("Rust rayon threads per MPI rank."),
        "precision_mode": _str("Precision mode, e.g. 'Full Precision (FP64)'."),
        "precision_zones": _map("dtype of each precision zone (verbosity high).", {"type": "string"}),
        "started": _str("Start timestamp as printed."),
        "ended": _str("End timestamp as printed; null if the run did not end."),
        "memory": _MEMORY,
    },
)

_INPUT = _obj(
    "The input as jqmc applied it.",
    {
        "input_file": _str("Name of the TOML input file."),
        "explicit": _map(
            "Input echoed from the TOML file: {section: {key: value}}.",
            {"type": "object", "additionalProperties": _ANY_VALUE},
        ),
        "defaults": _map("Keys jqmc filled with its default value: {key: value}.", _ANY_VALUE),
        "restored": _map(
            "Keys a restart took from the checkpoint: {key: value}; null for non-scalar ones (the Hamiltonian).",
            _ANY_VALUE,
        ),
    },
)

_SYSTEM = _obj(
    "The system as summarised by jqmc from the Hamiltonian.",
    {
        "pbc": _bool("Periodic boundary conditions."),
        "atoms": _list(
            "Atoms in the order of the Hamiltonian.",
            _obj(
                "",
                {
                    "element": _str("Element symbol.", False),
                    "label": _str("Atom label.", False),
                    "Z": _num("Nuclear (or effective core) charge.", None, False),
                    "position": {**_SHAPE, "items": {"type": "number"}, "description": "Cartesian position.", "unit": "bohr"},
                },
            ),
        ),
        "ecp": _bool("Effective core potentials are used."),
        "num_electron_up": _int("Number of up-spin electrons."),
        "num_electron_dn": _int("Number of down-spin electrons."),
        "occupied_mo_rows": _nullable({**_SHAPE, "description": "Occupied MO rows (up, dn)."}),
        "time_reversal": _bool("Time-reversal symmetry of the determinant."),
        "num_mos": _int("Number of molecular orbitals."),
        "num_aos": _int("Number of atomic orbitals."),
        "num_primitive_aos": _int("Number of primitive Gaussian AOs."),
        "mo_coefficients_shape": _nullable({**_SHAPE, "description": "Shape of the MO coefficient matrix."}),
        "jastrow_2b_param": _num("Two-body Jastrow parameter."),
        "jastrow_2b_type": _str("Two-body Jastrow functional form, e.g. 'pade'."),
        "jastrow_3b_matrix_shape": _nullable({**_SHAPE, "description": "Shape of the three-body Jastrow matrix."}),
        "mo_orthonormality": _list(
            "MO orthonormality checks (verbosity high).",
            _obj(
                "",
                {
                    "data": _str("Orbital data checked, e.g. 'SD_data'.", False),
                    "spin": _str("'up' or 'dn'.", False),
                    "max_diag_deviation": _num("max |M_ii - 1|.", None, False),
                    "max_offdiag": _num("max_(i!=j) |M_ij| / sqrt(M_ii M_jj).", None, False),
                },
            ),
        ),
        "initial_walker_shapes": _map("Shapes of the initial walker arrays {up, dn} (verbosity high).", _SHAPE),
        "lattice_vectors": _nullable(
            {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
                "description": "Lattice vectors A, B, C (periodic systems).",
                "unit": "bohr",
            }
        ),
        "ewald": _map(
            "Ewald summation parameters as printed (eta, r_real_cut, g_max, N_real_images, N_g, ii_energy); "
            "empty for open systems.",
            {"type": "number"},
        ),
        "ao_images": _obj(
            "Periodic GTOs: the lattice images summed by the AO expansion.",
            {
                "num_image_cells": _int("Number of lattice image cells.", False),
                "max_image_index": _int("Largest |n_a|, |n_b|, |n_c| over the image indices."),
                "truncation_tolerance": _num("Truncation tolerance of the image set."),
            },
            nullable=True,
        ),
        "bspline": _obj(
            "B-spline basis summary; null for Gaussian orbitals.",
            {
                "grid": {**_SHAPE, "description": "Spline grid (N0, N1, N2)."},
                "order": _int("Spline degree.", False),
                "control_points_per_ao": _int("Control points per AO.", False),
                "total_coefficients": _int("Stored coefficients (num_ao * cp_per_ao).", False),
                "complex_path": _bool("Complex (k != 0) evaluation.", False),
            },
            nullable=True,
            required=False,
        ),
        "jastrow_1b_param": _num("One-body Jastrow parameter."),
        "jastrow_3b_symmetric": _bool("The j3 part of the three-body Jastrow matrix is symmetric."),
        "jastrow_dtn": _obj(
            "DTN Jastrow summary; null without DTN.",
            {
                "terms": _map("Active channels, e.g. {use_u, use_chi, use_f}.", {"type": "boolean"}),
                "cusp": _str("Cusp treatment as printed.", False),
                "C": _int("Cutoff polynomial smoothness order.", False),
                "orders": _map("Polynomial orders (N_u, N_chi, N_eN, N_ee).", {"type": "integer"}),
                "cutoffs": _map("Cutoffs L_u (number), L_chi and L_f (per species).", _ANY_VALUE),
                "cutoff_ratios": _map("Cutoff ratios u, chi, f.", _ANY_VALUE),
                "cutoff_ratio_policy": _str("Policy of the cutoff ratios."),
                "independent_coefficients": _map("Shapes of the independent coefficients (alpha, beta, gamma).", _SHAPE),
            },
            nullable=True,
        ),
        "tiled_orbitals": _list(
            "Tiled orbitals: the first summary per spin channel; empty when the orbitals are not tiled.",
            _obj(
                "",
                {
                    "spin": _str("'up' or 'down'.", False),
                    "supertwist": {"type": "array", "items": {"type": "number"}, "description": "Supertwist (3 floats)."},
                    "tiling": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "integer"}},
                        "description": "Tiling matrix (3 x 3 integers).",
                    },
                    "num_k_blocks": _int("Primitive k blocks.", False),
                    "num_occupied_orbitals": _int("Occupied orbitals.", False),
                    "local_basis_sizes": {"type": "array", "items": {"type": "integer"}, "description": "AOs per k block."},
                    "orbital_offsets": {"type": "array", "items": {"type": "integer"}, "description": "Offset per k block."},
                    "normalization": _str("Normalization convention of the tiled orbitals.", False),
                },
            ),
        ),
        "lambda_matrix": _obj(
            "Geminal (JAGP) lambda matrix; null for a Slater determinant.",
            {
                "shape": _nullable({**_SHAPE, "description": "Shape of lambda_matrix."}),
                "paired_shape": _nullable({**_SHAPE, "description": "Shape of the paired block."}),
                "unpaired_shape": _nullable({**_SHAPE, "description": "Shape of the unpaired block."}),
                "symmetric": _bool("lambda_matrix is symmetric."),
            },
            nullable=True,
        ),
    },
)

_TWIST_AVERAGE = _obj(
    "BZ-averaged final output of a twist-averaged run; null for a single-twist run.",
    {
        **_value_error("energy", "BZ-averaged total energy E_TA.", "Ha"),
        "s_twist": _num("Twist dispersion, reported apart from the MC error.", "Ha"),
        **_value_error("dE_mpc", "BZ-averaged MPC correction.", "Ha"),
        **_value_error("energy_mpc", "BZ-averaged MPC-corrected total energy.", "Ha"),
        "per_twist": _list(
            "Per-twist energies.",
            _obj(
                "",
                {
                    "twist": _int("Twist index.", False),
                    "k_point": _nullable(
                        {"type": "array", "items": {"type": "number"}, "description": "k point; null for Gamma."}
                    ),
                    **_value_error("energy", "Energy of this twist.", "Ha"),
                },
            ),
        ),
    },
    nullable=True,
)

_COMMON = {
    "job_type": {"type": "string", "description": "jqmc job type."},
    "program_ended": _bool("The log reached 'Program ends'.", False),
    "stopped_by_max_time": _bool("The run stopped itself on max_time.", False),
    "stopped_by_stop_flag": _bool("The run stopped on the stop flag of its external control file.", False),
    "incomplete_twists": _list(
        "Twist-averaged runs: the twists max_time stopped before their requested steps (a restart reruns them).",
        _obj(
            "One stopped twist.",
            {
                "twist": _int("Twist index.", False),
                "steps": _int("Measurements (MCMC) or branching steps (LRDMC) done.", False),
                "requested_steps": _int("num_mcmc_steps requested.", False),
            },
        ),
    ),
    "warnings": _list(
        "Input warnings such as 'num_mcmc_warmup_steps should be larger than 30', and the plugin's warning when the "
        "jqmc that parsed the log is not the jqmc that wrote it.",
        {"type": "string"},
    ),
    "parser_jqmc": _obj(
        "The jqmc (on the AiiDA host) whose output_parser read the log; `environment.git_commit` is the one that ran.",
        {"jqmc_version": _str("jqmc version.", False), "git_commit": _str("Git commit, or 'unknown'.", False)},
    ),
    "non_finite_values": _list(
        "Values printed as NaN or infinite, stored as null (AiiDA cannot store them), e.g. 'energy' or "
        "'steps.3.energy'. A NaN energy or variance gives exit 320.",
        {"type": "string"},
    ),
    "restart_checkpoint": _str("Checkpoint file written at the end of the run."),
    **_value_error("energy", "Final total energy.", "Ha"),
    **_value_error("variance", "Final variance of the local energy.", "Ha^2"),
    **_value_error("dE_mpc", "MPC finite-size correction.", "Ha"),
    **_value_error("energy_mpc", "MPC-corrected total energy.", "Ha"),
    "twist_average": _TWIST_AVERAGE,
    "twist_dispatch": _obj(
        "Twist-averaged dispatch banner; null for a single-twist run.",
        {
            "num_twists": _int("Number of twists in the input.", False),
            "hamiltonian_file": _str("Twist-averaged Hamiltonian file.", False),
            "bz_weight_sum": _num("Sum of the BZ weights of the twists.", None, False),
        },
        nullable=True,
    ),
    "mpc_setup": _obj(
        "Setup of the MPC tables; null without MPC.",
        {
            "broadcast_ranks": _int("Ranks the tables are broadcast to."),
            "cutoff": _num("MPC cutoff.", "Ha", False),
            "grid": {**_SHAPE, "description": "Real-space grid of the tables."},
            "n_max": _int("Number of grids of the extrapolation.", False),
            "note": _str("Extrapolation note as printed."),
            "steps": _list(
                "Timed setup steps, in order.",
                _obj("", {"step": _str("Step as printed.", False), "time_sec": _num("Time of the step.", "sec", False)}),
            ),
            "total_time_sec": _num("Total setup time.", "sec"),
        },
        nullable=True,
    ),
    "num_mpi_processes": _int("Number of MPI processes."),
    "num_walkers_per_process": _int("Walkers per MPI process."),
    "jax_backend": _str("JAX backend, e.g. 'cpu' or 'gpu'."),
    "jax_devices": _list("Global JAX devices.", {"type": "string"}, nullable=True),
    "environment": _ENVIRONMENT,
    "input_parameters": _INPUT,
    "system": _SYSTEM,
    "unparsed_numeric_lines": _list(
        "Lines that print a number the parser does not structure; normally empty.",
        {"type": "string"},
    ),
    "exception": _str("Last exception reported by jqmc (added by the plugin, only when one was reported)."),
}

_MCMC_TIMING = _map(
    "Time per MCMC step for each part (mcmc_update, e_L, de_L_dR_dr, dln_Psi_dR_dr, dln_Psi_dc, de_L_dc, mpi_barrier, misc).",
    {"type": "number", "unit": "msec"},
)

_SAMPLING = {
    "acceptance_ratio": _num("Metropolis acceptance ratio (fraction)."),
    "avg_walker_weight": _num("Average walker weight."),
    "total_time_sec": _num("Total elapsed time of the sampling.", "sec"),
    "precompilation_time_sec": _num("Compilation time.", "sec"),
    "net_time_sec": _num("Time without compilation.", "sec"),
}

_MCMC = {
    **_COMMON,
    "mcmc_counter": _int("Cumulative number of MCMC steps in the checkpoint (restarts included)."),
    "mcmc_counter_target": _int("Counter value this run was asked to reach."),
    **_SAMPLING,
    "timing_breakdown": _MCMC_TIMING,
    "atomic_forces": _FORCES,
    "stress_tensor": _STRESS,
    "pressure": _PRESSURE,
}

_VMC_STEP = _obj(
    "One VMC optimization step.",
    {
        "step": _int("Cumulative optimization step number.", False),
        **_value_error("energy", "Energy before the update of this step.", "Ha"),
        "max_force": _num(
            "Largest generalized force (twist-averaged run: over the shared and every twist's own coordinates).", "Ha/a.u."
        ),
        "max_force_error": _num("Statistical error of max_force.", "Ha/a.u."),
        "signal_to_noise_ratio": _num("Largest generalized-force signal-to-noise ratio (twist-averaged run: as max_force)."),
        **_SAMPLING,
        "timing_breakdown": _MCMC_TIMING,
        "num_variational_parameters": _int("Number of variational (storage) parameters."),
        "num_optimization_coordinates": _int(
            "Real optimization coordinates: the dimension of the SR/LM system (tied stored values once, complex ones twice)."
        ),
        "num_variational_parameters_all_twists": _int(
            "Twist-averaged run: variational (storage) parameters of every twist, the shared ones once."
        ),
        "num_optimization_coordinates_all_twists": _int(
            "Twist-averaged run: real optimization coordinates of every twist, the shared ones once plus each twist's own."
        ),
        "twist_local_max_signal_to_noise": _list(
            "Twist-averaged run: largest generalized-force signal-to-noise over each twist's own coordinates.",
            _obj("", {"twist": _int("Twist index.", False), "signal_to_noise": _num("Signal-to-noise ratio.", None, False)}),
        ),
        "dumped_hamiltonian": _str("Hamiltonian file written after this step."),
        "num_mcmc_steps": _int("Sampling steps of this step."),
        "num_mcmc_warmup_steps": _int("Warmup steps of this step."),
        "num_mcmc_bin_blocks": _int("Bin blocks of this step."),
        "optimization_batch": _obj(
            "Layout of the optimization batch (verbosity high).",
            {
                "route": _str("'host' or 'device'.", False),
                "local_samples": _int("Samples on this rank.", False),
                "sampler_ranks": _int("Sampler MPI ranks.", False),
                "real_coordinates": _int("Real optimization coordinates.", False),
                "storage_parameters": _int("Stored parameters.", False),
            },
            nullable=True,
        ),
        "derivative_sources": _list(
            "Derivative sources per parameter block (verbosity high).",
            _obj(
                "",
                {
                    "block": _str("Parameter block.", False),
                    "O_shape": {**_SHAPE, "description": "Shape of the log-derivative samples O."},
                    "D_shape": _nullable(
                        {**_SHAPE, "description": "Shape of the local-energy derivative samples D; null without D (SR)."}
                    ),
                    "policy": _str("Derivative policy.", False),
                    "tangent": _str("Tangent space.", False),
                },
            ),
        ),
        "asr": _obj(
            "Accelerated SR (aSR) gamma scaling (verbosity high); empty without aSR.",
            {
                "gamma_plus": _num("Root gamma+."),
                "gamma_minus": _num("Root gamma-."),
                "selected_root": _str("'+' or '-'."),
                "selected_gamma": _num("Selected gamma."),
                "selection_rule": _str("Rule that selected the root."),
                "notes": _list("Further aSR notes, e.g. the direction of the selected root.", {"type": "string"}),
                "final_gamma": _num("Gamma applied."),
                "scale": _num("Step scale decided by aSR."),
                "model_fallback": _bool("aSR fell back to the model gamma."),
                "fallbacks": _list(
                    "Each aSR fallback line, in order: its reason and the numbers it printed.",
                    _obj(
                        "",
                        {
                            "reason": _str(
                                "'root_not_finite_or_zero', 'root_exceeds_gamma_max', 'negative_centered_metric', "
                                "'degenerate_denominator' or 'invalid_model_denominator'.",
                                False,
                            ),
                            "root_gamma": _num("Gamma of the selected root.", None, False),
                            "gamma_max": _num("Largest |gamma| allowed.", None, False),
                            "S_2": _num("Centered metric contraction S_2.", None, False),
                            "H_1": _num("Energy-gradient contraction H_1.", None, False),
                            "gamma": _num("Gamma used instead.", None, False),
                        },
                    )
                    | {"required": ["gamma", "reason"]},
                ),
                "clipped_discriminant": _num("Negative discriminant of the gamma equation, set to zero."),
            },
            required=False,
        ),
        "diagnostics": _obj(
            "Optimizer diagnostics (verbosity high).",
            {
                "coordinates": _obj(
                    "Optimization coordinates.",
                    {name: _int(f"Number of {name} coordinates.", False) for name in ("real", "storage", "active", "frozen")}
                    | {"atomic_groups": _int("Atomic groups of frozen coordinates.", False)},
                ),
                "coordinates_all_twists": _obj(
                    "Twist-averaged run: optimization coordinates of every twist, the shared ones once.",
                    {name: _int(f"Number of {name} coordinates.", False) for name in ("real", "storage", "active", "frozen")}
                    | {"atomic_groups": _int("Atomic groups of frozen coordinates.", False)},
                ),
                "metric_diagonal": _obj(
                    "Range of the SR metric diagonal.",
                    {
                        name: _num(name.replace("_", " "), None, False)
                        for name in ("all_min", "all_max", "active_min", "active_max")
                    },
                ),
                "sr_solve": _obj(
                    "Linear solve of the SR equations.",
                    {
                        "regime": _str("'wide' or 'tall'.", False),
                        "method": _str("Solver, e.g. 'CG'.", False),
                        "iterations": _int("Iterations used.", False),
                        "residual": _num("Final residual.", None, False),
                        "tolerance": _num("Requested tolerance.", None, False),
                        "converged": _bool("The solve converged.", False),
                    },
                ),
                "sr_direction": _obj(
                    "SR direction q_SR.",
                    {"norm": _num("||q_SR||.", None, False), "max_abs": _num("max |q_SR|.", None, False)},
                ),
                "ln_psi_change_norm": _num("Norm of the linearized change of ln|Psi|."),
                "final_update": _obj(
                    "Final real-coordinate update delta_q.",
                    {"norm": _num("||delta_q||.", None, False), "max_abs": _num("max |delta_q|.", None, False)},
                ),
            },
            required=False,
        ),
        "top_coordinates": _list(
            "Coordinates with the largest generalized-force signal-to-noise (verbosity high).",
            _obj(
                "",
                {
                    "coordinate": _str(
                        "Coordinate, e.g. 'j3_matrix[q1]'; a twist's own one e.g. 'mo_coefficients_up[t1:q7]'.", False
                    ),
                    "force": _num("Generalized force.", "Ha/a.u.", False),
                    "force_error": _num("Statistical error of the force.", "Ha/a.u.", False),
                    "signal_to_noise": _num("Signal-to-noise ratio.", None, False),
                    "metric_diag": _num("SR metric diagonal of the coordinate.", None, False),
                },
            ),
        ),
        "blocks": _list(
            "Force and update statistics per parameter block (verbosity high).",
            _obj(
                "",
                {
                    "block": _str("Parameter block; a twist's own block is labelled e.g. 'mo_coefficients_up[t1]'.", False),
                    "force_norm": _num("||force|| of the block.", "Ha/a.u.", False),
                    "max_abs_force": _num("max |force| of the block.", "Ha/a.u.", False),
                    "max_signal_to_noise": _num("max S/N of the block.", None, False),
                    "metric_diag_min": _num("Smallest metric diagonal of the block.", None, False),
                    "metric_diag_max": _num("Largest metric diagonal of the block.", None, False),
                    "update": _obj(
                        "Update of the block.",
                        {
                            "delta_q_norm": _num("||delta_q||.", None, False),
                            "max_abs_delta_q": _num("max |delta_q|.", None, False),
                            "delta_p_norm": _num("||delta_p||.", None, False),
                            "max_abs_delta_p": _num("max |delta_p|.", None, False),
                        },
                        nullable=True,
                    ),
                    "s_stiefel_tangent": _obj(
                        "MO blocks only: norms of the update's S-Stiefel tangent delta_C S^(1/2).",
                        {
                            "frobenius_norm": _num("Frobenius norm.", None, False),
                            "spectral_norm": _num("Spectral norm.", None, False),
                        },
                    ),
                },
            )
            | {
                "required": [
                    "block",
                    "force_norm",
                    "max_abs_force",
                    "max_signal_to_noise",
                    "metric_diag_max",
                    "metric_diag_min",
                    "update",
                ]
            },
        ),
        "sr_solver_path": _obj(
            "Solver path of this step.",
            {
                "regime": _str("'wide' or 'tall'.", False),
                "method": _str("Solver.", False),
                "backend": _str("e.g. 'CPU/MPI'.", False),
            },
            nullable=True,
        ),
        "theta": _obj(
            "Statistics of the parameter update theta; empty if not printed.",
            {
                "size": _int("Number of entries."),
                "nonzero": _int("Number of non-zero entries."),
                "max_abs": _num("max |theta|."),
                "min_abs": _num("min |theta|."),
            },
            required=False,
        ),
        "lm": _obj(
            "Linear-method (LM) step; empty for SR and aSR.",
            {
                "conditioning": _obj(
                    "Conditioning of the LM coordinates (every verbosity).",
                    {
                        "active": _int("Active coordinates.", False),
                        "coordinates": _int("Coordinates before conditioning.", False),
                        "tiny_diagonal_removed": _int("Coordinates removed for a tiny metric diagonal.", False),
                        "correlation_removed": _int("Coordinates removed for correlation.", False),
                        "min_correlation_eigenvalue": _num("Smallest eigenvalue of the correlation matrix.", None, False),
                        "cutoff": _num("Conditioning cutoff.", None, False),
                    },
                ),
                "conditioning_input": _obj(
                    "Input of the conditioning (verbosity high).",
                    {
                        "coordinates": _int("Coordinates.", False),
                        "cutoff": _num("Conditioning cutoff.", None, False),
                        "diag_min": _num("Smallest metric diagonal.", None, False),
                        "diag_max": _num("Largest metric diagonal.", None, False),
                    },
                ),
                "selected_state": _obj(
                    "Selected LM eigenstate (verbosity high).",
                    {
                        "reference_weight": _num("Weight of the reference state.", None, False),
                        "q_norm": _num("||q||.", None, False),
                        "max_abs_q": _num("max |q|.", None, False),
                    },
                ),
                "subspace": _obj(
                    "LM subspace: the SR collective direction plus coordinates (verbosity high).",
                    {
                        "coordinates": _int("Coordinates.", False),
                        "atomic_groups": _int("Atomic groups.", False),
                    },
                ),
                "decision": _obj(
                    "Whether the LM step was accepted (verbosity high).",
                    {
                        "accepted": _bool("The LM candidate was accepted.", False),
                        "reference_weight": _num("Reference weight of the candidate.", None, False),
                        "fallback_scale": _num("Step scale used instead of a rejected candidate.", None, False),
                        "reason": _str("Why the candidate was rejected.", False),
                    },
                )
                | {"required": ["accepted"]},
            },
            required=False,
        ),
        "parameter_updates": _list(
            "Applied update per parameter block; a block left unchanged has zero norms.",
            _obj(
                "",
                {
                    "method": _str("Optimizer method of the step: 'SR', 'aSR' or 'LM'.", False),
                    "block": _str("Parameter block.", False),
                    "size": _int("Parameters in the block.", False),
                    "delta_p_norm": _num("||delta_p||.", None, False),
                    "max_abs_delta_p": _num("max |delta_p|.", None, False),
                    "max_abs_applied_update": _num("max |applied update|.", None, False),
                },
            ),
        ),
        "optimizer": _str("Optimizer that updated the parameters."),
        "memory": _MEMORY,
    },
)
_VMC = {
    **_COMMON,
    "steps": _list("Optimization steps of this run, in order (of every job, for a run split into several jobs).", _VMC_STEP),
    "num_jobs": _int(
        "JqmcRestartWorkChain only, for a run split into several jobs: the number of jobs merged. Then "
        "`dumped_hamiltonians` and `warnings` are of every job, `total_opt_time_sec` is their sum and "
        "`opt_timing_breakdown` the average over all steps; the other fields are those of the last job."
    ),
    "opt_step_counter": _int("Last optimization step printed (cumulative)."),
    "total_opt_steps": _int("Optimization step this run was asked to reach."),
    "dumped_hamiltonians": _list("Hamiltonian files written, in order.", {"type": "string"}),
    "total_opt_time_sec": _num("Total time of the optimization loop.", "sec"),
    "opt_timing_breakdown": _map(
        "Time per optimization step for each part (sampling, get_E, get_gF, optimizer, param_update, mpi_barrier, misc).",
        {"type": "number", "unit": "sec"},
    ),
    "optimizer": _obj(
        "Optimizer settings.",
        {
            "name": _str("Optimizer, e.g. 'sr'."),
            "mode": _str("Mode line, e.g. the aSR setting."),
            "hyperparameters": _map("Optimizer hyperparameters as printed.", _ANY_VALUE),
        },
    ),
}

_LRDMC = {
    **_COMMON,
    "gfmc_counter": _int("Cumulative number of branching steps in the checkpoint (restarts included)."),
    "gfmc_counter_target": _int("Counter value this run was asked to reach."),
    "survived_walkers_ratio": _num("Fraction of walkers surviving a branching."),
    "avg_num_projections": _num("Average number of projections between branchings (lrdmc-tau)."),
    "e_scf_history": _list(
        "Adaptive energy shift E_scf printed during the run (lrdmc-bra, verbosity high).",
        _obj("", _value_error("energy", "E_scf.", "Ha")),
    ),
    "total_time_sec": _num("Total time of the branching loop.", "sec"),
    "precompilation_time_sec": _num("Compilation time.", "sec"),
    "net_time_sec": _num("Time without compilation.", "sec"),
    "timing_breakdown": _map(
        "Time per branching step for each part (projection, observable, mpi_barrier, collection, reconfiguration, ...).",
        {"type": "number", "unit": "msec"},
    ),
    "atomic_forces": _FORCES,
    "stress_tensor": _STRESS,
    "pressure": _PRESSURE,
}

_JOB_TYPES = {"mcmc": _MCMC, "vmc": _VMC, "lrdmc-bra": _LRDMC, "lrdmc-tau": _LRDMC}


def output_parameters_schema(job_type: str) -> dict:
    """JSON Schema of ``output_parameters`` of a JqmcCalculation of *job_type*."""
    if job_type not in _JOB_TYPES:
        raise ValueError(f"Unknown job_type {job_type!r}; choose from {sorted(_JOB_TYPES)}.")
    properties = copy.deepcopy(_JOB_TYPES[job_type])
    properties["job_type"] = {**properties["job_type"], "const": job_type}
    return {
        "$schema": _DRAFT,
        "title": f"JqmcCalculation output_parameters ({job_type})",
        **_obj(f"Results of one jqmc run with job_type = {job_type!r}.", properties),
        "required": sorted(set(properties) - {"exception", "num_jobs"}),
    }


_TOOL_ENERGY = {
    **_value_error("energy", "Total energy.", "Ha"),
    **_value_error("dE_mpc", "MPC correction; null without an MPC estimate.", "Ha"),
    **_value_error("energy_mpc", "MPC-corrected total energy.", "Ha"),
    "num_mpi_ranks": _int("MPI ranks stored in the checkpoint."),
    "num_binned_samples": _int("Total number of binned samples."),
    "num_twists": _int("Twist checkpoints combined; null for a single-twist checkpoint."),
}
_TOOL_FORCE = {
    "atomic_forces": _FORCES,
    "num_mpi_ranks": _int("MPI ranks stored in the checkpoint."),
    "num_binned_samples": _int("Total number of binned samples."),
}
_TOOL_EXTRAPOLATION = {
    **_value_error("extrapolated_energy", "Energy extrapolated to a -> 0.", "Ha"),
    **_value_error("extrapolated_energy_mpc", "MPC-corrected energy extrapolated to a -> 0.", "Ha"),
    "polynomial_order": _int("Polynomial order in a^2."),
    "per_alat_results": _list(
        "Energy at each lattice spacing.",
        _obj(
            "",
            {
                "alat": _num("Lattice spacing a.", "bohr", False),
                "quantity": _str("'E' or 'E_MPC'.", False),
                **_value_error("energy", "Energy at this spacing.", "Ha"),
            },
        ),
    ),
}
_TOOL_RESULTS = {
    "energy": _TOOL_ENERGY,
    "force": _TOOL_FORCE,
    "extrapolation": _TOOL_EXTRAPOLATION,
    "info": _SYSTEM["properties"],
}


def tool_commands_with_parameters() -> list[str]:
    """jqmc-tool subcommands whose JqmcToolCalculation has ``output_parameters``."""
    from aiida_jqmc.calculations.tool import SUBCOMMANDS

    return [command for command, spec in SUBCOMMANDS.items() if spec["result"] in _TOOL_RESULTS]


def tool_output_parameters_schema(command: str) -> dict:
    """JSON Schema of ``output_parameters`` of a JqmcToolCalculation running *command*."""
    from aiida_jqmc.calculations.tool import SUBCOMMANDS

    result = SUBCOMMANDS[command]["result"]
    if result not in _TOOL_RESULTS:
        raise ValueError(f"{command!r} produces a file, not output_parameters.")
    properties = copy.deepcopy(_TOOL_RESULTS[result])
    properties["environment"] = _obj(
        "The jqmc-tool that ran (its stderr banner).",
        {"jqmc_version": _str("jqmc version."), "git_commit": _str("Git commit of the jqmc build.")},
    )
    properties["parser_jqmc"] = copy.deepcopy(_COMMON["parser_jqmc"])
    properties["warnings"] = _list(
        "The plugin's warning when the jqmc that parsed the stdout is not the jqmc-tool that ran.", {"type": "string"}
    )
    properties["non_finite_values"] = copy.deepcopy(_COMMON["non_finite_values"])
    properties["exception"] = _str("Error reported by jqmc-tool (only when one was reported).")
    return {
        "$schema": _DRAFT,
        "title": f"JqmcToolCalculation output_parameters ({command})",
        **_obj(f"Results of `jqmc-tool {command}`.", properties),
        "required": sorted(set(properties) - {"exception"}),
    }
