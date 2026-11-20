"""Run-length estimates from finished calculations, for the caller to use or not.

These helpers compute numbers; they decide nothing. A caller (an AI agent or a
person) that wants an energy error bar of a given size can ask how many more
steps the central-limit scaling ``error ~ 1/sqrt(N)`` predicts, and then
extend the calculation with JqmcRestartWorkChain if it agrees with the
estimate. The scaling assumes that the binning already captures the
autocorrelation; check that the error bar is stable against the binning
before trusting it.
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

import itertools
import math

from aiida import orm

__all__ = (
    "additional_steps_for_target_error",
    "crossing",
    "energy_slope",
    "error_of_error",
    "extrapolation_error_factor",
    "steps_to_target_error",
    "vmc_indicators",
    "vmc_sampling_steps_for_target_error",
)

_COUNTERS = {"mcmc": "mcmc_counter", "lrdmc-bra": "gfmc_counter", "lrdmc-tau": "gfmc_counter"}
_WARMUP = {"mcmc": "num_mcmc_warmup_steps", "lrdmc-bra": "num_gfmc_warmup_steps", "lrdmc-tau": "num_gfmc_warmup_steps"}


def additional_steps_for_target_error(
    accumulated_steps: int, warmup_steps: int, current_error: float, target_error: float
) -> int:
    """Additional measurements that bring ``current_error`` down to ``target_error``.

    Only the ``accumulated_steps - warmup_steps`` measurements after the warmup
    enter the estimate: ``N_needed = ceil(N_used * (current_error / target_error)**2)``.

    Returns:
        int: ``N_needed - N_used`` (0 when the target is already reached).
    """
    if target_error <= 0:
        raise ValueError(f"target_error must be positive, got {target_error}.")
    if not current_error > 0 or not math.isfinite(current_error):
        raise ValueError(f"current_error must be positive and finite, got {current_error}.")
    used = accumulated_steps - warmup_steps
    if used <= 0:
        raise ValueError(f"No measurement after the warmup ({accumulated_steps} steps, warmup {warmup_steps}).")
    if current_error <= target_error:
        return 0
    return math.ceil(used * (current_error / target_error) ** 2) - used


def steps_to_target_error(calculation: orm.ProcessNode, target_error: float) -> int:
    """Additional steps for a finished MCMC or LRDMC calculation (or its restart work chain).

    The counter and the energy error come from ``output_parameters``, the warmup
    from the input of the job (jqmc's default when not given). For a work chain
    pass the work chain node: its outputs are those of the last job.

    Returns:
        int: Additional ``num_mcmc_steps`` to request when extending the calculation.
    """
    from jqmc.jqmc_miscs import cli_parameters

    results = calculation.outputs.output_parameters.get_dict()
    job_type = results["job_type"]
    if job_type not in _COUNTERS:
        raise ValueError(f"Only mcmc and lrdmc calculations have an energy error bar to target, got {job_type!r}.")
    parameters = calculation.inputs.parameters if "parameters" in calculation.inputs else calculation.inputs.jqmc.parameters
    section = parameters.get_dict().get(job_type, {})
    warmup = section.get(_WARMUP[job_type], cli_parameters[job_type][_WARMUP[job_type]])
    return additional_steps_for_target_error(results[_COUNTERS[job_type]], warmup, results["energy_error"], target_error)


def vmc_sampling_steps_for_target_error(calculation: orm.ProcessNode, target_error: float) -> int:
    """``num_mcmc_steps`` per optimization step for a target energy error of each VMC step.

    Uses the last optimization step of a finished VMC calculation (e.g. a short
    pilot): its sampling steps, warmup and energy error.

    Returns:
        int: The total ``num_mcmc_steps`` (warmup included) to give each step.
    """
    results = calculation.outputs.output_parameters.get_dict()
    if results["job_type"] != "vmc" or not results["steps"]:
        raise ValueError("A finished VMC calculation with at least one optimization step is required.")
    step = results["steps"][-1]
    # jqmc prints the sampling settings of every optimization step.
    steps = step["num_mcmc_steps"]
    warmup = step["num_mcmc_warmup_steps"]
    return steps + additional_steps_for_target_error(steps, warmup, step["energy_error"], target_error)


def energy_slope(energies: list[float], errors: list[float]) -> tuple[float, float]:
    """Weighted least-squares slope of the energy per optimization step, and its standard error.

    Model ``E_k = a + b k`` with weights ``1 / sigma_k**2`` (non-positive errors take the
    median positive error), as jqmc_workflow's plateau criterion uses it.
    """
    import numpy as np

    energy, sigma = np.asarray(energies, dtype=float), np.asarray(errors, dtype=float)
    if energy.size < 2:
        raise ValueError("At least two optimization steps are needed for a slope.")
    positive = sigma[sigma > 0]
    sigma = np.where(sigma > 0, sigma, float(np.median(positive)) if positive.size else 1.0)
    w, k = 1.0 / sigma**2, np.arange(energy.size, dtype=float)
    s, sk, skk, se, ske = w.sum(), (w * k).sum(), (w * k * k).sum(), (w * energy).sum(), (w * k * energy).sum()
    delta = s * skk - sk**2
    return float((s * ske - sk * se) / delta), float(np.sqrt(s / delta))


def vmc_indicators(calculations: list[orm.ProcessNode], window: int = 10) -> dict:
    """Numbers a caller can use to judge a VMC optimization, over its last ``window`` steps.

    ``calculations`` are VMC calculations (or their restart work chains) in
    order, e.g. an optimization and its continuations; their steps are joined.

    Returns:
        dict: ``steps`` (step numbers in the window), ``signal_to_noise_mean`` and
        ``signal_to_noise_last`` (largest generalized-force S/N of each step),
        ``energy_slope`` and ``energy_slope_error`` (Ha per step), and
        ``energy_last`` / ``energy_last_error``.
    """
    steps = []
    for calculation in calculations:
        steps.extend(calculation.outputs.output_parameters["steps"])
    steps = [step for step in steps if step["energy"] is not None][-window:]
    if not steps:
        raise ValueError("No optimization step with an energy.")
    snr = [step["signal_to_noise_ratio"] for step in steps if step["signal_to_noise_ratio"] is not None]
    slope, slope_error = (
        energy_slope([s["energy"] for s in steps], [s["energy_error"] for s in steps]) if len(steps) >= 2 else (None, None)
    )
    return {
        "steps": [step["step"] for step in steps],
        "signal_to_noise_mean": sum(snr) / len(snr) if snr else None,
        "signal_to_noise_last": snr[-1] if snr else None,
        "energy_slope": slope,
        "energy_slope_error": slope_error,
        "energy_last": steps[-1]["energy"],
        "energy_last_error": steps[-1]["energy_error"],
    }


def error_of_error(error: float, num_blocks: int) -> float:
    """Statistical uncertainty of an error bar estimated from ``num_blocks`` blocks: ``sigma / sqrt(2 (b - 1))``.

    For independent, normally distributed block averages the relative uncertainty of the sample
    standard error is about ``1 / sqrt(2 (b - 1))``.
    """
    if num_blocks < 2:
        raise ValueError("At least two blocks are needed for an error bar.")
    return float(error) / math.sqrt(2.0 * (num_blocks - 1))


def crossing(values: list[float], observed: list[float], target: float) -> float | None:
    """Where the piecewise-linear interpolation of ``observed(values)`` reaches ``target``.

    The points are sorted by value. Returns the value of the first crossing between two neighbouring
    points (an exact hit returns that point's value), or None when ``target`` is not bracketed.
    """
    points = sorted(zip(values, observed, strict=True))
    for (x0, y0), (x1, y1) in itertools.pairwise(points):
        if y0 == target:
            return float(x0)
        if (y0 - target) * (y1 - target) < 0:
            return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
    if points and points[-1][1] == target:
        return float(points[-1][0])
    return None


def extrapolation_error_factor(lattice_spacings: list[float], polynomial_order: int) -> float:
    """Error bar of the ``a -> 0`` extrapolated value relative to the (equal) error bar of each point.

    The fit is ``E(a) = E_0 + E_2 a^2 + ... + E_{2p} a^{2p}`` (``p = polynomial_order``) by least squares
    with equal weights; the factor is ``sqrt([(X^T X)^{-1}]_{00})`` for the design matrix ``X``.
    """
    import numpy as np

    x = np.asarray(lattice_spacings, dtype=float) ** 2
    if x.size < polynomial_order + 1:
        raise ValueError(f"{polynomial_order + 1} lattice spacings are needed for order {polynomial_order}.")
    design = np.vstack([x**k for k in range(polynomial_order + 1)]).T
    return float(np.sqrt(np.linalg.inv(design.T @ design)[0, 0]))
