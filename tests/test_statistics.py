"""Tests for the run-length estimate."""

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

from aiida_jqmc.statistics import additional_steps_for_target_error


def test_halving_the_error_needs_four_times_the_measurements():
    # 1000 steps, 100 of them warmup: 900 used; error / 2 -> 3600 used -> 2700 more.
    assert additional_steps_for_target_error(1000, 100, 2.0e-4, 1.0e-4) == 2700


def test_target_already_reached():
    assert additional_steps_for_target_error(1000, 100, 1.0e-4, 2.0e-4) == 0


@pytest.mark.parametrize(
    ("args", "message"),
    [((1000, 100, 1e-4, 0.0), "target_error"), ((1000, 100, 0.0, 1e-4), "current_error"), ((50, 100, 1e-4, 1e-5), "warmup")],
)
def test_invalid(args, message):
    with pytest.raises(ValueError, match=message):
        additional_steps_for_target_error(*args)


def test_energy_slope_of_a_straight_line():
    from aiida_jqmc.statistics import energy_slope

    slope, error = energy_slope([-1.0, -1.1, -1.2, -1.3], [0.01, 0.01, 0.01, 0.01])
    assert slope == pytest.approx(-0.1)
    assert error > 0


def test_vmc_indicators_join_runs(generate_vmc):
    from aiida_jqmc.statistics import vmc_indicators

    first = generate_vmc([(1, -1.0, 0.01, 50.0), (2, -1.1, 0.01, 40.0)])
    second = generate_vmc([(3, -1.2, 0.01, 30.0), (4, -1.3, 0.01, 20.0)])
    out = vmc_indicators([first, second], window=3)
    assert out["steps"] == [2, 3, 4]
    assert out["signal_to_noise_mean"] == pytest.approx(30.0)
    assert out["signal_to_noise_last"] == 20.0
    assert out["energy_slope"] == pytest.approx(-0.1)


@pytest.fixture
def generate_vmc():
    from types import SimpleNamespace

    def factory(rows):
        steps = [{"step": s, "energy": e, "energy_error": de, "signal_to_noise_ratio": snr} for s, e, de, snr in rows]
        return SimpleNamespace(outputs=SimpleNamespace(output_parameters={"steps": steps}))

    return factory


@pytest.fixture
def finished(aiida_localhost):
    """A stored calculation with the given input parameters and output_parameters."""
    from aiida import orm
    from aiida.common.links import LinkType

    def factory(parameters: dict, results: dict):
        node = orm.CalcJobNode(computer=aiida_localhost, process_type="aiida.calculations:jqmc.jqmc")
        node.base.links.add_incoming(orm.Dict(parameters).store(), link_type=LinkType.INPUT_CALC, link_label="parameters")
        node.store()
        output = orm.Dict(results)
        output.base.links.add_incoming(node, link_type=LinkType.CREATE, link_label="output_parameters")
        output.store()
        return node

    return factory


def test_steps_to_target_error_reads_the_calculation(finished):
    from aiida_jqmc.statistics import steps_to_target_error

    node = finished(
        {"control": {"job_type": "mcmc"}, "mcmc": {"num_mcmc_warmup_steps": 100}},
        {"job_type": "mcmc", "mcmc_counter": 1000, "energy_error": 2.0e-4},
    )
    assert steps_to_target_error(node, 1.0e-4) == 2700
    lrdmc = finished(
        {"control": {"job_type": "lrdmc-tau"}, "lrdmc-tau": {"num_gfmc_warmup_steps": 50}},
        {"job_type": "lrdmc-tau", "gfmc_counter": 250, "energy_error": 3.0e-4},
    )
    assert steps_to_target_error(lrdmc, 1.0e-4) == 1600  # 200 used -> 1800


def test_vmc_sampling_steps_for_target_error_reads_the_last_step(finished):
    from aiida_jqmc.statistics import vmc_sampling_steps_for_target_error

    node = finished(
        {"control": {"job_type": "vmc"}, "vmc": {}},
        {"job_type": "vmc", "steps": [{"step": 1, "energy_error": 4e-3, "num_mcmc_steps": 130, "num_mcmc_warmup_steps": 30}]},
    )
    # 100 used samples at 4e-3 -> 1600 for 1e-3, plus the warmup
    assert vmc_sampling_steps_for_target_error(node, 1e-3) == 1630


def test_steps_to_target_error_rejects_vmc(finished):
    from aiida_jqmc.statistics import steps_to_target_error

    node = finished({"control": {"job_type": "vmc"}, "vmc": {}}, {"job_type": "vmc", "energy_error": None})
    with pytest.raises(ValueError, match="Only mcmc and lrdmc"):
        steps_to_target_error(node, 1e-3)


def test_error_of_error():
    from aiida_jqmc.statistics import error_of_error

    assert error_of_error(1.0, 3) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        error_of_error(1.0, 1)


def test_crossing():
    from aiida_jqmc.statistics import crossing

    assert crossing([1.2, 1.5, 1.8], [0.71, 0.65, 0.61], 0.65) == pytest.approx(1.5)
    assert crossing([1.8, 1.2, 1.5], [0.61, 0.71, 0.65], 0.68) == pytest.approx(1.35)  # unsorted input
    assert crossing([1.2, 1.5], [0.71, 0.65], 0.5) is None  # not bracketed


def test_extrapolation_error_factor():
    from aiida_jqmc.statistics import extrapolation_error_factor

    grid = [0.10, 0.15, 0.20, 0.25, 0.30]
    assert extrapolation_error_factor(grid, 1) == pytest.approx(0.834, abs=1e-3)
    assert extrapolation_error_factor(grid, 2) == pytest.approx(1.406, abs=1e-3)
    with pytest.raises(ValueError):
        extrapolation_error_factor([0.1, 0.2], 2)
