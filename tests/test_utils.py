"""Tests for aiida_jqmc.utils."""

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

import toml

from aiida_jqmc.utils import parameters_from_toml

CLI_TOML = """
[control]
job_type = "mcmc"
mcmc_seed = 34456
number_of_walkers = 4
max_time = 86400
restart = false
restart_chk = "restart.h5"
hamiltonian_h5 = "hamiltonian_data.h5"
verbosity = "low"
[mcmc]
num_mcmc_steps = 1000
Dt = 1.2
[precision]
mode = "mixed"
"""


def test_parameters_from_toml_file(tmp_path):
    """The keys the plugin sets are removed and returned; everything else is kept unchanged."""
    path = tmp_path / "mcmc.toml"
    path.write_text(CLI_TOML)
    parameters, removed = parameters_from_toml(path)
    assert removed == {
        "restart": False,
        "restart_chk": "restart.h5",
        "hamiltonian_h5": "hamiltonian_data.h5",
        "max_time": 86400,
    }
    expected = toml.loads(CLI_TOML)
    for key in removed:
        del expected["control"][key]
    assert parameters == expected


def test_parameters_from_dict_does_not_modify_it():
    source = toml.loads(CLI_TOML)
    parameters, _ = parameters_from_toml(source)
    assert "restart" in source["control"] and "restart" not in parameters["control"]
