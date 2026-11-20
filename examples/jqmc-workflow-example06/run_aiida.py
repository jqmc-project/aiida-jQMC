"""jqmc-workflow-example06 through aiida-jqmc: eight-twist silicon QE conversion, mixed-precision
twist-averaged VMC optimization of the shared J2 and the per-twist determinants, and the published
multi-twist Hamiltonian handed to a downstream twist-averaged MCMC."""

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

import sys
from pathlib import Path

import h5py
from jqmc._twist_average import is_twist_average_input

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _runner import EXAMPLES, file, jqmc, orm, shell, tool

N_TWIST = 8
VMC = {
    "control": {"job_type": "vmc", "number_of_walkers": 8, "verbosity": "high"},
    "vmc": {
        "num_opt_steps": 3,
        "num_mcmc_steps": 150,
        "num_mcmc_warmup_steps": 30,
        "num_mcmc_bin_blocks": 10,
        "Dt": 2.0,
        "wf_dump_freq": 10,
        "opt_J1_param": False,
        "opt_J2_param": True,
        "opt_J3_param": False,
        "opt_geminal": True,
        "optimizer_kwargs": {"method": "sr", "delta": 0.1, "epsilon": 0.001, "use_lm": True},
    },
    "precision": {"mode": "mixed"},
}
MCMC = {
    "control": {"job_type": "mcmc", "number_of_walkers": 8, "verbosity": "high"},
    "mcmc": {"num_mcmc_steps": 200, "num_mcmc_warmup_steps": 30, "num_mcmc_bin_blocks": 10, "Dt": 2.0},
    "precision": {"mode": "mixed"},
}

pw_in = (EXAMPLES / "jqmc-example09/01DFT/si_twist.in").read_text().replace("'../../jqmc-example07/01DFT'", "'./'")
_, pw = shell(
    "pw.x",
    "pw",
    "-in {pwin}",
    nodes={
        "pwin": orm.SinglefileData.from_string(pw_in, filename="si_twist.in"),
        "upf": file("jqmc-example07/01DFT/Si.ccECP.upf"),
    },
    kind="pw",
)
ham = tool(
    "qe --twist-average",
    "qe",
    {"--kind": "bspline", "--jastrow-2b-parameter": 1.0, "--verbosity": "high"},
    kind="convert",
    twist_average=orm.Bool(True),
    parent_folders={"pw": pw.outputs.remote_folder},
    save_path=orm.Str("out/silicon_twist.save"),
)
vmc = jqmc("twist vmc", VMC, hamiltonian=ham.outputs.output_hamiltonian)
published = vmc.outputs.output_hamiltonian
with published.as_path() as path, h5py.File(path) as handle:
    meta = handle["_meta"].attrs
    assert is_twist_average_input(str(path)) and int(meta["n_twist"]) == N_TWIST, dict(meta)
print(f"    published {published.filename}: {N_TWIST}-twist Hamiltonian", vmc.outputs.output_parameters["dumped_hamiltonians"])
mcmc = jqmc("twist mcmc", MCMC, hamiltonian=published)
ta = mcmc.outputs.output_parameters["twist_average"]
print(f"    twist-averaged MCMC energy: {ta['energy']} +- {ta['energy_error']} Ha (s_twist {ta['s_twist']})")
