"""jqmc-example07 through AiiDA: periodic silicon from QE, B-spline determinant, DTN Jastrow, mixed-precision MPC."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _runner import file, jqmc, orm, shell, tool, tutorial_input

EX = Path("jqmc-example07")

# 01DFT: pw.x (the save directory stays in the remote folder of the shell job)
_, pw = shell(
    "pw.x",
    "pw",
    "-in {pwin}",
    nodes={"pwin": file(EX / "01DFT/pw.in"), "upf": file(EX / "01DFT/Si.ccECP.upf")},
    kind="pw",
)

# 02vmc_JSD: QE -> B-spline + DTN Jastrow, from the remote folder of pw.x
converted = tool(
    "qe",
    "qe",
    {"--kind": "bspline", "--jastrow-dtn": "u_chi_f"},
    kind="convert",
    parent_folders={"pw": pw.outputs.remote_folder},
    save_path=orm.Str("silicon.save"),
)
vmc = jqmc("vmc", tutorial_input(EX / "02vmc_JSD/vmc.toml"), hamiltonian=converted.outputs.output_hamiltonian)
optimized = vmc.outputs.output_hamiltonian


def energy(name, run, options, family="mcmc"):
    node = tool(f"{name} energy", f"{family} compute-energy", options, parent_folders={"run": run.outputs.remote_folder})
    print("   ", {k: v for k, v in node.outputs.output_parameters.get_dict().items() if v is not None})


mcmc = jqmc("mcmc", tutorial_input(EX / "03mcmc_JSD/mcmc.toml"), hamiltonian=optimized)
energy("mcmc", mcmc, {"-b": 10, "-w": 30})
lrdmc = jqmc("lrdmc a=0.30", tutorial_input(EX / "04lrdmc_JSD/lrdmc_tau.toml"), hamiltonian=optimized)
energy("lrdmc", lrdmc, {"-b": 50, "-w": 50, "-c": 20}, family="lrdmc")
mpc = jqmc("mcmc mixed MPC", tutorial_input(EX / "05mcmc_JSD_mixed_mpc/mcmc.toml"), hamiltonian=optimized)
energy("mcmc mixed MPC", mpc, {"-b": 10, "-w": 30})
out = mpc.outputs.output_parameters.get_dict()
print("MPC run:", out["energy"], out["dE_mpc"], out["energy_mpc"], out["environment"]["precision_mode"])
