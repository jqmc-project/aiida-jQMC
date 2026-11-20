"""jqmc-example08 through AiiDA: periodic all-electron hydrogen from QE (pGTO+PW, DTN cusp),
VMC optimization, MCMC and LRDMC energies and forces."""

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
from _runner import file, jqmc, orm, shell, show, tool, tutorial_input

EX = Path("jqmc-example08")

_, pw = shell(
    "pw.x",
    "pw",
    "-in {pwin}",
    nodes={"pwin": file(EX / "01DFT/h_scf.in"), "upf": file(EX / "01DFT/H.coulomb-ae.UPF")},
    kind="pw",
)
converted = tool(
    "qe pgto+pw",
    "qe",
    {"--kind": "pgto+pw", "--ecutwfc": "auto", "--jastrow-dtn": "u_chi"},
    kind="convert",
    parent_folders={"pw": pw.outputs.remote_folder},
    save_path=orm.Str("out/h_k025.save"),
)
vmc = jqmc("vmc", tutorial_input(EX / "02vmc_JSD_geminal/vmc.toml"), hamiltonian=converted.outputs.output_hamiltonian)
optimized = vmc.outputs.output_hamiltonian
for name, toml, family, options in (
    ("mcmc", "03mcmc_JSD_force/mcmc.toml", "mcmc", {"-b": 10, "-w": 30}),
    ("lrdmc", "04lrdmc_JSD_force/lrdmc_tau.toml", "lrdmc", {"-b": 50, "-w": 50, "-c": 20}),
):
    run = jqmc(name, tutorial_input(EX / toml), hamiltonian=optimized)
    for what in ("energy", "force"):
        show(
            f"{name} {what}",
            tool(f"{name} {what}", f"{family} compute-{what}", options, parent_folders={"run": run.outputs.remote_folder}),
        )
