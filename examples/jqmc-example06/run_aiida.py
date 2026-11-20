"""jqmc-example06 through AiiDA: periodic ice Ih from PySCF pGTOs at a non-Gamma complex twist,
VMC, MCMC, and LRDMC at three lattice spacings with continuum extrapolation."""

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
from _runner import file, jqmc, shell, show, tool, tutorial_input

EX = Path("jqmc-example06")

results, _ = shell(
    "pyscf (pbc)", "pyscf", "{script}", nodes={"script": file(EX / "01DFT/run_pyscf.py")}, outputs=["ice_ccecp_ccpvdz_cart.h5"]
)
trexio = results["ice_ccecp_ccpvdz_cart_h5"]
show("trexio show-info", tool("trexio show-info", "trexio show-info", kind="convert", input_file=trexio))
converted = tool("convert", "trexio convert-to", {"-j2": 1.0, "-j3": "ao-small"}, kind="convert", input_file=trexio)
# HF check: the determinant alone
hf = tool("convert (HF check)", "trexio convert-to", {"-j3": "none"}, kind="convert", input_file=trexio)
hf_mcmc = jqmc("mcmc HF check", tutorial_input(EX / "03mcmc_JSD/mcmc.toml"), hamiltonian=hf.outputs.output_hamiltonian)
show(
    "HF check energy",
    tool("mcmc energy", "mcmc compute-energy", {"-b": 100, "-w": 50}, parent_folders={"mcmc": hf_mcmc.outputs.remote_folder}),
)

vmc = jqmc("vmc", tutorial_input(EX / "02vmc_JSD/vmc.toml"), hamiltonian=converted.outputs.output_hamiltonian)
optimized = vmc.outputs.output_hamiltonian
mcmc = jqmc("mcmc", tutorial_input(EX / "03mcmc_JSD/mcmc.toml"), hamiltonian=optimized)
show(
    "mcmc energy",
    tool("mcmc energy", "mcmc compute-energy", {"-b": 100, "-w": 50}, parent_folders={"mcmc": mcmc.outputs.remote_folder}),
)
runs = {}
for alat in (0.20, 0.25, 0.30):
    runs[f"a{round(alat * 100):03d}"] = jqmc(
        f"lrdmc a={alat:.2f}",
        tutorial_input(EX / "04lrdmc_JSD/lrdmc_tau.toml", **{"lrdmc-tau.alat": alat}),
        hamiltonian=optimized,
    )
show(
    "lrdmc a=0.30 energy",
    tool(
        "lrdmc energy",
        "lrdmc compute-energy",
        {"-b": 50, "-w": 50, "-c": 20},
        parent_folders={"run": runs["a030"].outputs.remote_folder},
    ),
)
show(
    "extrapolate",
    tool(
        "extrapolate",
        "lrdmc extrapolate-energy",
        {"-p": 1, "-b": 50, "-w": 50, "-c": 20, "-s": "lrdmc_ext.jpg"},
        parent_folders={key: node.outputs.remote_folder for key, node in runs.items()},
    ),
)
