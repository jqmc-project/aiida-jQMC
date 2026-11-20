"""jqmc-example01 through AiiDA: water (ccECP), JSD optimization, MCMC, and lrdmc-tau at five
lattice spacings with continuum extrapolation."""

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
from _runner import SCALE, file, jqmc, shell, show, tool, tutorial_input

EX = Path("jqmc-example01")

results, _ = shell(
    "pyscf", "pyscf", "{script}", nodes={"script": file(EX / "01DFT/run_pyscf.py")}, outputs=["water_ccecp_ccpvtz.h5"]
)
trexio = results["water_ccecp_ccpvtz_h5"]
show("trexio show-info", tool("trexio show-info", "trexio show-info", kind="convert", input_file=trexio))
converted = tool("convert", "trexio convert-to", {"-j2": 1.0, "-j3": "mo"}, kind="convert", input_file=trexio)
vmc = jqmc("vmc", tutorial_input(EX / "02vmc_JSD/vmc.toml"), hamiltonian=converted.outputs.output_hamiltonian)
analysis = tool(
    "vmc analyze-output", "vmc analyze-output", {"-s": "vmc.jpg"}, parent_folders={"vmc": vmc.outputs.remote_folder}
)
print("    graph:", analysis.outputs.output_graph.filename)
optimized = vmc.outputs.output_hamiltonian

mcmc = jqmc("mcmc", tutorial_input(EX / "03mcmc_JSD/mcmc.toml"), hamiltonian=optimized)
# -w 500 compares a long warmup; shortened runs (SCALE < 1) have fewer measurements than that.
for warmup in (30, 500 if SCALE >= 1 else 60):
    show(
        f"mcmc energy -w {warmup}",
        tool(
            "mcmc energy", "mcmc compute-energy", {"-w": warmup, "-b": 10}, parent_folders={"mcmc": mcmc.outputs.remote_folder}
        ),
    )

runs = {}
for alat in (0.10, 0.15, 0.20, 0.25, 0.30):
    runs[f"a{round(alat * 100):03d}"] = jqmc(
        f"lrdmc a={alat:.2f}", tutorial_input(EX / "04lrdmc_JSD/lrdmc.toml", **{"lrdmc-tau.alat": alat}), hamiltonian=optimized
    )
# Warmup comparison of the tutorial; shortened runs (SCALE < 1) have fewer branchings than -w 200.
for warmup in (50, 100, 200) if SCALE >= 1 else (30, 40, 50):
    show(
        f"lrdmc a=0.10 -w {warmup}",
        tool(
            "lrdmc energy",
            "lrdmc compute-energy",
            {"-w": warmup, "-b": 10, "-c": 20},
            parent_folders={"run": runs["a010"].outputs.remote_folder},
        ),
    )
extrapolation = tool(
    "extrapolate",
    "lrdmc extrapolate-energy",
    {"-w": 100 if SCALE >= 1 else 40, "-b": 10, "-c": 20, "-s": "lrdmc_ext.jpg"},
    parent_folders={key: node.outputs.remote_folder for key, node in runs.items()},
)
show("extrapolate", extrapolation)
print("    graph:", extrapolation.outputs.output_graph.filename)
