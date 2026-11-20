"""jqmc-example09 through AiiDA: eight-twist silicon (QE, B-spline), twist-averaged MCMC and LRDMC
with stress, pressure and BZ-averaged MPC in mixed precision, extrapolation, and restart."""

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
from _runner import EXAMPLES, file, jqmc, orm, shell, tool, tutorial_input

EX = Path("jqmc-example09")

# 01DFT: pw.x on the shifted 2x2x2 grid. The input reads the UPF of example07 through a relative
# pseudo_dir; the job gets the UPF next to the input instead.
pw_in = (EXAMPLES / EX / "01DFT/si_twist.in").read_text().replace("'../../jqmc-example07/01DFT'", "'./'")
pwin = orm.SinglefileData.from_string(pw_in, filename="si_twist.in")
_, pw = shell(
    "pw.x",
    "pw",
    "-in {pwin}",
    nodes={"pwin": pwin, "upf": file("jqmc-example07/01DFT/Si.ccECP.upf")},
    kind="pw",
)

# 02: one twist-average Hamiltonian
converted = tool(
    "qe --twist-average",
    "qe",
    {"--kind": "bspline", "--jastrow-2b-parameter": 1.0},
    kind="convert",
    twist_average=orm.Bool(True),
    parent_folders={"pw": pw.outputs.remote_folder},
    save_path=orm.Str("out/silicon_twist.save"),
)
hamiltonian = converted.outputs.output_hamiltonian

mcmc = jqmc("mcmc twist", tutorial_input(EX / "02mcmc_twist_stress/mcmc.toml"), hamiltonian=hamiltonian)
out = mcmc.outputs.output_parameters.get_dict()
print("twist_average:", out["twist_average"])


def show(name, node):
    print("   ", name, {k: v for k, v in node.outputs.output_parameters.get_dict().items() if v is not None})


# Post-processing of the twist checkpoint directory
show(
    "compute-energy",
    tool("mcmc energy", "mcmc compute-energy", {"-b": 10, "-w": 30}, parent_folders={"mcmc": mcmc.outputs.remote_folder}),
)
show(
    "compute-force",
    tool("mcmc force", "mcmc compute-force", {"-b": 10, "-w": 30}, parent_folders={"mcmc": mcmc.outputs.remote_folder}),
)

# Restart of the completed directory: jqmc skips every completed twist and re-reads their summaries.
restart = jqmc(
    "mcmc restart (complete)",
    {
        "control": {"job_type": "mcmc"},
        "mcmc": {k: v for k, v in mcmc.inputs.jqmc.parameters["mcmc"].items() if k.startswith("num_")},
    },
    parent_folder=mcmc.outputs.remote_folder,
)
print(
    "    restart E_TA:", restart.outputs.output_parameters["twist_average"]["energy"], "first:", out["twist_average"]["energy"]
)

# 03/04: twist-averaged LRDMC at alat = 0.30, 0.20, 0.40 (fresh runs, seeds fixed before sampling), then extrapolation
runs = {}
for index, alat in enumerate((0.30, 0.20, 0.40)):
    updates = {"lrdmc-tau.alat": alat} if index == 0 else {"lrdmc-tau.alat": alat, "control.mcmc_seed": 34468 + index - 1}
    runs[f"a{int(alat * 100):03d}"] = jqmc(
        f"lrdmc twist a={alat:.2f}",
        tutorial_input(EX / "03lrdmc_twist_stress/lrdmc_tau.toml", **updates),
        hamiltonian=hamiltonian,
    )
show(
    "lrdmc a=0.30 compute-energy",
    tool(
        "lrdmc energy",
        "lrdmc compute-energy",
        {"-b": 50, "-w": 50, "-c": 20},
        parent_folders={"run": runs["a030"].outputs.remote_folder},
    ),
)
extrapolation = tool(
    "extrapolate",
    "lrdmc extrapolate-energy",
    {"-p": 1, "-b": 50, "-w": 50, "-c": 20},
    parent_folders={key: node.outputs.remote_folder for key, node in runs.items()},
)
show("extrapolate-energy", extrapolation)
