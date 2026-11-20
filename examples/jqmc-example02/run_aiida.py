"""jqmc-example02 through AiiDA: water with a two-body Jastrow, VMC optimization with a restart,
MCMC energy and atomic forces."""

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

EX = Path("jqmc-example02")

results, _ = shell(
    "pyscf", "pyscf", "{script}", nodes={"script": file(EX / "01DFT/run_pyscf.py")}, outputs=["water_ccecp_ccpvtz.h5"]
)
converted = tool(
    "convert", "trexio convert-to", {"-j2": 1.0, "-j3": "none"}, kind="convert", input_file=results["water_ccecp_ccpvtz_h5"]
)
parameters = tutorial_input(EX / "02vmc/vmc.toml")
vmc = jqmc("vmc", parameters, hamiltonian=converted.outputs.output_hamiltonian)
# Continuation (out_vmc_cont): more optimization steps from the checkpoint
cont = jqmc(
    "vmc continuation",
    {
        "control": {"job_type": "vmc"},
        "vmc": {k: v for k, v in parameters["vmc"].items() if k not in ("Dt", "epsilon_AS", "num_mcmc_per_measurement")},
    },
    parent_folder=vmc.outputs.remote_folder,
)
steps = cont.outputs.output_parameters["steps"]
print("    continuation steps:", [s["step"] for s in steps])
analysis = tool(
    "vmc analyze-output",
    "vmc analyze-output",
    {"-s": "vmc.jpg"},
    parent_folders={"a_vmc": vmc.outputs.remote_folder, "b_cont": cont.outputs.remote_folder},
)
mcmc = jqmc("mcmc", tutorial_input(EX / "03mcmc/mcmc.toml"), hamiltonian=cont.outputs.output_hamiltonian)
for what in ("energy", "force"):
    show(
        f"mcmc {what}",
        tool(f"mcmc {what}", f"mcmc compute-{what}", {"-w": 30, "-b": 10}, parent_folders={"mcmc": mcmc.outputs.remote_folder}),
    )
