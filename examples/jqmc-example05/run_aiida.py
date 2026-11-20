"""jqmc-example05 through AiiDA: all-electron H2, JSD and JAGP, energies and forces."""

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

EX = Path("jqmc-example05")

# 01DFT: PySCF -> TREXIO
results, _ = shell("pyscf", "pyscf", "{script}", nodes={"script": file(EX / "01DFT/run_pyscf.py")}, outputs=["H2_R_0.74.h5"])
trexio = results["H2_R_0_74_h5"]

# 02vmc_JSD
converted = tool("convert", "trexio convert-to", {"-j1": 1.0, "-j2": 1.0, "-j3": "mo"}, kind="convert", input_file=trexio)
vmc_jsd = jqmc("vmc JSD", tutorial_input(EX / "02vmc_JSD/vmc.toml"), hamiltonian=converted.outputs.output_hamiltonian)


def sample(name, toml, hamiltonian, energy_options):
    run = jqmc(name, tutorial_input(EX / toml), hamiltonian=hamiltonian)
    family = "lrdmc" if "lrdmc" in toml else "mcmc"
    for what in ("energy", "force"):
        node = tool(
            f"{name} {what}", f"{family} compute-{what}", energy_options, parent_folders={"run": run.outputs.remote_folder}
        )
        print("   ", {k: v for k, v in node.outputs.output_parameters.get_dict().items() if v is not None})
    return run


sample("mcmc JSD", "03mcmc_JSD/mcmc.toml", vmc_jsd.outputs.output_hamiltonian, {"-w": 30, "-b": 10})
sample("lrdmc JSD", "04lrdmc_JSD/lrdmc.toml", vmc_jsd.outputs.output_hamiltonian, {"-w": 25, "-b": 10, "-c": 5})

# 05vmc_JAGP: JSD -> JAGP
jagp = tool(
    "conv-wf", "hamiltonian conv-wf", {"--convert-to": "jagp"}, kind="convert", input_file=vmc_jsd.outputs.output_hamiltonian
)
vmc_jagp = jqmc("vmc JAGP", tutorial_input(EX / "05vmc_JAGP/vmc.toml"), hamiltonian=jagp.outputs.output_hamiltonian)
sample("mcmc JAGP", "06mcmc_JAGP/mcmc.toml", vmc_jagp.outputs.output_hamiltonian, {"-w": 30, "-b": 10})
sample("lrdmc JAGP", "07lrdmc_JAGP/lrdmc.toml", vmc_jagp.outputs.output_hamiltonian, {"-w": 25, "-b": 10, "-c": 5})
print("example05 done", orm.load_node(vmc_jagp.pk).outputs.output_parameters["opt_step_counter"])
