"""jqmc-example03 through AiiDA: water, Jastrow-only optimization, then geminal/MO optimization,
and the MCMC energies of both trials."""

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

EX = Path("jqmc-example03")

results, _ = shell(
    "pyscf", "pyscf", "{script}", nodes={"script": file(EX / "01DFT/run_pyscf.py")}, outputs=["water_ccecp_ccpvtz.h5"]
)
converted = tool(
    "convert", "trexio convert-to", {"-j2": 1.0, "-j3": "ao-small"}, kind="convert", input_file=results["water_ccecp_ccpvtz_h5"]
)
jastrow = jqmc("vmc jastrow", tutorial_input(EX / "02vmc_jastrow/vmc.toml"), hamiltonian=converted.outputs.output_hamiltonian)
geminal = jqmc("vmc geminal", tutorial_input(EX / "03vmc_geminal/vmc.toml"), hamiltonian=jastrow.outputs.output_hamiltonian)
for name, toml, trial in (("jastrow", "04mcmc_jastrow", jastrow), ("geminal", "05mcmc_geminal", geminal)):
    mcmc = jqmc(f"mcmc {name}", tutorial_input(EX / f"{toml}/mcmc.toml"), hamiltonian=trial.outputs.output_hamiltonian)
    show(
        f"mcmc {name} energy",
        tool("mcmc energy", "mcmc compute-energy", {"-w": 30, "-b": 10}, parent_folders={"mcmc": mcmc.outputs.remote_folder}),
    )
