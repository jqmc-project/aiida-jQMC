"""jqmc-example10 through AiiDA: unrestricted open shells, a nitrogen atom with ccECP (PySCF UHF)
and a silicon atom from QE LSDA (B-spline) in mixed precision."""

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
from _runner import EXAMPLES, file, jqmc, orm, shell, show, tool, tutorial_input

EX = Path("jqmc-example10")

results, _ = shell(
    "pyscf UHF",
    "pyscf",
    "{script}",
    nodes={"script": file(EX / "01SCF/run_pyscf.py")},
    outputs=["nitrogen_uhf_ccecp_ccpvdz.h5"],
)
converted = tool(
    "convert",
    "trexio convert-to",
    {"--jastrow-2b-parameter": 1.0},
    kind="convert",
    input_file=results["nitrogen_uhf_ccecp_ccpvdz_h5"],
)
show(
    "hamiltonian show-info",
    tool("show-info", "hamiltonian show-info", kind="convert", input_file=converted.outputs.output_hamiltonian),
)
vmc = jqmc("vmc open shell", tutorial_input(EX / "02vmc_open_shell/vmc.toml"), hamiltonian=converted.outputs.output_hamiltonian)
mcmc = jqmc("mcmc open shell", tutorial_input(EX / "03mcmc_open_shell/mcmc.toml"), hamiltonian=vmc.outputs.output_hamiltonian)
show(
    "mcmc energy",
    tool("mcmc energy", "mcmc compute-energy", {"-b": 10, "-w": 30}, parent_folders={"mcmc": mcmc.outputs.remote_folder}),
)

# Silicon atom, QE LSDA (the input reads the UPF of example07 through a relative pseudo_dir)
pw_in = (EXAMPLES / EX / "04QE_LSDA/si_atom_lsda.in").read_text().replace("'../../jqmc-example07/01DFT'", "'./'")
# With QE 7.6 this SCF did not converge in 200 iterations (serial or 4 ranks; pw.x still exits 0):
# softer mixing and more iterations. The conversion below needs a converged save.
pw_in = pw_in.replace("mixing_beta      = 0.4", "mixing_beta      = 0.1")
pw_in = pw_in.replace("electron_maxstep = 200", "electron_maxstep = 600")
_, pw = shell(
    "pw.x LSDA",
    "pw_serial",
    "-in {pwin}",
    nodes={
        "pwin": orm.SinglefileData.from_string(pw_in, filename="si_atom_lsda.in"),
        "upf": file("jqmc-example07/01DFT/Si.ccECP.upf"),
    },
    # Serial, as the tutorial's command.
    kind="pw_serial",
)
si = tool(
    "qe LSDA",
    "qe",
    {"--kind": "bspline", "--jastrow-2b-parameter": 1.0},
    kind="convert",
    parent_folders={"pw": pw.outputs.remote_folder},
    save_path=orm.Str("out/si_atom_lsda.save"),
)
si_mcmc = jqmc("mcmc Si LSDA", tutorial_input(EX / "04QE_LSDA/mcmc.toml"), hamiltonian=si.outputs.output_hamiltonian)
show(
    "Si mcmc energy",
    tool("mcmc energy", "mcmc compute-energy", {"-b": 10, "-w": 30}, parent_folders={"mcmc": si_mcmc.outputs.remote_folder}),
)
