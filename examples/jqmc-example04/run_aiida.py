"""jqmc-example04 through AiiDA: S22 water-dimer binding energy with JSD and JAGP
(VMC, MCMC and LRDMC for two monomers and the dimer), the binding energy as a calcfunction."""

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

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _runner import file, jqmc, orm, shell, tool, tutorial_input
from aiida.engine import calcfunction

EX = Path("jqmc-example04")
HA_TO_KCAL = 627.509
SYSTEMS = {
    "dimer": ("03_S22_water_dimer", "water_dimer.h5"),
    "monomer_1": ("01_S22_water_monomer_1", "water_monomer_1.h5"),
    "monomer_2": ("02_S22_water_monomer_2", "water_monomer_2.h5"),
}


@calcfunction
def binding_energy(dimer: orm.Dict, monomer_1: orm.Dict, monomer_2: orm.Dict) -> orm.Dict:
    """E_dimer - E_monomer_1 - E_monomer_2 from the final energies of three runs."""
    e = [r["energy"] for r in (dimer, monomer_1, monomer_2)]
    s = [r["energy_error"] for r in (dimer, monomer_1, monomer_2)]
    e_bind, s_bind = e[0] - e[1] - e[2], math.sqrt(sum(x * x for x in s))
    return orm.Dict(
        {
            "binding_energy": e_bind,
            "binding_energy_error": s_bind,
            "binding_energy_kcal_mol": e_bind * HA_TO_KCAL,
            "binding_energy_error_kcal_mol": s_bind * HA_TO_KCAL,
        }
    )


energies = {}
for name, (directory, trexio_name) in SYSTEMS.items():
    root = EX / directory
    results, _ = shell(
        f"{name} pyscf", "pyscf", "{script}", nodes={"script": file(root / "01DFT/run_pyscf.py")}, outputs=[trexio_name]
    )
    converted = tool(
        f"{name} convert",
        "trexio convert-to",
        {"-j2": 1.0, "-j3": "ao-medium"},
        kind="convert",
        input_file=results[trexio_name.replace(".", "_")],
    )
    vmc = jqmc(f"{name} vmc JSD", tutorial_input(root / "02vmc_JSD/vmc.toml"), hamiltonian=converted.outputs.output_hamiltonian)
    cont = tutorial_input(root / "02vmc_JSD/vmc_cont.toml")
    for key in ("mcmc_seed", "number_of_walkers"):
        cont["control"].pop(key, None)
    for key in ("Dt", "epsilon_AS", "num_mcmc_per_measurement"):
        cont["vmc"].pop(key, None)
    vmc_cont = jqmc(f"{name} vmc JSD continuation", cont, parent_folder=vmc.outputs.remote_folder)
    tool(
        f"{name} analyze-output",
        "vmc analyze-output",
        {"-s": "vmc_JSD.jpg"},
        parent_folders={"a": vmc.outputs.remote_folder, "b": vmc_cont.outputs.remote_folder},
    )
    jsd = vmc_cont.outputs.output_hamiltonian
    energies[(name, "JSD", "VMC")] = jqmc(f"{name} mcmc JSD", tutorial_input(root / "03mcmc_JSD/mcmc.toml"), hamiltonian=jsd)
    energies[(name, "JSD", "LRDMC")] = jqmc(
        f"{name} lrdmc JSD", tutorial_input(root / "04lrdmc_JSD/lrdmc.toml"), hamiltonian=jsd
    )
    jagp = tool(f"{name} conv-wf", "hamiltonian conv-wf", {"--convert-to": "jagp"}, kind="convert", input_file=jsd)
    vmc_jagp = jqmc(
        f"{name} vmc JAGP", tutorial_input(root / "05vmc_JAGP/vmc.toml"), hamiltonian=jagp.outputs.output_hamiltonian
    )
    energies[(name, "JAGP", "VMC")] = jqmc(
        f"{name} mcmc JAGP", tutorial_input(root / "06mcmc_JAGP/mcmc.toml"), hamiltonian=vmc_jagp.outputs.output_hamiltonian
    )
    energies[(name, "JAGP", "LRDMC")] = jqmc(
        f"{name} lrdmc JAGP", tutorial_input(root / "07lrdmc_JAGP/lrdmc.toml"), hamiltonian=vmc_jagp.outputs.output_hamiltonian
    )

for ansatz in ("JSD", "JAGP"):
    for method in ("VMC", "LRDMC"):
        result = binding_energy(*(energies[(name, ansatz, method)].outputs.output_parameters for name in SYSTEMS))
        print(f"{ansatz:4s} {method:5s}", result.get_dict(), f"(calcfunction <{result.creator.pk}>)")
