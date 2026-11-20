"""jqmc-workflow-example01 through aiida-jqmc: H2 potential-energy surface at seven bond lengths
(PySCF RHF -> TREXIO -> JSD -> VMC with the determinant -> MCMC and LRDMC energies and forces),
and the forces compared with the derivative of the PES.

jqmc_workflow decides the run lengths from pilot runs and target error bars, and stops the VMC
on its own criteria. Here those decisions are written out (sample_to_target, vmc_indicators),
as a caller of aiida-jqmc takes them.
"""

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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _runner import jqmc, orm, sample_to_target, shell, target, tool

from aiida_jqmc.statistics import vmc_indicators, vmc_sampling_steps_for_target_error

R_VALUES = [0.50, 0.60, 0.70, 0.74, 0.80, 0.90, 1.00]
PYSCF = """from pyscf import gto, scf
from pyscf.tools import trexio
R = {R}
mol = gto.Mole()
mol.atom = f'H 0 0 {{-R / 2}}\\nH 0 0 {{R / 2}}'
mol.basis = "ccpvtz"
mol.unit = "A"
mol.cart = True
mol.build()
mf = scf.RHF(mol)
mf.max_cycle = 200
mf.kernel()
trexio.to_trexio(mf, "H2.h5")
"""
VMC = {
    "control": {"job_type": "vmc", "number_of_walkers": 4},
    "vmc": {
        "num_opt_steps": 6,
        "num_mcmc_steps": 100,
        "num_mcmc_warmup_steps": 30,
        "num_mcmc_bin_blocks": 10,
        "Dt": 1.2,
        "wf_dump_freq": 20,
        "opt_J1_param": True,
        "opt_J2_param": True,
        "opt_J3_param": True,
        "opt_geminal": True,
        "fix_geminal_rank": 1,
        "optimizer_kwargs": {"method": "sr", "delta": 0.300, "epsilon": 0.010, "use_lm": True},
    },
}
MCMC = {
    "control": {"job_type": "mcmc", "number_of_walkers": 4},
    "mcmc": {"num_mcmc_steps": 400, "num_mcmc_warmup_steps": 50, "num_mcmc_bin_blocks": 50, "Dt": 1.2, "atomic_force": True},
}
LRDMC = {
    "control": {"job_type": "lrdmc-tau", "number_of_walkers": 4},
    "lrdmc-tau": {
        "num_mcmc_steps": 600,
        "alat": 0.3,
        "num_gfmc_warmup_steps": 50,
        "num_gfmc_bin_blocks": 50,
        "num_gfmc_collect_steps": 20,
        "atomic_force": True,
    },
}

rows = []
for R in R_VALUES:
    script = orm.SinglefileData.from_string(PYSCF.format(R=R), filename="run_pyscf.py")
    results, _ = shell(f"R={R:.2f} pyscf", "pyscf", "{script}", nodes={"script": script}, outputs=["H2.h5"])
    ham = tool(
        f"R={R:.2f} convert",
        "trexio convert-to",
        {"-j1": 1.5, "-j2": 1.0, "-j3": "ao-small"},
        kind="convert",
        input_file=results["H2_h5"],
    )
    # VMC: a short pilot fixes the samples per optimization step for the target VMC error
    pilot = jqmc(
        f"R={R:.2f} vmc pilot", {**VMC, "vmc": {**VMC["vmc"], "num_opt_steps": 2}}, hamiltonian=ham.outputs.output_hamiltonian
    )
    steps = vmc_sampling_steps_for_target_error(pilot, target(5e-4))
    vmc = jqmc(
        f"R={R:.2f} vmc", {**VMC, "vmc": {**VMC["vmc"], "num_mcmc_steps": steps}}, hamiltonian=ham.outputs.output_hamiltonian
    )
    print("    VMC indicators:", vmc_indicators([vmc], window=5))
    optimized = vmc.outputs.output_hamiltonian
    mcmc = sample_to_target(f"R={R:.2f} mcmc", MCMC, target(5e-5), hamiltonian=optimized, extensions=1)
    lrdmc = sample_to_target(f"R={R:.2f} lrdmc", LRDMC, target(5e-5), hamiltonian=optimized, extensions=1)
    row = [R]
    for node in (mcmc, lrdmc):
        out = node.outputs.output_parameters.get_dict()
        row += [out["energy"], out["energy_error"], out["atomic_forces"][1]["Fz"], out["atomic_forces"][1]["Fz_err"]]
    rows.append(row)

data = np.array(rows)
r_bohr = data[:, 0] / 0.529177210903
# Force on the second atom along +z from the PES: F = -dE/dR, from a cubic fit of E(R)
derivative = {
    name: -np.polyval(np.polyder(np.polyfit(r_bohr, data[:, column], 3)), r_bohr)
    for name, column in (("MCMC", 1), ("LRDMC", 5))
}
print(f"{'R (A)':>6} {'E_MCMC':>12} {'F_MCMC':>10} {'-dE/dR':>10} {'E_LRDMC':>12} {'F_LRDMC':>10} {'-dE/dR':>10}")
for i, R in enumerate(data[:, 0]):
    print(
        f"{R:6.2f} {data[i, 1]:12.5f} {data[i, 3]:10.4f} {derivative['MCMC'][i]:10.4f}"
        f" {data[i, 5]:12.5f} {data[i, 7]:10.4f} {derivative['LRDMC'][i]:10.4f}"
    )
