"""jqmc-workflow-example03 through aiida-jqmc: water (ccECP, cc-pVTZ) VMC (Pade J2 + AO-small J3,
determinant fixed), then MCMC and LRDMC energies and atomic forces to target error bars.

The two VMC production runs of jqmc_workflow (max_continuation = 2) are a VMC calculation and its
extension; the run lengths of MCMC and LRDMC follow from a pilot and the target error bar."""

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
from _runner import continuation, jqmc, orm, sample_to_target, shell, target, tool

from aiida_jqmc.statistics import vmc_indicators, vmc_sampling_steps_for_target_error

PYSCF = '''from pyscf import gto, scf
from pyscf.tools import trexio
mol = gto.Mole()
mol.atom = """
    O    5.00000000   7.14707700   7.65097100
    H    4.06806600   6.94297500   7.56376100
    H    5.38023700   6.89696300   6.80798400
"""
mol.basis = "ccecp-ccpvtz"
mol.unit = "A"
mol.ecp = "ccecp"
mol.cart = True
mol.build()
mf = scf.HF(mol)
mf.max_cycle = 200
mf.kernel()
trexio.to_trexio(mf, "water_ccecp_ccpvtz.h5")
'''
VMC = {
    "control": {"job_type": "vmc", "number_of_walkers": 1},
    "vmc": {
        "num_opt_steps": 4,
        "num_mcmc_steps": 100,
        "num_mcmc_warmup_steps": 30,
        "num_mcmc_bin_blocks": 10,
        "Dt": 2.0,
        "wf_dump_freq": 20,
        "opt_J1_param": False,
        "opt_J2_param": True,
        "opt_J3_param": True,
        "opt_geminal": False,
        "optimizer_kwargs": {"method": "sr", "delta": 0.350, "epsilon": 0.001, "use_lm": True},
    },
}
MCMC = {
    "control": {"job_type": "mcmc", "number_of_walkers": 1},
    "mcmc": {"num_mcmc_steps": 200, "num_mcmc_warmup_steps": 30, "num_mcmc_bin_blocks": 10, "Dt": 2.0, "atomic_force": True},
}
LRDMC = {
    "control": {"job_type": "lrdmc-tau", "number_of_walkers": 1},
    "lrdmc-tau": {
        "num_mcmc_steps": 300,
        "alat": 0.30,
        "num_gfmc_warmup_steps": 30,
        "num_gfmc_bin_blocks": 10,
        "num_gfmc_collect_steps": 20,
        "atomic_force": True,
    },
}

script = orm.SinglefileData.from_string(PYSCF, filename="run_pyscf.py")
results, _ = shell("pyscf", "pyscf", "{script}", nodes={"script": script}, outputs=["water_ccecp_ccpvtz.h5"])
ham = tool(
    "convert", "trexio convert-to", {"-j2": 1.0, "-j3": "ao-small"}, kind="convert", input_file=results["water_ccecp_ccpvtz_h5"]
)
pilot = jqmc("vmc pilot", {**VMC, "vmc": {**VMC["vmc"], "num_opt_steps": 2}}, hamiltonian=ham.outputs.output_hamiltonian)
VMC["vmc"]["num_mcmc_steps"] = vmc_sampling_steps_for_target_error(pilot, target(3e-3))
vmc = jqmc("vmc run 1", VMC, hamiltonian=ham.outputs.output_hamiltonian)
vmc2 = jqmc("vmc run 2", continuation(VMC, VMC["vmc"]["num_opt_steps"]), parent_folder=vmc.outputs.remote_folder)
print("    VMC indicators:", vmc_indicators([vmc, vmc2], window=5))
optimized = vmc2.outputs.output_hamiltonian
for label, parameters in (("mcmc", MCMC), ("lrdmc", LRDMC)):
    node = sample_to_target(label, parameters, target(5e-4), hamiltonian=optimized)
    out = node.outputs.output_parameters.get_dict()
    print(f"    {label}: E = {out['energy']} +- {out['energy_error']}, forces = {out['atomic_forces']}")
