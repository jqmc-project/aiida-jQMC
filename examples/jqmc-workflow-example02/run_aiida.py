"""jqmc-workflow-example02 through aiida-jqmc: GPU walker-scaling benchmark for water (ccECP,
cc-pVTZ). PySCF DFT-LDA trial, VMC of Pade J2 + AO-small J3 on one GPU, then MCMC and LRDMC
(E_scf from the MCMC energy) for 8 ... 8192 walkers per MPI rank, and the timing table.

Needs a GPU node: [options.gpu] (the GPU request, e.g. custom_scheduler_commands) and codes.jqmc_gpu in config.toml."""

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
from _runner import jqmc, orm, shell, tool
from aiida.engine import calcfunction

WALKER_COUNTS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
PYSCF = '''from pyscf import gto, scf
from pyscf.tools import trexio
mol = gto.Mole()
mol.atom = """
O 5.000000 7.147077 7.650971
H 4.068066 6.942975 7.563761
H 5.380237 6.896963 6.807984
"""
mol.basis = "ccecp-ccpvtz"
mol.unit = "A"
mol.ecp = "ccecp"
mol.cart = True
mol.build()
mf = scf.KS(mol).density_fit()
mf.max_cycle = 200
mf.xc = "LDA_X,LDA_C_PZ"
mf.kernel()
trexio.to_trexio(mf, "water_trexio.hdf5")
'''
GPU = {"kind": "gpu", "code_name": "jqmc_gpu"}


@calcfunction
def lrdmc_input(parameters: orm.Dict, mcmc: orm.Dict) -> orm.Dict:
    """The LRDMC input with E_scf from the MCMC energy (kept in the provenance)."""
    result = parameters.get_dict()
    result["lrdmc-bra"]["E_scf"] = mcmc["energy"]
    return orm.Dict(result)


script = orm.SinglefileData.from_string(PYSCF, filename="run_pyscf.py")
results, _ = shell("pyscf", "pyscf", "{script}", nodes={"script": script}, outputs=["water_trexio.hdf5"])
ham = tool(
    "convert", "trexio convert-to", {"-j2": 1.0, "-j3": "ao-small"}, kind="convert", input_file=results["water_trexio_hdf5"]
)
vmc = jqmc(
    "vmc",
    {
        "control": {"job_type": "vmc", "number_of_walkers": 256},
        "vmc": {
            "num_opt_steps": 5,
            "num_mcmc_steps": 100,
            "num_mcmc_warmup_steps": 30,
            "num_mcmc_bin_blocks": 10,
            "Dt": 2.0,
            "wf_dump_freq": 10,
            "opt_J1_param": False,
            "opt_J2_param": True,
            "opt_J3_param": True,
            "opt_geminal": False,
            "optimizer_kwargs": {"method": "sr", "delta": 0.350, "epsilon": 0.001, "use_lm": True},
        },
    },
    hamiltonian=ham.outputs.output_hamiltonian,
    **GPU,
)
trial = vmc.outputs.output_hamiltonian

print(f"{'walkers':>8} {'MCMC ms/step':>13} {'LRDMC ms/branch':>16}")
for walkers in WALKER_COUNTS:
    mcmc = jqmc(
        f"mcmc w={walkers}",
        {
            "control": {"job_type": "mcmc", "number_of_walkers": walkers},
            "mcmc": {"num_mcmc_steps": 100, "num_mcmc_warmup_steps": 30, "num_mcmc_bin_blocks": 10, "Dt": 2.0},
        },
        hamiltonian=trial,
        **GPU,
    )
    base = orm.Dict(
        {
            "control": {"job_type": "lrdmc-bra", "number_of_walkers": walkers},
            "lrdmc-bra": {
                "num_mcmc_steps": 100,
                "alat": 0.30,
                "num_mcmc_per_measurement": 40,
                "num_gfmc_warmup_steps": 25,
                "num_gfmc_bin_blocks": 10,
                "num_gfmc_collect_steps": 5,
            },
        }
    )
    lrdmc = jqmc(f"lrdmc w={walkers}", lrdmc_input(base, mcmc.outputs.output_parameters), hamiltonian=trial, **GPU)
    m, g = mcmc.outputs.output_parameters.get_dict(), lrdmc.outputs.output_parameters.get_dict()
    mcmc_ms = m["timing_breakdown"].get("mcmc_update", float("nan"))
    lrdmc_ms = g["timing_breakdown"].get("projection", float("nan"))
    print(f"{walkers:8d} {mcmc_ms:13.2f} {lrdmc_ms:16.2f}")
