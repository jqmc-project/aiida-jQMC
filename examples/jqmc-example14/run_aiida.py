"""jQMC example14 through AiiDA: tiled Li, weighted-sector VMC, restart, MCMC and LRDMC.

The small committed PySCF/TREXIO source belongs to jQMC's converter tests and
is the same source used by the example14 documentation.  Orbital tiling is a
conversion input only; every QMC process receives the converted HDF5.
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

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _runner import EXAMPLES, jqmc, orm, tool  # noqa: I001


SOURCE = EXAMPLES.parent / "tests/trexio_example_files/lithium_tiling_sectors.h5"
source = orm.SinglefileData(SOURCE)
single_sector = os.environ.get("JQMC_EXAMPLE14_SINGLE_SECTOR") == "1"
selection = {"supertwist": orm.List(list=[0.22, 0.13, 0.09])} if single_sector else {"twist_average": orm.Bool(True)}
converted = tool(
    "TREXIO tiled single sector" if single_sector else "TREXIO tiled ensemble",
    "trexio convert-to",
    {"-j1": 0.5, "-j2": 0.7, "-j3": "ao-small"},
    kind="convert",
    input_file=source,
    tiling=orm.List(list=[2, 1, 1]),
    **selection,
)
metadata = converted.outputs.conversion_metadata.get_dict()
print("    physical sites:", len(converted.outputs.physical_structure.sites))
print("    sectors:", metadata["supertwists"], "weights:", metadata["normalized_sector_weights"])
print("    jQMC:", metadata["jqmc"])

production_steps = 40 if single_sector else 130

vmc_parameters = {
    "control": {"job_type": "vmc", "mcmc_seed": 34456, "number_of_walkers": 4, "verbosity": "high"},
    "vmc": {
        "num_mcmc_steps": 100,
        "num_mcmc_per_measurement": 10,
        "num_mcmc_warmup_steps": 30,
        "num_mcmc_bin_blocks": 10,
        "num_opt_steps": 2,
        "wf_dump_freq": 1,
        "opt_J1_param": True,
        "opt_J2_param": True,
        "opt_J3_param": True,
        "opt_geminal": False,
        "opt_lambda_basis_exp": False,
        "opt_lambda_basis_coeff": False,
        "opt_J3_basis_exp": False,
        "opt_J3_basis_coeff": False,
    },
}
vmc = jqmc("tiled ensemble VMC", vmc_parameters, hamiltonian=converted.outputs.output_hamiltonian)
optimized = vmc.outputs.output_hamiltonian

mcmc_parameters = {
    "control": {"job_type": "mcmc", "mcmc_seed": 34457, "number_of_walkers": 4, "verbosity": "high"},
    "mcmc": {
        "num_mcmc_steps": production_steps,
        "num_mcmc_per_measurement": 10,
        "num_mcmc_warmup_steps": 30,
        "num_mcmc_bin_blocks": 10,
        "atomic_force": True,
        "compute_stress": True,
    },
}
mcmc = jqmc("tiled ensemble MCMC", mcmc_parameters, hamiltonian=optimized)
restart = jqmc(
    "single-sector file restart" if single_sector else "completed ensemble directory restart",
    {
        "control": {"job_type": "mcmc"},
        "mcmc": {
            "num_mcmc_steps": 1 if single_sector else production_steps,
            "num_mcmc_per_measurement": 10,
            "num_mcmc_warmup_steps": 30,
            "num_mcmc_bin_blocks": 10,
        },
    },
    parent_folder=mcmc.outputs.remote_folder,
)

lrdmc = jqmc(
    "tiled ensemble LRDMC",
    {
        "control": {"job_type": "lrdmc-tau", "mcmc_seed": 34458, "number_of_walkers": 4, "verbosity": "high"},
        "lrdmc-tau": {
            "num_mcmc_steps": production_steps,
            "num_gfmc_warmup_steps": 30,
            "num_gfmc_bin_blocks": 10,
            "num_gfmc_collect_steps": 5 if single_sector else 10,
            "tau": 0.1,
            "alat": 0.3,
            "non_local_move": "tmove",
        },
    },
    hamiltonian=optimized,
)

for name, node in (("VMC", vmc), ("MCMC", mcmc), ("restart", restart), ("LRDMC", lrdmc)):
    output = node.outputs.output_parameters.get_dict()
    backend = output["environment"]["execution_backend"]
    if "Rust" not in backend:
        raise RuntimeError(f"{name} used {backend!r}; the CPU production check requires Rust")
    print(f"    {name}: backend={backend}, E={output.get('energy')}, twist={output.get('twist_average')}")
