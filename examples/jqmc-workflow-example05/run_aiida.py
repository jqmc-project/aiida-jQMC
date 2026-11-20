"""jqmc-workflow-example05 through aiida-jqmc: periodic all-electron hydrogen, QE -> pGTO+PW
(DTN u_chi) -> VMC (DTN Jastrow + MO coefficients, two production runs) -> MCMC and LRDMC with
forces to a target error bar. The workflow counterpart of jqmc-example08."""

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
from _runner import continuation, file, jqmc, orm, sample_to_target, shell, target, tool

from aiida_jqmc.statistics import vmc_indicators, vmc_sampling_steps_for_target_error

EX = Path("jqmc-workflow-example05/qe-pbc")
VMC = {
    "control": {"job_type": "vmc", "number_of_walkers": 1, "verbosity": "high"},
    "vmc": {
        "num_opt_steps": 3,
        "num_mcmc_steps": 100,
        "num_mcmc_warmup_steps": 30,
        "num_mcmc_bin_blocks": 10,
        "Dt": 1.0,
        "wf_dump_freq": 20,
        "opt_J1_param": False,
        "opt_J2_param": False,
        "opt_J3_param": False,
        "opt_JDTN_u_param": True,
        "opt_JDTN_chi_param": True,
        "opt_JDTN_f_param": False,
        "opt_geminal": True,
        "optimizer_kwargs": {"method": "sr", "delta": 0.100, "epsilon": 0.001, "use_lm": True},
    },
}
MCMC = {
    "control": {"job_type": "mcmc", "number_of_walkers": 1, "verbosity": "high"},
    "mcmc": {"num_mcmc_steps": 200, "num_mcmc_warmup_steps": 30, "num_mcmc_bin_blocks": 10, "Dt": 1.0, "atomic_force": True},
}
LRDMC = {
    "control": {"job_type": "lrdmc-tau", "number_of_walkers": 1, "verbosity": "high"},
    "lrdmc-tau": {
        "num_mcmc_steps": 300,
        "alat": 0.30,
        "non_local_move": "tmove",
        "num_gfmc_warmup_steps": 30,
        "num_gfmc_bin_blocks": 10,
        "num_gfmc_collect_steps": 20,
        "atomic_force": True,
    },
}

_, pw = shell(
    "pw.x", "pw", "-in {pwin}", nodes={"pwin": file(EX / "h_scf.in"), "upf": file(EX / "H.coulomb-ae.UPF")}, kind="pw"
)
ham = tool(
    "qe",
    "qe",
    {"--kind": "pgto+pw", "--ecutwfc": "auto", "--jastrow-dtn": "u_chi"},
    kind="convert",
    parent_folders={"pw": pw.outputs.remote_folder},
    save_path=orm.Str("out/h_k025.save"),
)
pilot = jqmc("vmc pilot", {**VMC, "vmc": {**VMC["vmc"], "num_opt_steps": 2}}, hamiltonian=ham.outputs.output_hamiltonian)
VMC["vmc"]["num_mcmc_steps"] = vmc_sampling_steps_for_target_error(pilot, target(5e-3))
vmc = jqmc("vmc run 1", VMC, hamiltonian=ham.outputs.output_hamiltonian)
vmc2 = jqmc("vmc run 2", continuation(VMC, VMC["vmc"]["num_opt_steps"]), parent_folder=vmc.outputs.remote_folder)
print("    VMC indicators:", vmc_indicators([vmc, vmc2], window=5))
for label, parameters in (("mcmc", MCMC), ("lrdmc", LRDMC)):
    node = sample_to_target(label, parameters, target(5e-4), hamiltonian=vmc2.outputs.output_hamiltonian)
    out = node.outputs.output_parameters.get_dict()
    print(f"    {label}: E = {out['energy']} +- {out['energy_error']}, forces = {out['atomic_forces']}")
