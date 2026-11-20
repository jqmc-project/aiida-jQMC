"""jqmc-workflow-example04 through aiida-jqmc: silicon equation of state over five lattice
constants (QE on the shifted 2x2x2 grid -> twist-averaged B-spline + DTN u_chi_f -> VMC -> MCMC
and LRDMC), a Birch-Murnaghan fit of E(V) as a calcfunction, and an LRDMC alat -> 0
extrapolation at the scan point closest to the LRDMC a0."""

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
from _runner import SCALE, file, jqmc, orm, shell, show, tool
from aiida.engine import calcfunction

LATTICE_CONSTANTS = [5.25, 5.35, 5.45, 5.55, 5.65]
EXT_ALATS = [0.20, 0.40]
WALKERS = 4


def steps(n: int, minimum: int) -> int:
    return max(minimum, int(n * SCALE))


def pw_input(a: float) -> str:
    h = a / 2.0
    return f"""&CONTROL
    calculation = 'scf'
    prefix      = 'si'
    outdir      = './'
    pseudo_dir  = './'
/
&SYSTEM
    ibrav = 0, nat = 2, ntyp = 1, nbnd = 8, ecutwfc = 100.0, ecutrho = 400.0,
    occupations = 'fixed', input_dft = 'pz', nosym = .true., noinv = .true.
/
&ELECTRONS
    conv_thr = 1.0d-9, mixing_beta = 0.5
/
ATOMIC_SPECIES
    Si  28.085  Si.ccECP.upf
CELL_PARAMETERS {{angstrom}}
    0.000000  {h:.6f}  {h:.6f}
    {h:.6f}  0.000000  {h:.6f}
    {h:.6f}  {h:.6f}  0.000000
ATOMIC_POSITIONS {{crystal}}
    Si  0.00  0.00  0.00
    Si  0.25  0.25  0.25
K_POINTS automatic
    2 2 2 1 1 1
"""


VMC = {
    "control": {"job_type": "vmc", "number_of_walkers": WALKERS, "verbosity": "high"},
    "vmc": {
        "num_opt_steps": steps(80, 3),
        "num_mcmc_steps": steps(500, 100),
        "num_mcmc_per_measurement": 40,
        "num_mcmc_warmup_steps": 30,
        "num_mcmc_bin_blocks": 10,
        "Dt": 1.5,
        "wf_dump_freq": 10,
        "opt_J1_param": False,
        "opt_J2_param": False,
        "opt_J3_param": False,
        "opt_JDTN_u_param": True,
        "opt_JDTN_chi_param": True,
        "opt_JDTN_f_param": True,
        "opt_geminal": False,
        "optimizer_kwargs": {"method": "sr", "delta": 0.15, "epsilon": 0.001, "use_lm": True},
    },
}
MCMC = {
    "control": {"job_type": "mcmc", "number_of_walkers": WALKERS, "verbosity": "high"},
    "mcmc": {
        "num_mcmc_steps": steps(20000, 200),
        "num_mcmc_per_measurement": 40,
        "num_mcmc_warmup_steps": 30,
        "num_mcmc_bin_blocks": 10,
        "Dt": 1.5,
    },
}


def lrdmc(alat: float) -> dict:
    return {
        "control": {"job_type": "lrdmc-tau", "number_of_walkers": WALKERS, "verbosity": "high"},
        "lrdmc-tau": {
            "num_mcmc_steps": steps(20000, 600),
            "tau": 0.10,
            "alat": alat,
            "num_gfmc_warmup_steps": 50,
            "num_gfmc_bin_blocks": 50,
            "num_gfmc_collect_steps": 20,
        },
    }


@calcfunction
def fit_eos(volumes: orm.List, energies: orm.List, errors: orm.List) -> orm.Dict:
    """Birch-Murnaghan E(V) with B0' = 4 fixed: V0 and B0 with bootstrap errors (volumes in bohr^3, E in Ha)."""
    from scipy.optimize import curve_fit

    def bm(v, e0, v0, b0, b0p=4.0):
        x = (v0 / v) ** (2.0 / 3.0)
        return e0 + 9.0 * v0 * b0 / 16.0 * ((x - 1.0) ** 3 * b0p + (x - 1.0) ** 2 * (6.0 - 4.0 * x))

    v, e, s = np.asarray(volumes.get_list()), np.asarray(energies.get_list()), np.asarray(errors.get_list())
    popt, _ = curve_fit(bm, v, e, p0=[e.min(), v[np.argmin(e)], 0.003], sigma=s, absolute_sigma=True, maxfev=20000)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(500):
        try:
            boot.append(curve_fit(bm, v, e + rng.normal(0.0, s), p0=popt, sigma=s, maxfev=20000)[0])
        except RuntimeError:
            continue
    boot = np.array(boot)
    a0 = (4 * popt[1]) ** (1 / 3) * 0.529177210903  # fcc primitive cell: V = a^3 / 4
    return orm.Dict(
        {
            "V0": popt[1],
            "V0_error": boot[:, 1].std(),
            "B0_GPa": popt[2] * 29421.015697,
            "B0_GPa_error": boot[:, 2].std() * 29421.015697,
            "a0_angstrom": a0,
            "E0": popt[0],
        }
    )


scan = {}
upf = file("jqmc-example07/01DFT/Si.ccECP.upf")
for a in LATTICE_CONSTANTS:
    _, pw = shell(
        f"a={a} pw.x",
        "pw",
        "-in {pwin}",
        nodes={"pwin": orm.SinglefileData.from_string(pw_input(a), filename="pw.in"), "upf": upf},
        kind="pw",
    )
    ham = tool(
        f"a={a} qe",
        "qe",
        {"--kind": "bspline", "--jastrow-dtn": "u_chi_f"},
        kind="convert",
        twist_average=orm.Bool(True),
        parent_folders={"pw": pw.outputs.remote_folder},
        save_path=orm.Str("si.save"),
    )
    vmc = jqmc(f"a={a} vmc", VMC, hamiltonian=ham.outputs.output_hamiltonian)
    trial = vmc.outputs.output_hamiltonian
    scan[a] = {
        "trial": trial,
        "mcmc": jqmc(f"a={a} mcmc", MCMC, hamiltonian=trial),
        "lrdmc": jqmc(f"a={a} lrdmc", lrdmc(0.30), hamiltonian=trial),
    }

volumes = [(a / 0.529177210903) ** 3 / 4 for a in LATTICE_CONSTANTS]
eos = {}
for method in ("mcmc", "lrdmc"):
    ta = [scan[a][method].outputs.output_parameters["twist_average"] for a in LATTICE_CONSTANTS]
    eos[method] = fit_eos(orm.List(volumes), orm.List([t["energy"] for t in ta]), orm.List([t["energy_error"] for t in ta]))
    print(f"    EOS {method}:", eos[method].get_dict())

a_star = min(LATTICE_CONSTANTS, key=lambda a: abs(a - eos["lrdmc"]["a0_angstrom"]))
runs = {"a030": scan[a_star]["lrdmc"]}
for alat in EXT_ALATS:
    runs[f"a{round(alat * 100):03d}"] = jqmc(f"a={a_star} lrdmc alat={alat}", lrdmc(alat), hamiltonian=scan[a_star]["trial"])
show(
    f"extrapolation at a={a_star}",
    tool(
        "extrapolate",
        "lrdmc extrapolate-energy",
        {"-p": 1, "-b": 50, "-w": 50, "-c": 20},
        parent_folders={k: n.outputs.remote_folder for k, n in runs.items()},
    ),
)
