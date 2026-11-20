(examples_tags)=

# Examples

The repository's
[`examples/`](https://github.com/jqmc-project/aiida-jQMC/tree/main/examples)
directory runs the jQMC tutorials (`examples/` of
[jqmc-project/jQMC](https://github.com/jqmc-project/jQMC/tree/main/examples),
matching your installed jqmc version) through AiiDA: each
`jqmc-exampleNN/run_aiida.py` follows the command-line tutorial of the same
name, and each `jqmc-workflow-exampleNN/run_aiida.py` does what the
`jqmc_workflow` tutorial of that name does, with aiida-jqmc only. The tutorial
inputs are read from a jQMC checkout of the same version; every step,
mean-field calculations included, is an AiiDA process. See
[`examples/README.md`](https://github.com/jqmc-project/aiida-jQMC/blob/main/examples/README.md)
for the setup (`config.toml`: codes, computers, scheduler options, run-length
scale):

```bash
pip install "aiida-jqmc[examples]"   # aiida-shell and scipy
cp examples/config.example.toml examples/config.toml   # then adapt it
python examples/jqmc-example05/run_aiida.py
```

| script | what it runs |
|---|---|
| `jqmc-example01` | water (ccECP): PySCF → TREXIO → JSD VMC → MCMC → lrdmc-tau at five lattice spacings → extrapolation (with graph) |
| `jqmc-example02` | water, J2 only: VMC and its continuation → MCMC energy and forces |
| `jqmc-example03` | water: Jastrow-only VMC, then geminal/MO VMC; MCMC of both trials |
| `jqmc-example04` | S22 water dimer: three systems × JSD/JAGP × VMC/MCMC/LRDMC; binding energy as a `calcfunction` |
| `jqmc-example05` | all-electron H2: JSD and JAGP, MCMC and LRDMC energies and forces |
| `jqmc-example06` | periodic ice Ih (PySCF pGTOs, complex twist): HF check, VMC, MCMC, LRDMC at three spacings, extrapolation |
| `jqmc-example07` | silicon from QE (B-spline + DTN): VMC, MCMC, LRDMC, mixed-precision MPC |
| `jqmc-example08` | periodic all-electron hydrogen from QE (pGTO+PW, DTN): VMC, MCMC and LRDMC energies and forces |
| `jqmc-example09` | eight-twist silicon: twist-averaged MCMC/LRDMC with forces, stress and MPC, post-processing of the twist directory, restart, extrapolation |
| `jqmc-example10` | open shells: N atom (PySCF UHF) and Si atom (QE LSDA, B-spline) |
| `jqmc-example14` | orbital-tiled Li from multi-k TREXIO: weighted-sector conversion, physical-supercell J3 VMC, ensemble restart, MCMC and LRDMC |
| `jqmc-workflow-example01` | H2 potential-energy surface (7 bond lengths), forces against the PES derivative |
| `jqmc-workflow-example02` | GPU walker-scaling benchmark (needs a GPU node) |
| `jqmc-workflow-example03` | water VMC → MCMC/LRDMC energies and forces to target error bars |
| `jqmc-workflow-example04` | silicon equation of state (QE at five lattice constants, twist-averaged), Birch-Murnaghan fit as a `calcfunction`, LRDMC extrapolation |
| `jqmc-workflow-example05` | QE hydrogen (the workflow counterpart of example08), VMC in two production runs |
| `jqmc-workflow-example06` | twist-averaged VMC, the published multi-twist Hamiltonian handed to a twist-averaged MCMC |

Where `jqmc_workflow` decides by itself (pilot runs, target error bars, VMC
plateau), the scripts take the decision explicitly with
{mod}`aiida_jqmc.statistics`, as described in {doc}`agents`.

The CPU examples have been exercised with the default `scale` of 0.02 (the
workflow examples and jqmc-example14 use fixed short runs, see `examples/README.md`).
Configure MPI resources for the target computer. jqmc-workflow-example02 needs a
GPU node, requested as the scheduler expects it (e.g. `custom_scheduler_commands`
in `[options.gpu]` of `config.toml`).

Three jQMC examples have no AiiDA counterpart, because they do not run
the `jqmc` or `jqmc-tool` commands that aiida-jqmc wraps:

- `jqmc-example11` calls the orbital and Hamiltonian kernels from Python; it
  runs no job.
- `jqmc-example12` and `jqmc-example13` drive the same water calculation as
  `jqmc-example01` (VMC, then LRDMC at several spacings and the extrapolation)
  through the Python API, serial and with MPI; run `jqmc-example01` instead.
