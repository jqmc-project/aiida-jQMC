# Examples: the jQMC tutorials through AiiDA

Each `jqmc-exampleNN/run_aiida.py` runs the command-line tutorial of the same
name in the jQMC tutorials (`examples/` of
[jqmc-project/jQMC](https://github.com/jqmc-project/jQMC/tree/main/examples),
matching your installed jqmc version), and each
`jqmc-workflow-exampleNN/run_aiida.py` does what the `jqmc_workflow` tutorial
of that name does, with aiida-jqmc only (no `jqmc_workflow`). The tutorials'
TOML inputs, scripts and QE files are read from a jQMC checkout; every step is
an AiiDA process, so the whole chain is in the provenance graph:

| step | AiiDA process |
|---|---|
| PySCF, `pw.x` | aiida-shell `ShellJob` (localhost) |
| `jqmc-tool` conversions, `show-info` | `JqmcToolCalculation` |
| `jqmc` runs (split over jobs as needed) | `JqmcRestartWorkChain` |
| `jqmc-tool` post-processing of checkpoints | `JqmcToolCalculation` |
| binding energy, equation-of-state fit | `calcfunction` |

## Setup

```bash
pip install "aiida-jqmc[examples]"   # aiida-shell and scipy
git clone https://github.com/jqmc-project/aiida-jQMC.git   # these scripts: examples/
git clone https://github.com/jqmc-project/jQMC.git   # the tutorials; check out the tag of your jqmc version
cd aiida-jQMC/examples
cp config.example.toml config.toml   # codes, computers, scheduler options, the jQMC examples path
python jqmc-example05/run_aiida.py
python jqmc-example14/run_aiida.py   # orbital tiling
```

The examples need the jqmc that the plugin requires (the supported versions
are given by the `jqmc` dependency in `pyproject.toml`), and the tutorial
inputs of the same jqmc version.

The Quantum ESPRESSO examples read two pseudopotential files from the jQMC
`examples/` tree, which does not ship them. Download them there once, as the
jQMC tutorials say:

```bash
cd /path/to/jQMC/examples
curl -L -o jqmc-example07/01DFT/Si.ccECP.upf https://pseudopotentiallibrary.org/recipes/Si/ccECP/Si.ccECP.upf
curl -L -o jqmc-example08/01DFT/H.coulomb-ae.UPF https://pseudopotentials.quantum-espresso.org/upf_files/H.coulomb-ae.UPF
cp jqmc-example08/01DFT/H.coulomb-ae.UPF jqmc-workflow-example05/qe-pbc/
```

`Si.ccECP.upf` is used by jqmc-example07, 09 and 10 and by jqmc-workflow-example04
and 06; `H.coulomb-ae.UPF` by jqmc-example08 and jqmc-workflow-example05.

`scale` (or `JQMC_EXAMPLES_SCALE`) shortens the runs of jqmc-example01 to
jqmc-example10 and of jqmc-workflow-example04 to that fraction of the
tutorial's length (default 0.02 for a smoke test; wall time depends on the
configured resources); `1` runs the published sizes. The other workflow
examples and jqmc-example14 use fixed short run lengths, written in each
script; for the workflow examples `scale` only loosens the target error bars
(`error / sqrt(scale)`).

The scripts pass the Hamiltonian of the last VMC step (`output_hamiltonian`)
to the sampling steps. The tutorials pass the average of the last dumps
(`jqmc-tool vmc average-wf`); `JqmcToolCalculation` runs that command on the
`output_hamiltonians` of a VMC run with `all_hamiltonians`.

The scripts and the tutorials they follow are listed in the
[documentation](https://jqmc-project.github.io/aiida-jQMC/examples.html),
which also says which jQMC examples have no script here.

## What jqmc_workflow decides, and who decides it here

`jqmc_workflow` picks run lengths from pilot runs and target error bars,
continues runs, and stops VMC on its own criteria. aiida-jqmc leaves those
decisions to the caller: `aiida_jqmc.statistics` computes the numbers
(`steps_to_target_error`, `vmc_sampling_steps_for_target_error`,
`vmc_indicators`), and the scripts take the decisions explicitly
(`_runner.sample_to_target`), as an agent would.

## Notes

- `pw.x` exits with status 0 also when the SCF did not converge; a `ShellJob`
  cannot see that (aiida-quantumespresso's parser can). jqmc-example10 runs its
  LSDA SCF serially, as the tutorial's command does, with a softer mixing and
  more iterations than the tutorial input (`mixing_beta = 0.1`,
  `electron_maxstep = 600`): with QE 7.6 the tutorial input did not converge in
  200 iterations.
- Twist-averaged runs (jqmc-example09, jqmc-workflow-example04/06) need a wall time long
  enough for one pass over the twists: a twist stopped on `max_time` ends with
  `ERROR_TWIST_STOPPED_EARLY`.
- jqmc-workflow-example02 (GPU walker benchmark) needs a GPU node: request the
  GPUs as the computer's scheduler plugin expects them, e.g. with
  `custom_scheduler_commands` in `[options.gpu]`.
