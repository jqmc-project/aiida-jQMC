# Changelog

## v0.1.0a1 (alpha)

The first alpha of aiida-jqmc. It requires Python 3.11 or later,
aiida-core 2.8 or later and jqmc (the supported versions are given by the
`jqmc` dependency in `pyproject.toml`).

### Calculations and work chains

- `JqmcCalculation` (`jqmc.jqmc`): one run of `jqmc` (`mcmc`, `vmc`,
  `lrdmc-bra`, `lrdmc-tau`), new or restarted from a checkpoint. jqmc's
  `max_time` is set from the required `max_wallclock_seconds`, so that jqmc
  stops with a checkpoint before the wall time. Twist-averaged inputs run with
  their per-twist checkpoint directory, recognized by jqmc's own check of the
  combined layout. Every VMC dump is stored on request (`all_hamiltonians`).
  `control.verbosity` defaults to `"high"`, so that every diagnostic the
  parser knows is in `output_parameters`.
- `JqmcRestartWorkChain` (`jqmc.restart`): one calculation carried to its
  target across as many jobs as needed, continuing from the checkpoint and
  leaving the keys that jqmc restores from it out of the continuations.
  Interrupted jobs are resubmitted; a job that ends with neither an exception
  nor a scheduler verdict is retried once (then `ERROR_REPEATED_CRASH`). A
  parent checkpoint must come from a successful `JqmcCalculation`. The merged
  `output_parameters` of a VMC run split into several jobs list the dumped
  Hamiltonians and warnings of every job.
- `JqmcToolCalculation` (`jqmc.tool`): `jqmc-tool` conversions (TREXIO, QE
  from a remote or stored folder, `conv-wf`), `show-info`, `to-xyz`,
  energies, forces, the LRDMC extrapolation, `analyze-output` and
  `average-wf`, with saved graphs. A file written by a run that reported an
  exception, or a tiled Hamiltonian that jQMC cannot read back as tiled, is not
  taken as its result.

### Orbital tiling

- `tiling`, `supertwist` and `twist_average` inputs of the QE and TREXIO
  conversions: Gamma, single-supertwist and weighted-ensemble supercells from
  primitive-cell orbitals. The physical supercell structure
  (`physical_structure`) and the conversion record (`conversion_metadata`)
  are outputs; a tiled VMC rejects Slater optimization flags before submission.

### Results for a caller

- `output_parameters` structures every number of the jqmc log (at the default
  `verbosity = "high"`, every VMC optimizer diagnostic), the input as applied,
  the environment and the system summary, described by closed JSON Schemas in
  `aiida_jqmc.schemas`. It records `parser_jqmc`, the jqmc that parsed the
  log, and warns when it is not the jqmc that ran.
- A run that printed "Program ends" and its checkpoint is a success even when
  the scheduler reported a failure afterwards. NaN or infinite values are
  stored as null and listed in `non_finite_values`; a NaN energy or variance
  gives `ERROR_NON_FINITE_RESULT`.
- `aiida_jqmc.statistics`: additional steps for a target error bar, VMC
  samples per step, VMC signal-to-noise and energy slope.
- `aiida_jqmc.utils.parameters_from_toml`: jqmc CLI TOML to `parameters`.

### Packaging

- The version comes from the git tags (`setuptools_scm`);
  `aiida_jqmc.__version__` (AiiDA's `version.plugin`) and
  `aiida_jqmc.__git_commit__` follow an editable checkout, and a wheel reports
  the commit it was built from.
- Releases on PyPI (`pip install aiida-jqmc`) and the documentation on GitHub
  Pages.

### Examples

- The jQMC command-line and `jqmc_workflow` tutorials through AiiDA
  (`examples/`).
