# Logs recorded on workstations

Stdout of real `jqmc` runs, checked against the output schema in `tests/test_schemas.py`.

`vmc_mo_high.out`, `vmc_sr_high.out`, `vmc_lm_low.out` and `vmc_lm_high.out` are VMC optimizations of
`../H2_ecp_ccpvdz_jsd.h5` (copied to `hamiltonian_data.h5`), run with `mpirun -np 2 jqmc input.toml` and this input:

```toml
[control]
job_type = "vmc"
number_of_walkers = 4
mcmc_seed = 1
verbosity = "high"  # "low" for vmc_lm_low.out
max_time = 3600
restart = false
restart_chk = "restart.h5"
hamiltonian_h5 = "hamiltonian_data.h5"

[vmc]
num_mcmc_steps = 40
num_mcmc_warmup_steps = 10
num_mcmc_bin_blocks = 5
num_opt_steps = 2
```

- `vmc_mo_high.out`: MO optimization of the determinant (`opt_geminal = true` in `[vmc]`) with the default
  optimizer (aSR); its MO blocks print the S-Stiefel tangent.
- `vmc_sr_high.out`: plain SR, without aSR (`[vmc.optimizer_kwargs]` `method = "sr"`, `use_lm = false`,
  `delta = 0.01`); its derivative sources carry no local-energy derivative D.
- `vmc_lm_low.out` and `vmc_lm_high.out`: the full linear method (`[vmc.optimizer_kwargs]` `lm_subspace_dim = -1`);
  every verbosity prints the conditioning of the LM coordinates, high verbosity also the selected state, the
  subspace and the decision.

The other three logs are periodic runs with 2 walkers per MPI process:

- `mcmc_pbc_pgto_forces_stress_mpc_high.out`: MCMC with `atomic_force`, `compute_stress` and `mpc`, 2 MPI processes,
  high verbosity.
- `mcmc_twist_stopped_8_twists.out`: twist-averaged MCMC (8 twists, 8 MPI processes, high verbosity) with
  `max_time = 1`, so that it stops before its twists are complete.
- `vmc_twist_8_twists_low.out`: twist-averaged VMC (8 twists, one optimization step, low verbosity).

The host name, release and version of the `platform` line were replaced by `test-host`, `test-release` and
`test-version`.
