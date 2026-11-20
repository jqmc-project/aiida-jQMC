(calculations_tags)=

# JqmcCalculation

{class}`~aiida_jqmc.calculations.jqmc.JqmcCalculation` (entry point
`jqmc.jqmc`) runs `jqmc aiida.toml` once, for one of the job types `mcmc`,
`vmc`, `lrdmc-bra` or `lrdmc-tau`.

## Parameters

`parameters` is the jqmc TOML input as a dict, with the same sections and keys
as described in the jQMC manual. The plugin checks only its structure:

- a `control` section with a `job_type` in `mcmc`, `vmc`, `lrdmc-bra`,
  `lrdmc-tau`, and the section of that job type;
- `control.verbosity` is `low`, `high` or `devel` (other levels write one log
  per MPI rank instead of the rank-0 stdout that the parser reads);
- none of the `control` keys that the plugin sets itself: `restart`,
  `restart_chk`, `hamiltonian_h5`, `max_time`.

All other values are written to the TOML file unchanged. jqmc validates them;
an input it rejects ends the calculation with `ERROR_INVALID_INPUT` (303) and
jqmc's message.

```python
from aiida import orm
from aiida.engine import submit
from aiida.plugins import CalculationFactory

JqmcCalculation = CalculationFactory("jqmc.jqmc")

builder = JqmcCalculation.get_builder()
builder.code = orm.load_code("jqmc@localhost")
builder.hamiltonian = orm.SinglefileData("/path/to/hamiltonian_data.h5")
builder.parameters = orm.Dict(
    {
        "control": {"job_type": "mcmc", "number_of_walkers": 4, "mcmc_seed": 1},
        "mcmc": {"num_mcmc_steps": 10000, "num_mcmc_warmup_steps": 1000, "num_mcmc_bin_blocks": 10},
    }
)
builder.metadata.options = {
    "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 2},
    "max_wallclock_seconds": 600,
}
node = submit(builder)
```

## New run and restart

Exactly one of the two inputs is given:

`hamiltonian`
: A new run. The file is copied to the working directory as
  `hamiltonian_data.h5`.

`parent_folder`
: A restart: the `remote_folder` of a previous `JqmcCalculation`. Its checkpoint
  `restart.h5` is symlinked (jqmc writes a new file, so the parent's checkpoint
  is never modified) and `control.restart` is set. The step counters in the
  checkpoint keep accumulating, so `num_mcmc_steps` or `num_opt_steps` are the
  *additional* steps.

A restart must keep what the checkpoint stores per MPI rank. The input
validator rejects, before anything is uploaded,

- a parent on a different computer,
- a different `job_type`,
- a different total number of MPI processes,
- an explicit `number_of_walkers` different from the one in the restart chain.

Keys that jqmc restores from the checkpoint ("frozen on restart") may be left
out of a restart input; if given, jqmc requires them to be unchanged.

## Twist-averaged runs

A Hamiltonian with the combined twist-averaged layout (`jqmc-tool qe --twist-average`:
HDF5 groups `shared` and `twists`) makes jqmc sample every twist in its own group of MPI ranks; the
number of MPI processes must be a multiple of the number of twists. The
plugin detects such an input with jqmc's own check and uses the checkpoint directory
`restart_twists` (one file per twist plus `shared_state.h5`). A restart copies
that directory instead of linking it, because jqmc writes into it, and copies
the Hamiltonian that started the chain, from which jqmc selects the twist route.

The results are in `output_parameters["twist_average"]` (E_TA, its error,
`s_twist`, MPC and the per-twist table).

jqmc applies `max_time` to each twist. A twist-averaged run that stops on
`max_time` ends with `ERROR_TWIST_STOPPED_EARLY`, and the work chain does not
continue it: its BZ average is over incomplete twists, which
`output_parameters["incomplete_twists"]` lists with the steps each one did. The caller either
restarts it from its `remote_folder`, which skips the complete twists and
reruns the others in full, or reruns it with a wall time long enough for one
pass over its twists. A job killed by the scheduler is resubmitted from the
input the work chain started with (the Hamiltonian, or the given
`parent_folder`): the twists that the killed job completed are run again.

## Tiled Slater and supercell Jastrow

For a Hamiltonian made with the converter's `tiling` input, the determinant
keeps frozen primitive-cell orbital blocks while every observable belongs to
the physical supercell. Before a VMC submission, the plugin reads the
Hamiltonian with jQMC and rejects a true value for any of
`opt_geminal`, `opt_lambda_basis_exp` or `opt_lambda_basis_coeff`. It does not
silently turn these flags off.

J2, DTN and J3 are ordinary phase-free correlation factors on the physical
supercell. The J3 basis is independent of the tiled Slater blocks: QE uses
UPF-derived supercell GTOs and TREXIO expands its GTO basis over the physical
cell. Therefore `opt_J3_param`, `opt_J3_basis_exp` and
`opt_J3_basis_coeff` retain their usual meaning and are not prohibited by
tiling. The existing J1/J2/J3 versus DTN exclusions still come from jQMC.

VMC writes the full optimized Hamiltonian. For an ensemble this includes all
sectors and the shared Jastrow state, so `output_hamiltonian` is the input for
a fresh MCMC or LRDMC calculation. Its combined layout continues to select
the ensemble path; AiiDA does not average the already weighted result again.
An ensemble checkpoint is always the `restart_twists` directory, including a
combined ensemble containing only one sector.

## Wall time and `max_time`

`max_wallclock_seconds` is required. The plugin sets

```text
control.max_time = max_wallclock_seconds - max_time_margin
```

so that jqmc stops gracefully, writes its checkpoint and exits before the
scheduler kills the job. The margin defaults to 10 % of the wall time, clamped
to 60–1800 s, and can be set with the `max_time_margin` option. It must cover
the start-up of jqmc (imports, JIT compilation) and the writing of the
checkpoint.

A calculation that stopped on `max_time` is **successful** (exit status 0):
`output_parameters` reports `stopped_by_max_time` and the counter reached.
Carrying on is the job of {doc}`JqmcRestartWorkChain <workflows>` or of the
caller.

## Plugin-specific options

Besides aiida-core's scheduler options (`resources`, `max_wallclock_seconds`,
`queue_name`, `account`, ...), `metadata.options` takes:

`max_time_margin`
: Seconds reserved between jqmc's `max_time` and the wall time (default: 10 %
  of the wall time, clamped to 60–1800 s; see
  [Wall time and `max_time`](#wall-time-and-max_time)).

`all_hamiltonians`
: VMC: also store every `hamiltonian_data_opt_step_*.h5` written, as
  `output_hamiltonians` (default `False`: only the last one, as
  `output_hamiltonian`).

`retrieve_checkpoint`
: Also retrieve the checkpoint into the repository (default `False`). It
  stays in the remote folder anyway. With atomic forces or stress, jqmc keeps
  their per-step history in the checkpoint: MCMC per walker (steps × walkers ×
  atoms, tens of GB for about 20 atoms, 64 walkers and 10^6 steps), LRDMC
  averaged over walkers (steps × atoms). Every restart, and so every job of a
  `JqmcRestartWorkChain`, writes a full new checkpoint into its own remote
  folder. Plan the remote disk space for long force runs, and retrieve such
  checkpoints only when needed.

`withmpi` defaults to `True`.

## Outputs

`output_parameters`
: Every number the jqmc stdout prints, structured by
  `jqmc.output_parser.parse_jqmc_log`: the results (energy, variance,
  MPC, forces, stress, twist averages), the counters, the sampling and timing
  diagnostics, every VMC optimization step, the input as applied (explicit,
  defaulted and restored from the checkpoint), the software and precision
  environment and the system summary, plus `exception` when jqmc raised one.
  Keys whose line was not printed are present with value `None`.
  `unparsed_numeric_lines` lists any printed number that is not structured;
  it is empty for the logs the tests cover. The fields, their types and units
  are in the {doc}`output_schema`.

  For VMC, `control.verbosity = "high"` prints the optimizer details of every
  step (aSR gammas, the SR solve, the coordinates with the largest
  signal-to-noise, per-block forces and updates), all of which are extracted;
  the default `"low"` gives the energy, the maximum force and the timings.

`output_hamiltonian`
: VMC only: the last `hamiltonian_data_opt_step_*.h5` written. It can be given
  directly as the `hamiltonian` of a following LRDMC calculation.

`output_hamiltonians`
: VMC with `all_hamiltonians = True`: every Hamiltonian written, one per
  dumped optimization step, as `output_hamiltonians.step_<N>`. The last one is
  also `output_hamiltonian`.

`remote_folder`
: The working directory, which holds the checkpoint. It is the
  `parent_folder` of a restart. The checkpoint is retrieved into the
  repository only with `retrieve_checkpoint = True`.

## Exit codes

The parser reports breakage that is unambiguous; it never judges whether a
result is physically sensible.

| status | label | meaning | worth resubmitting? |
|---|---|---|---|
| 301 | `ERROR_OUTPUT_STDOUT_MISSING` | the stdout was not retrieved | — |
| 302 | `ERROR_OUTPUT_STDOUT_INCOMPLETE` | jqmc did not end and raised nothing (killed, node failure) | yes |
| 303 | `ERROR_INVALID_INPUT` | jqmc rejected the input before sampling | no |
| 304 | `ERROR_JQMC_EXCEPTION` | jqmc raised an exception | no |
| 305 | `ERROR_NOT_STARTED` | jqmc wrote nothing (e.g. the MPI launch failed) | no |
| 310 | `ERROR_NO_CHECKPOINT` | jqmc ended without reporting its checkpoint | no |
| 311 | `ERROR_OUTPUT_HAMILTONIAN_MISSING` | VMC: the optimized Hamiltonian was not retrieved | no |
| 312 | `ERROR_TWIST_STOPPED_EARLY` | a twist-averaged run stopped on `max_time` (see [twist-averaged runs](#twist-averaged-runs)) | no |
| 313 | `ERROR_TWIST_RESULT_MISSING` | a twist-averaged run ended without its BZ-averaged result | no |
| 320 | `ERROR_NON_FINITE_RESULT` | the final energy or variance is NaN or infinite | no |

When the scheduler has already set a verdict, e.g.
`ERROR_SCHEDULER_OUT_OF_WALLTIME` (120) for a job killed at its wall time, and
jqmc did not end, the parser keeps that verdict.

## Specification

```{eval-rst}
.. aiida-calcjob:: JqmcCalculation
    :module: aiida_jqmc.calculations.jqmc
```
