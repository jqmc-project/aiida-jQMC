(overview_tags)=

# Overview

## What the plugin decides, and what it does not

The plugin executes a calculation that the caller has fully specified. It
makes **execution** decisions only:

- how a run is cut into scheduler jobs (`max_time` against the wall time),
- when to continue from a checkpoint and when to resubmit an interrupted job,
- when to stop and hand the decision back to the caller.

Every **physical or statistical** parameter is taken verbatim from the
`parameters` input and passed unchanged to every job: the number of steps,
`Dt`, `tau`, `alat`, `epsilon_AS`, the number of walkers, the seed, and so on.
The plugin does not decide when a VMC optimization has converged, how small an
error bar must be, or whether an energy is physically sensible. Those
decisions belong to the caller, typically an AI agent or a person driving
AiiDA through its Python API or `verdi`, who inspects `output_parameters` and
then extends, restarts or abandons the calculation.

The parser follows the same rule. It reports what jqmc printed
(through `jqmc.output_parser`, which is part of jqmc) and flags only
unambiguous breakage: a missing or truncated log, an exception, a missing
checkpoint, a NaN or infinite energy.

The plugin does not use the `jqmc_workflow` subpackage of jQMC.

## One calculation, several jobs

A QMC run of many hours does not fit the wall time of one batch job. jqmc can
stop itself gracefully: given `control.max_time`, it finishes the current
block, writes its checkpoint (`restart.h5`) and exits normally. A restart from
that checkpoint continues the step counters (`mcmc_counter`, `gfmc_counter`,
`opt_step_counter`).

The plugin builds on this:

1. {class}`~aiida_jqmc.calculations.jqmc.JqmcCalculation` sets
   `max_time = max_wallclock_seconds - max_time_margin`, so jqmc stops before
   the scheduler kills it.
2. {class}`~aiida_jqmc.workflows.restart.JqmcRestartWorkChain` reads the
   counter of each finished job and, while the target is not reached, submits
   the next job from the latest checkpoint with the remaining number of steps.

The caller asks for, e.g., 600 000 MCMC steps once; whether that takes one job
or ten is an execution detail recorded in the provenance graph.

To *extend* a finished calculation (more steps, more VMC optimization steps),
the caller starts a new work chain with `jqmc.parent_folder` set to the
`remote_folder` of the last job, and the number of additional steps.

## Glossary

checkpoint
: `restart.h5`, written by jqmc at the end of every run. It stores one group
  per MPI rank, so a restart must use the same number of MPI processes and
  walkers.

frozen key
: An input key that jqmc restores from the checkpoint on restart (e.g.
  `mcmc_seed`, `number_of_walkers`, `Dt`, `alat`, `E_scf`). jqmc rejects a
  restart whose explicit value differs from the checkpoint, so continuation
  jobs leave these keys out.

counter
: The cumulative number of steps done in a chain of restarts, as printed by
  jqmc.
