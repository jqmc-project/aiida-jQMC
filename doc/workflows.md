(workflows_tags)=

# JqmcRestartWorkChain

{class}`~aiida_jqmc.workflows.restart.JqmcRestartWorkChain` (entry point
`jqmc.restart`) carries **one** jqmc calculation to completion, across as many
scheduler jobs as needed. It is an AiiDA
`BaseRestartWorkChain` (`aiida.engine.BaseRestartWorkChain`)
around {doc}`JqmcCalculation <calculations>`, whose inputs it exposes in the
`jqmc` namespace.

## The target

`jqmc.parameters` describes the whole calculation exactly as for one
uninterrupted jqmc run. The number of steps of the job type,

| job type | steps | counter |
|---|---|---|
| `mcmc` | `num_mcmc_steps` | `mcmc_counter` |
| `vmc` | `num_opt_steps` | `opt_step_counter` |
| `lrdmc-bra`, `lrdmc-tau` | `num_mcmc_steps` | `gfmc_counter` |

is the number of steps to do in this work chain. The target counter is that
number, plus the counter of the parent checkpoint when `jqmc.parent_folder` is
given. The parent must then come from a successful `JqmcCalculation`, whose
`output_parameters` hold its counter (otherwise
`ERROR_PARENT_COUNTER_UNKNOWN` (410)).

```python
from aiida import orm
from aiida.engine import submit
from aiida.plugins import WorkflowFactory

JqmcRestartWorkChain = WorkflowFactory("jqmc.restart")

builder = JqmcRestartWorkChain.get_builder()
builder.jqmc.code = orm.load_code("jqmc@<computer>")
builder.jqmc.hamiltonian = orm.SinglefileData("/path/to/hamiltonian_data.h5")
builder.jqmc.parameters = orm.Dict(
    {
        "control": {"job_type": "mcmc", "number_of_walkers": 4, "mcmc_seed": 1},
        "mcmc": {"num_mcmc_steps": 600000, "num_mcmc_warmup_steps": 1000, "num_mcmc_bin_blocks": 10},
    }
)
builder.jqmc.metadata.options = {
    "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 4},  # adapt to the computer
    "queue_name": "<queue>",
    "account": "<account>",
    "max_wallclock_seconds": 3600,
}
builder.max_iterations = orm.Int(10)
node = submit(builder)
```

## What happens after each job

Each job runs with `max_time` set from its wall time (see
{doc}`calculations`). After it, the work chain decides:

**The job finished successfully.**
: If its counter reached the target, the work chain ends and exposes the
  outputs of that job; for a VMC run that took several jobs,
  `output_parameters` holds the `steps` of every job and `output_hamiltonians`
  the Hamiltonians of every job. If not, the next job continues from its
  checkpoint with the remaining steps (`target - counter`). A job that did not advance the
  counter ends the work chain with `ERROR_NO_PROGRESS` (411).

**The job was cut off before jqmc ended**
(`ERROR_OUTPUT_STDOUT_INCOMPLETE`, `ERROR_SCHEDULER_OUT_OF_WALLTIME`,
`ERROR_SCHEDULER_NODE_FAILURE`).
: The job is resubmitted from the last good checkpoint (or from the start if
  there is none yet), with `max_time_margin` doubled, but never more than half
  the wall time. Two such jobs in a row are resubmitted; a third ends the work
  chain with `ERROR_TOO_MANY_INTERRUPTED_JOBS` (412). A job that ended with neither an
  exception nor a scheduler verdict (`ERROR_OUTPUT_STDOUT_INCOMPLETE`, e.g. a
  segmentation fault or an MPI abort) is resubmitted only once: the same
  checkpoint and random state would crash again, so a second one in a row ends
  the work chain with `ERROR_REPEATED_CRASH` (413).

**Anything else** (invalid input, an exception in jqmc, NaN, no checkpoint, jqmc not started).
: The work chain stops and reports the exit code of the job. Resubmitting
  would not change the result; the decision is the caller's.

`max_iterations` (default 5) bounds the total number of jobs.

## Continuation jobs

Only the inputs that wire the restart change from job to job:

- `parent_folder` is the `remote_folder` of the job whose checkpoint is
  continued, and `hamiltonian` is dropped;
- the step count is the remaining number of steps;
- the keys that jqmc restores from the checkpoint ("frozen on restart", e.g.
  `mcmc_seed`, `number_of_walkers`, `Dt`, `alat`, `E_scf`) are left out. jqmc
  rejects a restart whose explicit value differs from the checkpoint, and some
  of them change during a run (`E_scf` of `lrdmc-bra`). The list comes from
  jqmc's own restart policy (`jqmc.jqmc_miscs.cli_parameters`).

Every other parameter is passed unchanged.

## Extending a finished calculation

The caller decides whether a result is good enough. To add steps, start a new
work chain from the `remote_folder` of the finished one, with the same code
and resources:

```python
builder = JqmcRestartWorkChain.get_builder()
builder.jqmc.code = orm.load_code("jqmc@<computer>")
builder.jqmc.parent_folder = node.outputs.remote_folder
builder.jqmc.parameters = orm.Dict(
    {
        "control": {"job_type": "mcmc"},
        "mcmc": {"num_mcmc_steps": 400000},  # 400 000 more steps
    }
)
builder.jqmc.metadata.options = {
    "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 4},  # the same MPI layout as before
    "queue_name": "<queue>",
    "account": "<account>",
    "max_wallclock_seconds": 3600,
}
submit(builder)
```

The same applies to VMC: run a few optimization steps, inspect
`output_parameters["steps"]`, and extend with more `num_opt_steps` until the
caller is satisfied.

## Specification

```{eval-rst}
.. aiida-workchain:: JqmcRestartWorkChain
    :module: aiida_jqmc.workflows.restart
```
