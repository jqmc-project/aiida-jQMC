(agents_tags)=

# Using aiida-jqmc: submit, read, decide, extend

aiida-jqmc is designed to be driven by a caller that takes the scientific
decisions itself: an AI agent, a script or a person at the AiiDA shell. The
plugin runs what it is given, records everything, and reports every number
jqmc printed; it never chooses a run length, a time step or an error bar, and
never declares a calculation converged. This page shows the loop such a caller,
typically an AI agent, runs. It continues the {doc}`quickstart`.

## 1. Start from a known input

The CLI inputs of jQMC (TOML) translate directly:

```python
from aiida import load_profile, orm
from aiida.engine import submit
from aiida.plugins import WorkflowFactory

from aiida_jqmc.utils import parameters_from_toml

load_profile()
parameters, removed = parameters_from_toml("vmc.toml")  # drops restart, restart_chk, hamiltonian_h5, max_time
# control.verbosity defaults to "high": all optimizer diagnostics in output_parameters

builder = WorkflowFactory("jqmc.restart").get_builder()
builder.jqmc.code = orm.load_code("jqmc@<computer>")
builder.jqmc.parameters = orm.Dict(parameters)
builder.jqmc.hamiltonian = orm.SinglefileData("/path/to/hamiltonian_data.h5")
builder.jqmc.metadata.options = {
    "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 4},  # adapt to the computer
    "queue_name": "<queue>",
    "account": "<account>",
    "max_wallclock_seconds": 3600,
}
node = submit(builder)
```

## 2. Read the results, not the log

`output_parameters` holds every number of the log, structured; its fields,
types and units are described by a JSON Schema the caller can read:

```python
from aiida_jqmc.schemas import output_parameters_schema

schema = output_parameters_schema("vmc")  # a dict (JSON Schema, draft 2020-12)
out = node.outputs.output_parameters.get_dict()
last = out["steps"][-1]
last["energy"], last["energy_error"], last["max_force"], last["signal_to_noise_ratio"]
last["diagnostics"]["sr_solve"]  # {regime, method, iterations, residual, tolerance, converged}
last["top_coordinates"]  # the coordinates with the largest force signal-to-noise
out["input_parameters"]  # the input as jqmc applied it: explicit, defaults, restored
```

If `out["unparsed_numeric_lines"]` is not empty, jqmc printed numbers the
parser does not structure yet; they are listed verbatim so nothing is lost.
The full field list is in {doc}`output_schema`.

## 3. Decide

{mod}`aiida_jqmc.statistics` computes the numbers that `jqmc_workflow` uses for
its automatic decisions; whether to act on them is the caller's choice:

```python
from aiida_jqmc.statistics import steps_to_target_error, vmc_indicators, vmc_sampling_steps_for_target_error

vmc_indicators([vmc_run, vmc_continuation], window=10)
# {'signal_to_noise_mean': ..., 'energy_slope': ..., 'energy_slope_error': ..., ...}
vmc_sampling_steps_for_target_error(vmc_pilot, target_error=1e-3)  # num_mcmc_steps per optimization step
steps_to_target_error(mcmc_run, target_error=1e-4)  # additional num_mcmc_steps
```

The estimates assume `error ~ 1/sqrt(N)` with the binning already longer than
the autocorrelation time; check that the error bar is stable against the
binning (`jqmc-tool ... compute-energy -b`) before relying on them.

## 4. Act: extend, restart or start over

Extend a finished calculation from its last checkpoint (VMC: more
optimization steps; MCMC/LRDMC: more measurements). Leave out the keys jqmc
restores from the checkpoint:

```python
last = [c for c in node.called if c.process_label == "JqmcCalculation"][-1]
builder = WorkflowFactory("jqmc.restart").get_builder()
builder.jqmc.code = node.inputs.jqmc.code
builder.jqmc.parent_folder = node.outputs.remote_folder
builder.jqmc.parameters = orm.Dict({"control": {"job_type": "mcmc"}, "mcmc": {"num_mcmc_steps": 40000}})
options = last.get_options()  # the same layout of MPI processes: the checkpoint is stored per rank
builder.jqmc.metadata.options = {
    k: options[k] for k in ("resources", "queue_name", "account", "max_wallclock_seconds") if k in options
}
submit(builder)
```

To use the optimized wavefunction downstream, pass `output_hamiltonian` (the
last dump) as the `hamiltonian` of the next calculation; with
`metadata.options.all_hamiltonians = True` every dump is stored as
`output_hamiltonians.step_<N>`, so an earlier step can be chosen instead.

## 5. Handle failures by their exit code

Each failure has its own exit code (see {doc}`calculations` and
{doc}`workflows`); the message carries jqmc's own error text when there is
one. The work chain resubmits only interruptions (wall time, node failure);
everything else stops and is left to the caller: `ERROR_INVALID_INPUT` (303)
means the input must change, `ERROR_JQMC_EXCEPTION` (304) is a jqmc error to
report, `ERROR_TWIST_STOPPED_EARLY` (312) asks for a restart from its
`remote_folder` (the incomplete twists are rerun) or a longer wall time.

## 6. Ask the provenance

Everything above is in the AiiDA graph, so a caller can answer questions such
as "which Hamiltonian did this energy come from?" without bookkeeping:

```python
from aiida.orm import QueryBuilder, WorkflowNode

qb = QueryBuilder()
qb.append(WorkflowNode, filters={"attributes.process_label": "JqmcRestartWorkChain"}, tag="wc")
qb.append(
    orm.Dict,
    with_incoming="wc",
    edge_filters={"label": "output_parameters"},
    project=["attributes.energy", "attributes.energy_error"],
)
qb.all()
```
