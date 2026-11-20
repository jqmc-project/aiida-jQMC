(quickstart_tags)=

# Quickstart

This page runs a first calculation on the local machine: it converts a TREXIO
file of H{sub}`2` into a jQMC Hamiltonian with
{class}`~aiida_jqmc.calculations.tool.JqmcToolCalculation`, runs a short MCMC
calculation with {class}`~aiida_jqmc.workflows.restart.JqmcRestartWorkChain`
and reads its energy. It needs aiida-jqmc installed ({doc}`install`) and
`jqmc`, `jqmc-tool` and `mpirun` on `PATH`. The run lengths are a smoke test,
not a converged calculation.

## 1. A profile, the computer and the codes

```bash
verdi presto        # a SQLite profile and the `localhost` computer
verdi code create core.code.installed -L jqmc -Y localhost \
    --filepath-executable "$(which jqmc)" --default-calc-job-plugin jqmc.jqmc
verdi code create core.code.installed -L jqmc-tool -Y localhost \
    --filepath-executable "$(which jqmc-tool)" --default-calc-job-plugin jqmc.tool
```

`verdi presto` creates the `localhost` computer (transport `core.local`,
scheduler `core.direct`, MPI command `mpirun -np {tot_num_mpiprocs}`). With an
existing profile, set up the computer as in {doc}`install`.

## 2. A TREXIO file

The TREXIO file of H{sub}`2` (ccECP, cc-pVDZ, Cartesian GTOs) used by jQMC's
own tests:

```bash
wget https://raw.githubusercontent.com/jqmc-project/jQMC/main/tests/trexio_example_files/H2_ecp_ccpvdz_cart.h5
```

## 3. Convert it to a Hamiltonian

Save the following as `quickstart.py`. The steps run with
`aiida.engine.run_get_node`, which blocks until the process ends and needs no
daemon; `aiida.engine.submit` hands them to the daemon instead
(`verdi daemon start`).

```python
import pathlib

from aiida import load_profile, orm
from aiida.engine import run_get_node
from aiida.plugins import CalculationFactory, WorkflowFactory

load_profile()
JqmcToolCalculation = CalculationFactory("jqmc.tool")
JqmcRestartWorkChain = WorkflowFactory("jqmc.restart")

# jqmc-tool trexio convert-to H2_ecp_ccpvdz_cart.h5 -j2 1.0
_, convert = run_get_node(
    JqmcToolCalculation,
    code=orm.load_code("jqmc-tool@localhost"),
    command=orm.Str("trexio convert-to"),
    input_file=orm.SinglefileData(pathlib.Path("H2_ecp_ccpvdz_cart.h5").resolve()),  # an absolute path
    options=orm.Dict({"-j2": 1.0}),
)
print("conversion", convert.pk, convert.exit_status)
hamiltonian = convert.outputs.output_hamiltonian
```

`options` holds the `jqmc-tool` options as on the command line; here a J2
Jastrow factor with the initial parameter 1.0. The plugin sets the output file
itself and stores it as `output_hamiltonian`. See {doc}`tools`.

## 4. Run MCMC

Append to `quickstart.py`:

```python
builder = JqmcRestartWorkChain.get_builder()
builder.jqmc.code = orm.load_code("jqmc@localhost")
builder.jqmc.hamiltonian = hamiltonian
builder.jqmc.parameters = orm.Dict(
    {
        "control": {"job_type": "mcmc", "number_of_walkers": 4, "mcmc_seed": 1},
        "mcmc": {"num_mcmc_steps": 200, "num_mcmc_warmup_steps": 20, "num_mcmc_bin_blocks": 10},
    }
)
builder.jqmc.metadata.options = {
    "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 2},
    "max_wallclock_seconds": 600,
}
_, node = run_get_node(builder)
print("work chain", node.pk, node.exit_status)

out = node.outputs.output_parameters
print(out["energy"], out["energy_error"], out["mcmc_counter"])
```

`parameters` is the jqmc TOML input as a dict and is passed to jqmc
unchanged. `max_wallclock_seconds` is required: jqmc's `max_time` is set from
it, so that a run longer than the wall time stops with a checkpoint and the
work chain continues it in the next job. See {doc}`calculations` and
{doc}`workflows`.

Run the script:

```bash
python quickstart.py
```

## 5. Read the results

A process that finished well has exit status 0. Every other exit status has a
label and a message, listed in {doc}`calculations` (for example
`ERROR_INVALID_INPUT` (303) when jqmc rejected the input) and {doc}`workflows`:

```bash
verdi process list -a -p 1          # the processes of the last day, with their exit status
verdi process show <PK>             # inputs, outputs, called processes, exit status
verdi process report <PK>           # the log of the work chain
verdi calcjob outputcat <PK>        # the jqmc stdout of a JqmcCalculation
```

In Python, `node.exit_status` and `node.exit_message` give the same, and
`output_parameters` holds every number of the jqmc log:

```python
from aiida import load_profile, orm

load_profile()
node = orm.load_node(PK)  # the work chain
out = node.outputs.output_parameters.get_dict()
out["energy"], out["energy_error"]  # Ha
out["variance"], out["acceptance_ratio"], out["mcmc_counter"]
out["input_parameters"]  # the input as jqmc applied it
```

The fields, their types and units are listed in {doc}`output_schema`.

## Next

- {doc}`agents`: extend a calculation from its checkpoint, decide run lengths
  from the error bar, optimize a wavefunction with VMC and handle failures.
- {doc}`examples`: the jQMC tutorials through AiiDA.
