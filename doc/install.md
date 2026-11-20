(install_tags)=

# Installation and setup

## Requirements

- Python 3.11 or later (as jqmc)
- [aiida-core](https://aiida.readthedocs.io) 2.8 or later, with a configured
  profile (PostgreSQL or SQLite storage, and a message broker for `submit`)
- [jqmc](https://pypi.org/project/jqmc/), installed by pip with the plugin (the
  supported versions are given by the `jqmc` dependency in `pyproject.toml`).
  It is needed on the machine that runs AiiDA, because the parser uses
  `jqmc.output_parser` and the restart logic reads jqmc's restart policy, and
  on every computer that runs the calculations, which provides the `jqmc` and
  `jqmc-tool` executables.

## Install the plugin

From PyPI:

```bash
pip install aiida-jqmc
verdi plugin list aiida.calculations   # jqmc.jqmc, jqmc.tool
```

From a checkout:

```bash
git clone https://github.com/jqmc-project/aiida-jQMC.git
cd aiida-jQMC
pip install -e .            # or: pip install -e ".[tests,docs]"
```

Entry points:

| group | name | class |
|---|---|---|
| `aiida.calculations` | `jqmc.jqmc` | {class}`~aiida_jqmc.calculations.jqmc.JqmcCalculation` |
| `aiida.calculations` | `jqmc.tool` | {class}`~aiida_jqmc.calculations.tool.JqmcToolCalculation` |
| `aiida.parsers` | `jqmc.jqmc` | {class}`~aiida_jqmc.parsers.jqmc.JqmcParser` |
| `aiida.parsers` | `jqmc.tool` | {class}`~aiida_jqmc.parsers.tool.JqmcToolParser` |
| `aiida.workflows` | `jqmc.restart` | {class}`~aiida_jqmc.workflows.restart.JqmcRestartWorkChain` |

The plugin's version comes from the git tags (`setuptools_scm`, as for jQMC):
`aiida_jqmc.__version__`, which AiiDA records on every node as `version.plugin`, and
`aiida_jqmc.__git_commit__`. An editable install follows the checkout; a wheel reports the
commit it was built from.

## Set up a computer and the codes

### Local machine

`verdi presto` creates a profile together with the `localhost` computer; in that
case skip the `verdi computer` commands. Otherwise, set up the computer:

```bash
verdi computer setup -L localhost -H localhost -T core.local -S core.direct \
    -w /path/to/aiida_work --mpirun-command "mpirun -np {tot_num_mpiprocs}"
verdi computer configure core.local localhost --safe-interval 0
```

Then create the `jqmc` and `jqmc-tool` codes:

```bash
verdi code create core.code.installed -L jqmc -Y localhost \
    --filepath-executable "$(which jqmc)" --default-calc-job-plugin jqmc.jqmc
verdi code create core.code.installed -L jqmc-tool -Y localhost \
    --filepath-executable "$(which jqmc-tool)" --default-calc-job-plugin jqmc.tool
```

### A cluster

The plugin ships no scheduler plugin. Use one of aiida-core's (`core.slurm`, `core.pbspro`,
`core.sge`, `core.lsf`, ...) or a site-specific scheduler plugin installed separately.
Site-specific settings (queues, accounts, MPI launch options, modules) belong to the computer
and code setup, not to this plugin.

The MPI command must start the number of processes per node the job asks for. The code's
prepend text puts the Python environment that provides `jqmc` on `PATH` and sets the thread
counts. Environment variables set there are not seen by the MPI ranks on other nodes with every
MPI library; with Open MPI, forward them with `OMPI_MCA_mca_base_env_list`:

```bash
cat > prepend.sh <<'EOF'
export JQMC_FLAG_RUST_ON_CPU=1 JQMC_RS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMPI_MCA_mca_base_env_list="PATH;LD_LIBRARY_PATH;JQMC_FLAG_RUST_ON_CPU;JQMC_RS_NUM_THREADS;RAYON_NUM_THREADS;OMP_NUM_THREADS;OPENBLAS_NUM_THREADS;MKL_NUM_THREADS;XLA_FLAGS"
export PATH=/path/to/<jqmc-env>/bin:$PATH
EOF
verdi code create core.code.installed -L jqmc -Y <computer> \
    --filepath-executable /path/to/<jqmc-env>/bin/jqmc \
    --default-calc-job-plugin jqmc.jqmc --prepend-text "$(cat prepend.sh)"
```

Create the `jqmc-tool` code the same way (`--filepath-executable /path/to/<jqmc-env>/bin/jqmc-tool
--default-calc-job-plugin jqmc.tool`) on every computer whose checkpoints are post-processed:
the post-processing runs where the checkpoints are.

A `JqmcCalculation` needs `max_wallclock_seconds` (jqmc's `max_time` is set from it). The
queue and the account are the scheduler's usual options:

```python
options = {
    "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 4},  # adapt to the computer
    "queue_name": "<queue>",
    "account": "<account>",
    "max_wallclock_seconds": 3600,
}
```

A GPU is requested as the scheduler plugin expects it, for example with
`"custom_scheduler_commands": "#SBATCH --gpus=1"` for `core.slurm`; the
`resources` of aiida-core's schedulers accept no GPU count.

Next: {doc}`quickstart`.
