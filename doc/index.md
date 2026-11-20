# Welcome to aiida-jqmc

**aiida-jqmc** is an [AiiDA](https://www.aiida.net) plugin for
[jQMC](https://github.com/jqmc-project/jQMC), the real-space *ab initio*
quantum Monte Carlo code. It runs the `jqmc` command-line program (job types
`mcmc`, `vmc`, `lrdmc-bra` and `lrdmc-tau`) and `jqmc-tool` as AiiDA
calculations, splits long runs over several scheduler jobs, and records the
full provenance.

The plugin provides:

- {ref}`JqmcCalculation <calculations_tags>`: one run of `jqmc`, a new run or a restart from a checkpoint.
- {ref}`JqmcToolCalculation <tools_tags>`: one run of `jqmc-tool`: TREXIO and Quantum ESPRESSO conversions, energies, forces and the LRDMC extrapolation from checkpoints.
- {ref}`JqmcRestartWorkChain <workflows_tags>`: one jqmc calculation carried to completion across as many jobs as needed.

For a first run, follow {doc}`install`, then {doc}`quickstart`.
{doc}`overview` explains what the plugin decides and what it deliberately
leaves to the caller, and {doc}`agents` shows the loop a caller (an AI agent
or a script) runs: submit, read the structured results, decide, extend.
{doc}`examples` runs the jQMC tutorials through AiiDA.

```{toctree}
:caption: Getting started
:maxdepth: 1

overview
install
quickstart
```

```{toctree}
:caption: User guide
:maxdepth: 1

agents
examples
```

```{toctree}
:caption: Reference
:maxdepth: 1

calculations
tools
workflows
output_schema
api_reference
```

```{toctree}
:caption: Project
:maxdepth: 1

development
changelog
```

## License

BSD-3-Clause license.

## Developer(s)

Kosuke Nakano, National Institute for Materials Science (NIMS), Japan
