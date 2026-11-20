# aiida-jqmc

**aiida-jqmc** is an [AiiDA](https://www.aiida.net) plugin for
[jQMC](https://github.com/jqmc-project/jQMC), the real-space *ab initio*
quantum Monte Carlo code (variational Monte Carlo and lattice-regularized
diffusion Monte Carlo). It runs `jqmc` and `jqmc-tool` as AiiDA processes and
records the full provenance. The documentation is available from
[GitHub Pages](https://jqmc-project.github.io/aiida-jQMC/).

![license](https://img.shields.io/github/license/jqmc-project/aiida-jQMC)
![tag](https://img.shields.io/github/v/tag/jqmc-project/aiida-jQMC)
![fork](https://img.shields.io/github/forks/jqmc-project/aiida-jQMC?style=social)
![stars](https://img.shields.io/github/stars/jqmc-project/aiida-jQMC?style=social)
![Public distribution check](https://github.com/jqmc-project/aiida-jQMC/actions/workflows/public-distribution-check.yml/badge.svg)
![Lint](https://github.com/jqmc-project/aiida-jQMC/actions/workflows/public-lint-ruff.yml/badge.svg)
[![Docs](https://github.com/jqmc-project/aiida-jQMC/actions/workflows/public-deploy-gh-pages.yml/badge.svg)](https://jqmc-project.github.io/aiida-jQMC/)
[![Full Python coverage](https://raw.githubusercontent.com/jqmc-project/aiida-jQMC/ci-metadata/coverage.svg)](https://github.com/jqmc-project/aiida-jQMC/blob/ci-metadata/coverage-latest.json)
![DL](https://img.shields.io/pypi/dm/aiida-jqmc)
![python_version](https://img.shields.io/pypi/pyversions/aiida-jqmc)
[![aiida_version](https://img.shields.io/badge/AiiDA-%E2%89%A52.8%2C%20%3C3-blue)](https://www.aiida.net)
![pypi_version](https://badge.fury.io/py/aiida-jqmc.svg)

Full Python coverage measures `aiida_jqmc` over all tests, including the ones
that run jqmc, in the latest forge snapshot promoted to this repository. Click
the badge for the tested commit and update date. The badge becomes available
after the first promotion; until then no coverage value is published.

## Features

- **Runs** `jqmc` (`mcmc`, `vmc`, `lrdmc-bra`, `lrdmc-tau`) and `jqmc-tool`
  (TREXIO/QE conversions, energies, forces, extrapolation) as AiiDA calculations.
- **Converts** QE and PySCF/TREXIO primitive sources into jQMC orbital-tiled
  Gamma, single-supertwist and weighted-ensemble supercells, retaining the
  physical structure and sector metadata in provenance.
- **Splits** long runs over scheduler jobs from checkpoints, resubmits
  interrupted jobs, and extends finished calculations; twist-averaged runs included.
- **Reports** every number jqmc prints as structured `output_parameters`,
  described by JSON Schemas, so that an AI agent or a script can take the
  scientific decisions (run lengths, error bars, convergence) itself.

The plugin takes no scientific decision: every physical or statistical
parameter is passed to jqmc as given.

## Requirements

- Python 3.11 or later
- [aiida-core](https://aiida.readthedocs.io) 2.8 or later, with a configured profile
- [jqmc](https://pypi.org/project/jqmc/), installed by pip with the plugin (the
  supported versions are given by the `jqmc` dependency in `pyproject.toml`),
  both where AiiDA runs and on every computer that runs the calculations

## Install

```bash
pip install aiida-jqmc
verdi plugin list aiida.calculations   # jqmc.jqmc, jqmc.tool
```

The latest version can be installed from a clone of this repository:

```bash
git clone https://github.com/jqmc-project/aiida-jQMC.git
cd aiida-jQMC
pip install -e .
```

## Quick start

```bash
verdi presto        # a profile and the `localhost` computer
verdi code create core.code.installed -L jqmc -Y localhost \
    --filepath-executable "$(which jqmc)" --default-calc-job-plugin jqmc.jqmc
verdi code create core.code.installed -L jqmc-tool -Y localhost \
    --filepath-executable "$(which jqmc-tool)" --default-calc-job-plugin jqmc.tool
```

Then convert a TREXIO file into a jQMC Hamiltonian, run an MCMC calculation
and read its energy, as the
[quickstart](https://jqmc-project.github.io/aiida-jQMC/quickstart.html) shows.

## Documentation

<https://jqmc-project.github.io/aiida-jQMC/>: getting started (overview,
installation, quickstart), the user guide (submit, read, decide, extend;
examples), the reference (calculations, `jqmc-tool`, the work chain, the output
schema, the API), development and the changelog.

## Examples

[`examples/`](https://github.com/jqmc-project/aiida-jQMC/blob/main/examples/README.md) runs the jQMC tutorials (`examples/` of
[jqmc-project/jQMC](https://github.com/jqmc-project/jQMC/tree/main/examples),
matching your installed jqmc version) through AiiDA: every step, mean-field
calculations included, is an AiiDA process.

## Development and contributing

Active development of aiida-jqmc, including the addition of new features,
takes place in a private repository. External code contributions are not
merged directly through pull requests in this public repository. We
nevertheless welcome and appreciate pull requests as proposals: a reproducer
or suggested fix can help us understand a problem more clearly.

Such pull requests are not merged as submitted. Instead, the developers
investigate the problem, implement and test the changes in the private
repository, and then integrate the reviewed and validated changes into this
public repository.

If you would like to take part in development, please contact the developers
directly. Bug reports, feature requests, and suggestions are also welcome as
[GitHub Issues](https://github.com/jqmc-project/aiida-jQMC/issues) on this
public repository.

The CI of this public repository checks the lint, the distributions and the
documentation; it runs no test. The complete tests, which run jqmc, run in the
private development repository `aiida-jQMC-forge` and pass there for the
commit that is published here. Running the tests,
building the documentation and the branches are described in
[Development](https://jqmc-project.github.io/aiida-jQMC/development.html).

## Citation

If you used aiida-jqmc in your research project, please cite jQMC:

- "jQMC: A JAX-based ab initio quantum Monte Carlo package designed for GPU-accelerated computing",

  [K. Nakano and M. Casula, arXiv:2607.13781 (2026)](https://arxiv.org/abs/2607.13781)

  ```
  @article{nakano-jqmc,
      author  = {Nakano, Kousuke and Casula, Michele},
      title   = {jQMC: A JAX-based ab initio quantum Monte Carlo package designed for GPU-accelerated computing},
      journal = {arXiv preprint},
      eprint  = {2607.13781},
      archivePrefix = {arXiv},
      primaryClass  = {physics.chem-ph},
      year    = {2026},
      url     = {https://arxiv.org/abs/2607.13781}
  }
  ```

- "Load-Balanced Diffusion Monte Carlo Method with Lattice Regularization",

  [K. Nakano, S. Sorella, and M. Casula, J. Chem. Phys. 163, 194117 (2025)](https://doi.org/10.1063/5.0296986)

  ```
  @article{10.1063/5.0296986,
      author = {Nakano, Kousuke and Sorella, Sandro and Casula, Michele},
      title = {Load-balanced diffusion Monte Carlo method with lattice regularization},
      journal = {J. Chem. Phys.},
      volume = {163},
      number = {19},
      pages = {194117},
      year = {2025},
      month = {11},
      doi = {10.1063/5.0296986}
  }
  ```

and AiiDA:

- S. P. Huber *et al.*, "AiiDA 1.0, a scalable computational infrastructure for
  automated reproducible workflows and data provenance",
  [Sci. Data 7, 300 (2020)](https://doi.org/10.1038/s41597-020-00638-4)
- M. Uhrin *et al.*, "Workflows in AiiDA: Engineering a high-throughput,
  event-based engine for robust and modular computational workflows",
  [Comput. Mater. Sci. 187, 110086 (2021)](https://doi.org/10.1016/j.commatsci.2020.110086)

## Developer(s)

Kosuke Nakano (National Institute for Materials Science (NIMS), Japan)

## License

BSD-3-Clause.
