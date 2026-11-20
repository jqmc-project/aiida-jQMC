(development_tags)=

# Development

Active development of aiida-jqmc takes place in a private repository. Pull
requests to the public repository
[jqmc-project/aiida-jQMC](https://github.com/jqmc-project/aiida-jQMC) are
welcome as proposals (a reproducer or a suggested fix), but they are not
merged as submitted: the developers implement and test the change and then
integrate it into the public repository. Bug reports, feature requests and
suggestions are welcome as
[GitHub Issues](https://github.com/jqmc-project/aiida-jQMC/issues).

The CI of this public repository checks the lint, the distributions and the
documentation; it runs no test. The complete tests, which run jqmc, run in the
private development repository `aiida-jQMC-forge` and pass there for the
commit that is published here.

## Run the tests

```bash
pip install -e ".[tests]"
pytest -m "not slow"    # plugin logic, on recorded logs
pytest                  # also the `slow` tests, which run jqmc with MPI
```

The slow tests need `jqmc`, `jqmc-tool` and `mpirun` on `PATH`; without them
they are skipped. `--fail-on-skip` makes any skipped test (and a module
skipped at collection) a failure, so that a complete run cannot pass with
tests left out.

The Quantum ESPRESSO fixtures of the slow tests read the pseudopotential
`Li.ccECP.upf`, which is not kept in the repository. `tests/data/pseudopotentials.toml`
gives its source and checksum; the tests download it once per host into
`$JQMC_TEST_DATA_CACHE` (default `~/.cache/jqmc/test-data`) and fail if it cannot
be provided. `python -m tests._test_data_cache` fills the cache in advance, for
a host that has no network access while the tests run.

A job that a work chain recovers from (for example one resubmitted after its
output was cut off) does not fail the test. The test warns about it instead,
with the end of the job's stdout, and `AIIDA_JQMC_KEEP_LOGS=<directory>` also
keeps the job's stdout and scheduler stderr there, one subdirectory per job
UUID.

## Build the documentation

The documentation of the latest release is on
[GitHub Pages](https://jqmc-project.github.io/aiida-jQMC/). To build it from a checkout:

```bash
pip install -e ".[docs]"
make -C doc html        # warnings are errors; output in doc/_build/html
```

The process pages are rendered from the process specs, which needs an AiiDA
profile; without one, `verdi presto` makes a SQLite profile. Without network
access, `AIIDA_JQMC_DOCS_OFFLINE=1 make -C doc html` builds without the
cross-references to the aiida-core and Python documentation.

## Branches

- `main`: the main branch.
- `rc`: the latest stable version ready for package deployment (TestPyPI on a push, PyPI on a `v*` tag).
- `rc-gh-pages`: the latest stable version ready for documentation deployment (GitHub Pages).
