# Orbital-tiling source fixtures

These are unchanged copies of small converter fixtures of jQMC's test suite.

- `qe/` is jQMC's `tests/qe_example_files/Li_tiling_sectors/`. Its pseudopotential `Li.ccECP.upf` is not kept
  in git; the tests link it into a mirror of this tree (`../pseudopotentials.toml`).
- `trexio/lithium_tiling_sectors.h5` is the spherical PySCF/TREXIO source.
- `trexio/lithium_tiling_sectors_cart.h5` is its Cartesian export.

The sources contain two complete `2 1 1` sectors: Gamma and the general
supertwist `(0.22, 0.13, 0.09)`, with source weights `0.2` and `0.8`. They are
conversion fixtures rather than converged bulk-lithium calculations.

The files are intentionally copied instead of generated in the aiida-jqmc
test suite. This makes a failed AiiDA/direct-CLI comparison diagnose transport
and argument handling rather than a changing PySCF or Quantum ESPRESSO run.

`hamiltonian/` holds two tiled Hamiltonians converted from `trexio/lithium_tiling_sectors_cart.h5` with
`jqmc-tool trexio convert-to <src> --tiling "2 1 1" --supertwist "0 0 0"` (`lithium_tiled_211_gamma.h5`)
and `--tiling "2 1 1" --twist-average` (`lithium_tiled_211_twist_average.h5`). They check
`is_tiled_hamiltonian`; the twist-averaged one is also the combined twist input of the CalcJob and parser tests.
