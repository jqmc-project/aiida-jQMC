# Orbital tiling through AiiDA

`run_aiida.py` is the AiiDA counterpart of jQMC `jqmc-example14`. It converts
the multi-k Li TREXIO file of the jQMC checkout
(`tests/trexio_example_files/lithium_tiling_sectors.h5`) into a `2 x 1 x 1` weighted-sector
Hamiltonian, optimizes its physical-supercell J1/J2/J3 while keeping the
tiled Slater frozen, restarts the ensemble checkpoint directory, and passes
the complete optimized ensemble to fresh MCMC and LRDMC calculations.

The script prints the `conversion_metadata` and `physical_structure` outputs
and requires the production Rust CPU backend. Select a resource count divisible
by the two sectors in `config.toml`.

`JQMC_EXAMPLE14_SINGLE_SECTOR=1 python run_aiida.py` selects the general
`(0.22, 0.13, 0.09)` sector. This is useful for a serial local smoke test; the
default exercises the weighted ensemble and its checkpoint directory.
