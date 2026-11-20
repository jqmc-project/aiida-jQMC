(tools_tags)=

# JqmcToolCalculation

{class}`~aiida_jqmc.calculations.tool.JqmcToolCalculation` (entry point
`jqmc.tool`) runs one `jqmc-tool` subcommand: a conversion that produces a
Hamiltonian, or the post-processing of checkpoints written by
{doc}`JqmcCalculation <calculations>`.

## Subcommands

| command | reads | produces |
|---|---|---|
| `trexio convert-to` | `input_file` (a TREXIO file) | `output_hamiltonian` |
| `trexio show-info`, `hamiltonian show-info` | `input_file` (a TREXIO or Hamiltonian file) | `output_parameters` (the system summary) |
| `hamiltonian to-xyz` | `input_file` (a Hamiltonian) | `output_structure` |
| `vmc analyze-output` | the jqmc stdout of each `parent_folders` entry, in key order | `output_graph` with `-s` |
| `vmc average-wf` | the Hamiltonian files of the optimization steps, one per `input_files` entry, in key order | `output_hamiltonian` (the average) |
| `qe` | a Quantum ESPRESSO output directory: one `parent_folders` entry or `input_folder`, and `save_path` | `output_hamiltonian` |
| `hamiltonian conv-wf` | `input_file` (a Hamiltonian) | `output_hamiltonian` |
| `mcmc compute-energy`, `lrdmc compute-energy` | one `parent_folders` entry | `output_parameters` |
| `mcmc compute-force`, `lrdmc compute-force` | one `parent_folders` entry | `output_parameters` |
| `lrdmc extrapolate-energy` | one `parent_folders` entry per lattice spacing | `output_parameters` |

## Options

`options` holds the options of the subcommand exactly as on the command line.
A value of `true` is a flag, and a list repeats the option:

```python
options = orm.Dict({"-j2": 1.0, "-j3": "mo"})  # trexio convert-to
options = orm.Dict({"-k": "pgto+pw", "-u": ["Li=Li.upf", "H=H.upf"]})  # qe
options = orm.Dict({"-w": 100, "-b": 10})  # compute-energy
```

The plugin chooses nothing here: Jastrow parameters, projection kinds,
warmup and binning are the caller's decisions. It sets only the output file
(`-o`) of the conversions and `to-xyz`. With `-s`/`--save-graph`
(`analyze-output`, `extrapolate-energy`) the graph is stored as
`output_graph`, in the format of the given suffix. Options that open a window
(`-g`/`--plot-graph`, `-p`/`--plot_graph` of `analyze-output`) are rejected.

`vmc analyze-output` tabulates what the VMC calculations already report in
`output_parameters["steps"]`; use it for its graph.

### Orbital tiling

`qe` and `trexio convert-to` have first-class inputs for jQMC orbital tiling:

```python
builder.tiling = orm.List([2, 1, 1])
builder.supertwist = orm.List([0.22, 0.13, 0.09])  # one sector
# or, instead of supertwist:
builder.twist_average = orm.Bool(True)  # weighted ensemble
```

`tiling` contains either three diagonal integers or nine row-major integers.
It must be non-singular. `supertwist` contains three finite fractional
supercell coordinates, requires `tiling`, and is mutually exclusive with
`twist_average`. AiiDA serializes each matrix or vector as one argv element,
for example `--tiling`, `"2 1 1 1 1 1 3 2 1"`; quote characters are not part
of the value.

The converter checks the source k-point cosets. A nontrivial Gamma supercell
still needs the complete primitive k-point coset. AiiDA neither fills missing
points nor substitutes Gamma for an invalid sector. Converter errors are
reported in the CalcJob exit message.

The six source/sector routes use the same ports:

| source | Gamma supercell | one general supertwist | weighted ensemble |
|---|---|---|---|
| QE | `tiling=[2,1,1]`, `supertwist=[0,0,0]` | `tiling=[2,1,1]`, `supertwist=[0.22,0.13,0.09]` | `tiling=[2,1,1]`, `twist_average=True` |
| PySCF/TREXIO | same | same | same |

For QE, `options` selects `--kind` from `pw`, `bspline`, `pgto` and
`pgto+pw`. For TREXIO, the converter preserves the Cartesian or spherical GTO
form; `--ao-conv-to` is rejected for tiled input. `pgto+bspline` and tiled
`-j3 mo` are unsupported and fail explicitly. Without these three inputs, the
conversion is untiled. `--tiling`, `--supertwist` and `--twist-average` are
given only through these inputs; in `options` they are rejected.

The tiled conversions through AiiDA are validated against the direct
`jqmc-tool` conversion of the same source by
[`tests/test_tiling.py`](https://github.com/jqmc-project/aiida-jQMC/blob/main/tests/test_tiling.py).

Tiling ends at conversion. Do not add `tiling` or `supertwist` to the QMC
TOML: `JqmcCalculation` receives the converted HDF5 as its `hamiltonian`
input, as usual.

## Files

`input_file`
: Copied into the working directory under its own file name, which is passed
  as the argument.

`input_files` (`vmc average-wf`)
: Each file is copied as `<key>/<file name>` and passed in key order; the
  original file names are kept because `jqmc-tool` records them as the sources
  of the average. Pass the steps of one optimization, e.g. the
  `output_hamiltonians` of a VMC `JqmcCalculation` run with `all_hamiltonians`;
  `jqmc-tool` refuses files that are not steps of one optimization.

`parent_folders` (post-processing)
: The `remote_folder` of a `JqmcCalculation`. Its checkpoint `restart.h5` is
  linked as `<key>/restart.h5` and passed as the argument, in key order; for a
  twist-averaged parent, its checkpoint directory `restart_twists` is linked
  instead. For
  `extrapolate-energy`, use one key per lattice spacing, e.g.
  `{"a020": ..., "a030": ...}`.

`parent_folders` and `save_path` (`qe`)
: The folder, e.g. the `remote_folder` of an aiida-quantumespresso
  `PwCalculation`, is linked whole as `<key>`. `save_path` is the path of
  `prefix.save`, or of the directory holding exactly one `*.save`, inside it:
  `out` for aiida-quantumespresso.

`input_folder` (`qe`)
: Instead of `parent_folders`: a Quantum ESPRESSO output directory stored in
  AiiDA (a `FolderData`), copied into the working directory as
  `input_folder`. `save_path` points inside it as above. The other
  subcommands reject it.

The tool runs on the computer of the parent folders (a `parent_folders`
entry must be on the computer of `code`), without MPI by default
(`withmpi = False`, one process).

```python
from aiida import orm
from aiida.engine import submit
from aiida.plugins import CalculationFactory

JqmcToolCalculation = CalculationFactory("jqmc.tool")

pw = orm.load_node(PW_PK)  # a finished PwCalculation
builder = JqmcToolCalculation.get_builder()
builder.code = orm.load_code("jqmc-tool@<computer>")
builder.command = orm.Str("qe")
builder.parent_folders = {"pw": pw.outputs.remote_folder}
builder.save_path = orm.Str("out")
builder.options = orm.Dict({"-k": "pgto+pw", "-j2": 1.0})
builder.metadata.options = {"queue_name": "<queue>", "account": "<account>", "max_wallclock_seconds": 600}
node = submit(builder)
# node.outputs.output_hamiltonian -> the `hamiltonian` of a JqmcCalculation
```

## Outputs and exit codes

`output_hamiltonian` (conversions) is the `hamiltonian_data.h5` written.
For a tiled conversion, `physical_structure` is the supercell structure read
back from that HDF5 with jQMC's loader. `conversion_metadata` records the
actual tiling matrix, representation and primitive block types, selected
supertwists, raw and normalized sector weights, electron counts, J3 basis,
jQMC build, source node UUID and Code UUID. These are output nodes of the
conversion CalcJob, while AiiDA's normal input links retain the primitive
source. The primitive input structure is never presented as the structure of
the QMC result.

`output_parameters` (post-processing) is what
`jqmc.output_parser` extracts from the stdout:
`parse_tool_energy`, `parse_tool_force` or `parse_lrdmc_extrapolation`, plus
`exception` when jqmc-tool reported one.

| status | label | meaning |
|---|---|---|
| 301 | `ERROR_OUTPUT_STDOUT_MISSING` | the stdout was not retrieved |
| 303 | `ERROR_TOOL_FAILED` | jqmc-tool reported an error or raised an exception |
| 305 | `ERROR_NOT_STARTED` | jqmc-tool wrote nothing |
| 312 | `ERROR_OUTPUT_MISSING` | jqmc-tool ended without its result (no Hamiltonian, energy, forces or extrapolation) |
| 313 | `ERROR_TILING_CONVERSION_FAILED` | jQMC rejected a tiled source, coset or option combination; the converter message is retained |
| 314 | `ERROR_TILING_OUTPUT_INVALID` | jQMC could not deserialize the tiled Hamiltonian that the converter wrote; it is not stored as `output_hamiltonian` |

As for `JqmcCalculation`, a verdict set by the scheduler (e.g. the wall time) is
kept.

## Specification

```{eval-rst}
.. aiida-calcjob:: JqmcToolCalculation
    :module: aiida_jqmc.calculations.tool
```
