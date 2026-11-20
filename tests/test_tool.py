"""Tests for JqmcToolCalculation: input validation and the generated command line."""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the jqmc project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import json
import pathlib
import shutil

import pytest
from aiida import orm
from aiida.engine import run_get_node
from aiida.plugins import CalculationFactory

from .conftest import DATA_DIR

JqmcToolCalculation = CalculationFactory("jqmc.tool")


@pytest.fixture
def tool_code(aiida_code_installed, aiida_localhost):
    """`jqmc-tool` registered as a serial InstalledCode (only dry runs use it here)."""
    return aiida_code_installed(
        filepath_executable=shutil.which("jqmc-tool") or "/bin/true", computer=aiida_localhost, with_mpi=False
    )


@pytest.fixture
def parent_folder(aiida_localhost, tmp_path):
    return orm.RemoteData(computer=aiida_localhost, remote_path=str(tmp_path)).store()


def _dry_run(code, **inputs):
    _, node = run_get_node(JqmcToolCalculation, code=code, metadata={"dry_run": True, "store_provenance": False}, **inputs)
    return node.dry_run_info["folder"], node


def _command_line(folder) -> str:
    return (pathlib.Path(folder) / "_aiidasubmit.sh").read_text().strip().splitlines()[-1]


def _command_line_arguments(folder) -> list[str]:
    """The argv AiiDA serializes before its submit script adds shell quoting."""
    calcinfo = json.loads((pathlib.Path(folder) / ".aiida" / "calcinfo.json").read_text())
    return calcinfo["codes_info"][0]["cmdline_params"]


def test_convert_trexio(tool_code):
    """The TREXIO file is copied under its own name; the plugin names the output."""
    trexio = orm.SinglefileData(DATA_DIR / "H2_ecp_ccpvdz_jsd.h5", filename="H2_ecp_ccpvdz_cart.h5")
    folder, node = _dry_run(
        tool_code,
        command=orm.Str("trexio convert-to"),
        input_file=trexio,
        options=orm.Dict({"-j2": 1.0, "-j3": "mo"}),
    )
    line = _command_line(folder)
    assert "'trexio' 'convert-to' 'H2_ecp_ccpvdz_cart.h5' '-j2' '1' '-j3' 'mo' '-o' 'hamiltonian_data.h5'" in line
    assert "> 'aiida.out'" in line
    assert "mpirun" not in line
    assert node.get_option("withmpi") is False


@pytest.mark.parametrize("matrix", [[3, 3, 3], [2, 1, 1, 1, 1, 1, 3, 2, 1]])
def test_tiling_and_supertwist_are_single_arguments(tool_code, matrix):
    """The matrix/vector each occupy one argv entry; argv values contain no shell quotes."""
    trexio = orm.SinglefileData(DATA_DIR / "H2_ecp_ccpvdz_cart.h5")
    folder, _ = _dry_run(
        tool_code,
        command=orm.Str("trexio convert-to"),
        input_file=trexio,
        tiling=orm.List(list=matrix),
        supertwist=orm.List(list=[0.2, -0.1, 0.0]),
    )
    arguments = _command_line_arguments(folder)
    assert arguments[arguments.index("--tiling") + 1] == " ".join(str(item) for item in matrix)
    assert arguments[arguments.index("--supertwist") + 1] == "0.2 -0.1 0.0"
    assert all('"' not in argument and "'" not in argument for argument in arguments)


def test_tiling_qe_twist_average_arguments(tool_code, parent_folder):
    folder, _ = _dry_run(
        tool_code,
        command=orm.Str("qe"),
        parent_folders={"pw": parent_folder},
        tiling=orm.List(list=[2, 1, 1]),
        twist_average=orm.Bool(True),
        options=orm.Dict({"--kind": "pgto+pw"}),
    )
    arguments = _command_line_arguments(folder)
    assert arguments.count("--tiling") == 1
    assert arguments[arguments.index("--tiling") + 1] == "2 1 1"
    assert arguments.count("--twist-average") == 1


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({"tiling": [2, 1]}, "3 diagonal entries or 9"),
        ({"tiling": [2, 1, 0]}, "non-singular"),
        ({"tiling": [2, 1, 1.0]}, "must be an integer"),
        ({"supertwist": [0.0, 0.0, 0.0]}, "requires `tiling`"),
        ({"tiling": [2, 1, 1], "supertwist": [0.0, 0.0]}, "three fractional supercell coordinates"),
        ({"tiling": [2, 1, 1], "supertwist": [0.0, float("nan"), 0.0]}, "must be finite"),
        (
            {"tiling": [2, 1, 1], "supertwist": [0.0, 0.0, 0.0], "twist_average": True},
            "mutually exclusive",
        ),
    ],
)
def test_validate_tiling_inputs(tool_code, inputs, message):
    trexio = orm.SinglefileData(DATA_DIR / "H2_ecp_ccpvdz_cart.h5")
    converted = {
        key: orm.List(list=value) if key in ("tiling", "supertwist") else orm.Bool(value) for key, value in inputs.items()
    }
    with pytest.raises(ValueError, match=message):
        run_get_node(
            JqmcToolCalculation,
            code=tool_code,
            command=orm.Str("trexio convert-to"),
            input_file=trexio,
            **converted,
        )


@pytest.mark.parametrize(
    ("command", "options", "message"),
    [
        ("trexio convert-to", {"-j3": "mo"}, "does not support"),
        ("trexio convert-to", {"--ao-conv-to": "cart"}, "preserves the source angular form"),
        ("qe", {"--kind": "pgto+bspline"}, "supports"),
    ],
)
def test_validate_tiled_converter_combinations(tool_code, parent_folder, command, options, message):
    inputs = (
        {"input_file": orm.SinglefileData(DATA_DIR / "H2_ecp_ccpvdz_cart.h5")}
        if command.startswith("trexio")
        else {"parent_folders": {"pw": parent_folder}}
    )
    with pytest.raises(ValueError, match=message):
        run_get_node(
            JqmcToolCalculation,
            code=tool_code,
            command=orm.Str(command),
            options=orm.Dict(options),
            tiling=orm.List(list=[2, 1, 1]),
            **inputs,
        )


@pytest.mark.parametrize("option", [{"--tiling": "2 1 1"}, {"--supertwist": "0 0 0"}, {"--twist-average": True}])
def test_selectors_are_not_options(tool_code, option):
    with pytest.raises(ValueError, match="inputs, not `options`"):
        run_get_node(
            JqmcToolCalculation,
            code=tool_code,
            command=orm.Str("trexio convert-to"),
            input_file=orm.SinglefileData(DATA_DIR / "H2_ecp_ccpvdz_cart.h5"),
            options=orm.Dict(option),
        )


def test_twist_average_without_tiling(tool_code):
    folder, _ = _dry_run(
        tool_code,
        command=orm.Str("trexio convert-to"),
        input_file=orm.SinglefileData(DATA_DIR / "H2_ecp_ccpvdz_cart.h5"),
        twist_average=orm.Bool(True),
    )
    arguments = _command_line_arguments(folder)
    assert arguments.count("--twist-average") == 1
    assert "--tiling" not in arguments and "--supertwist" not in arguments


def test_extrapolate_links_every_checkpoint(tool_code, aiida_localhost, tmp_path):
    """Each parent's restart.h5 is linked into a subdirectory named by its key, in key order."""
    parents = {}
    for key in ("a030", "a020"):
        path = tmp_path / key
        path.mkdir()
        parents[key] = orm.RemoteData(computer=aiida_localhost, remote_path=str(path)).store()
    folder, _ = _dry_run(
        tool_code,
        command=orm.Str("lrdmc extrapolate-energy"),
        parent_folders=parents,
        options=orm.Dict({"-p": 1, "-w": 50}),
    )
    assert "'a020/restart.h5' 'a030/restart.h5' '-p' '1' '-w' '50'" in _command_line(folder)
    assert sorted(p.name for p in pathlib.Path(folder).iterdir() if p.is_dir() and not p.name.startswith(".")) == [
        "a020",
        "a030",
    ]


def test_average_wf_copies_every_step_in_key_order(tool_code):
    """`vmc average-wf`: each input file is copied under its key with its own name, in key order, and the
    averaged Hamiltonian is written to the file the parser retrieves."""
    files = {
        key: orm.SinglefileData(DATA_DIR / "H2_ecp_ccpvdz_jsd.h5", filename=f"hamiltonian_data_opt_step_{step}.h5")
        for key, step in (("s040", 40), ("s039", 39))
    }
    folder, _ = _dry_run(tool_code, command=orm.Str("vmc average-wf"), input_files=files)
    assert _command_line_arguments(folder)[:4] == [
        "vmc",
        "average-wf",
        "s039/hamiltonian_data_opt_step_39.h5",
        "s040/hamiltonian_data_opt_step_40.h5",
    ]
    assert _command_line_arguments(folder)[-2:] == ["-o", "hamiltonian_data.h5"]
    assert (pathlib.Path(folder) / "s039" / "hamiltonian_data_opt_step_39.h5").is_file()


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({}, "reads one or more `input_files`"),
        ({"input_file": True}, "reads one or more `input_files`"),
    ],
)
def test_average_wf_validation(tool_code, inputs, message):
    h5 = orm.SinglefileData(DATA_DIR / "H2_ecp_ccpvdz_jsd.h5")
    extra = {"input_files": {"s1": h5}} if inputs.get("input_file") else {}
    if inputs.get("input_file"):
        extra["input_file"] = h5
    with pytest.raises(ValueError, match=message):
        _dry_run(tool_code, command=orm.Str("vmc average-wf"), **extra)


def test_input_files_only_for_average_wf(tool_code):
    h5 = orm.SinglefileData(DATA_DIR / "H2_ecp_ccpvdz_jsd.h5")
    with pytest.raises(ValueError, match="`input_files` are only read by"):
        _dry_run(tool_code, command=orm.Str("trexio convert-to"), input_file=h5, input_files={"s1": h5})


def test_qe_links_the_whole_folder(tool_code, parent_folder):
    """The QE folder is linked whole; save_path points inside it; a list repeats an option."""
    folder, _ = _dry_run(
        tool_code,
        command=orm.Str("qe"),
        parent_folders={"pw": parent_folder},
        save_path=orm.Str("out/"),
        options=orm.Dict({"-k": "pgto+pw", "-u": ["H=H.upf", "O=O.upf"]}),
    )
    line = _command_line(folder)
    assert "'qe' 'pw/out' '-k' 'pgto+pw' '-u' 'H=H.upf' '-u' 'O=O.upf' '-o' 'hamiltonian_data.h5'" in line
    calcinfo = json.loads((pathlib.Path(folder) / ".aiida" / "calcinfo.json").read_text())
    assert calcinfo["remote_symlink_list"] == [[parent_folder.computer.uuid, parent_folder.get_remote_path(), "pw"]]


def test_qe_from_a_folder_in_the_repository(tool_code, tmp_path):
    """A QE directory stored as FolderData is copied in and read from there."""
    save = tmp_path / "out" / "si.save"
    save.mkdir(parents=True)
    (save / "data-file-schema.xml").write_text("<qes/>")
    folder, _ = _dry_run(tool_code, command=orm.Str("qe"), input_folder=orm.FolderData(tree=tmp_path), save_path=orm.Str("out"))
    assert "'qe' 'input_folder/out'" in _command_line(folder)
    calcinfo = json.loads((pathlib.Path(folder) / ".aiida" / "calcinfo.json").read_text())
    assert [entry[1:] for entry in calcinfo["local_copy_list"]] == [["", "input_folder"]]


def test_qe_takes_one_source(tool_code, parent_folder, tmp_path):
    with pytest.raises(ValueError, match="either one of `parent_folders` or an `input_folder`"):
        run_get_node(
            JqmcToolCalculation,
            code=tool_code,
            command=orm.Str("qe"),
            parent_folders={"pw": parent_folder},
            input_folder=orm.FolderData(tree=tmp_path),
        )


def test_save_path_only_for_qe(tool_code, parent_folder):
    with pytest.raises(ValueError, match="`save_path` is only read by"):
        run_get_node(
            JqmcToolCalculation,
            code=tool_code,
            command=orm.Str("mcmc compute-energy"),
            parent_folders={"vmc": parent_folder},
            save_path=orm.Str("out"),
        )


def test_flags(tool_code, parent_folder):
    """`true` is a flag, `false` leaves the option out."""
    folder, _ = _dry_run(
        tool_code,
        command=orm.Str("mcmc compute-energy"),
        parent_folders={"vmc": parent_folder},
        options=orm.Dict({"--flag-on": True, "--flag-off": False}),
    )
    line = _command_line(folder)
    assert "'vmc/restart.h5' '--flag-on'" in line
    assert "--flag-off" not in line


def test_integral_float_option_is_written_as_an_integer(tool_code, parent_folder):
    """jqmc-tool's integer options reject '10.0'; a float option still reads '10', and 0.5 stays 0.5."""
    folder, _ = _dry_run(
        tool_code,
        command=orm.Str("mcmc compute-energy"),
        parent_folders={"vmc": parent_folder},
        options=orm.Dict({"-b": 10.0, "-w": 0.5}),
    )
    line = _command_line(folder)
    assert "'-b' '10'" in line
    assert "'-w' '0.5'" in line


@pytest.mark.parametrize(
    ("command", "inputs", "message"),
    [
        ("vmc generate-input", {}, "`command` must be one of"),
        ("trexio convert-to", {"parent_folders": True}, "reads one `input_file`"),
        ("mcmc compute-energy", {}, "reads `parent_folders` and no `input_file`"),
        ("mcmc compute-energy", {"parent_folders": 2}, "exactly one of `parent_folders`"),
        ("qe", {"parent_folders": 2}, "exactly one of `parent_folders`"),
        ("mcmc compute-energy", {"parent_folders": "elsewhere"}, "on the same computer as `code`"),
        ("mcmc compute-energy", {"parent_folders": 1, "input_folder": True}, "`input_folder` is only read by"),
        ("mcmc compute-energy", {"parent_folders": 1, "tiling": [2, 1, 1]}, "only valid for"),
    ],
)
def test_validate_inputs(tool_code, aiida_localhost, aiida_computer_local, tmp_path, command, inputs, message):
    inputs = dict(inputs)
    if "parent_folders" in inputs:
        count = inputs["parent_folders"]
        computer = aiida_computer_local() if count == "elsewhere" else aiida_localhost
        count = 1 if count in (True, "elsewhere") else count
        inputs["parent_folders"] = {
            f"p{i}": orm.RemoteData(computer=computer, remote_path=str(tmp_path)).store() for i in range(count)
        }
    if "input_folder" in inputs:
        inputs["input_folder"] = orm.FolderData(tree=tmp_path)
    if "tiling" in inputs:
        inputs["tiling"] = orm.List(list=inputs["tiling"])
    with pytest.raises(ValueError, match=message):
        run_get_node(JqmcToolCalculation, code=tool_code, command=orm.Str(command), **inputs)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"j2": 1.0}, "as on the command line"),
        ({"-o": "x.h5"}, "set by the plugin"),
        ({"-j3": {"mo": 1}}, "give a number, a string"),
        ({"-u": []}, "give a number, a string"),
    ],
)
def test_validate_options(tool_code, parent_folder, options, message):
    with pytest.raises(ValueError, match=message):
        run_get_node(
            JqmcToolCalculation,
            code=tool_code,
            command=orm.Str("mcmc compute-energy"),
            parent_folders={"vmc": parent_folder},
            options=orm.Dict(options),
        )


@pytest.mark.parametrize(("command", "option"), [("lrdmc extrapolate-energy", "-g"), ("vmc analyze-output", "--plot_graph")])
def test_interactive_options_are_rejected(tool_code, parent_folder, command, option):
    with pytest.raises(ValueError, match="opens a window"):
        run_get_node(
            JqmcToolCalculation,
            code=tool_code,
            command=orm.Str(command),
            parent_folders={"run": parent_folder},
            options=orm.Dict({option: True}),
        )


def test_saved_graph_is_named_and_retrieved(tool_code, parent_folder):
    """-s keeps the requested format under a name the parser knows."""
    folder, _ = _dry_run(
        tool_code,
        command=orm.Str("vmc analyze-output"),
        parent_folders={"vmc1": parent_folder, "vmc0": parent_folder},
        options=orm.Dict({"-s": "vmc.jpg"}),
    )
    assert "'vmc0/aiida.out' 'vmc1/aiida.out' '-s' 'graph.jpg'" in _command_line(folder)
    calcinfo = json.loads((pathlib.Path(folder) / ".aiida" / "calcinfo.json").read_text())
    assert "graph.jpg" in calcinfo["retrieve_list"]
