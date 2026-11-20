#!/usr/bin/env python3
"""Check an installed aiida-jqmc distribution as a user runs it, without running jqmc.

Run with ``python -I`` after installing a wheel, by public-build-distributions.yml
and public-verify-index.yml::

    python -I scripts/check_installed_aiida_jqmc.py tests/data/H2_ecp_ccpvdz_jsd.h5 tests/data/mcmc.out

- aiida_jqmc must be imported from the installation, not from this checkout, and
  report the version of its distribution and the full commit it was built from
  (``--commit``: that commit).
- A throwaway AiiDA profile (``verdi presto``: SQLite, no broker) is made in a
  temporary directory; no user profile is read or changed.
- ``verdi plugin list`` lists, and the AiiDA plugin factories load, every entry
  point of the distribution.
- An MCMC JqmcCalculation writes its input files (a dry run: nothing is
  submitted, no code is run), and the installed parser parses a recorded jqmc
  MCMC log.

It checks that the shipped package works, not its numbers: the plugin is
tested elsewhere.
"""

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

import argparse
import math
import os
import re
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
from importlib.metadata import distribution
from pathlib import Path

CHECKOUT = Path(__file__).resolve().parents[1]
GROUPS = ("aiida.calculations", "aiida.parsers", "aiida.workflows")
MCMC = {
    "control": {"job_type": "mcmc", "number_of_walkers": 4},
    "mcmc": {"num_mcmc_steps": 60, "num_mcmc_warmup_steps": 30, "num_mcmc_bin_blocks": 10},
}
OPTIONS = {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 2}, "max_wallclock_seconds": 3600}


def verdi(*args: str) -> str:
    """Run the installed verdi console script, isolated from the checkout."""
    command = Path(sysconfig.get_path("scripts")) / "verdi"
    result = subprocess.run(
        [sys.executable, "-I", str(command), *args], check=True, capture_output=True, text=True, timeout=300
    )
    return result.stdout


def check_installation(commit: str) -> None:
    import aiida_jqmc

    location = Path(aiida_jqmc.__file__).resolve()
    if location.is_relative_to(CHECKOUT):
        raise SystemExit(f"aiida_jqmc is imported from the checkout ({location}), not from the installation")
    version = distribution("aiida-jqmc").version
    if aiida_jqmc.__version__ != version:
        raise SystemExit(f"aiida_jqmc.__version__ {aiida_jqmc.__version__} is not the distribution version {version}")
    if not re.fullmatch(r"[0-9a-f]{40}", aiida_jqmc.__git_commit__):
        raise SystemExit(f"The distribution does not report its full build commit: {aiida_jqmc.__git_commit__!r}")
    if commit and aiida_jqmc.__git_commit__ != commit:
        raise SystemExit(f"The distribution was built from {aiida_jqmc.__git_commit__}, not {commit}")
    print(f"aiida-jqmc {version} (commit {aiida_jqmc.__git_commit__}) from {location.parent}: OK")


def check_entry_points() -> None:
    from aiida.plugins import CalculationFactory, ParserFactory, WorkflowFactory

    factories = dict(zip(GROUPS, (CalculationFactory, ParserFactory, WorkflowFactory), strict=True))
    entry_points = [entry for entry in distribution("aiida-jqmc").entry_points if entry.group.startswith("aiida.")]
    if {entry.group for entry in entry_points} != set(GROUPS):
        raise SystemExit(f"Unexpected entry point groups: {sorted({entry.group for entry in entry_points})}")
    for group in GROUPS:
        listed = verdi("plugin", "list", group)
        for entry in (entry for entry in entry_points if entry.group == group):
            if entry.name not in listed:
                raise SystemExit(f"verdi plugin list {group} does not list {entry.name}:\n{listed}")
            if factories[group](entry.name) is not entry.load():
                raise SystemExit(f"{group} {entry.name} does not load {entry.value}")
            print(f"{group} {entry.name} -> {entry.value}: OK")


def check_calculation(hamiltonian: Path, log: Path, workdir: Path) -> None:
    from aiida import orm
    from aiida.common.links import LinkType
    from aiida.engine import run_get_node
    from aiida.plugins import CalculationFactory, ParserFactory

    # verdi presto has its own `localhost`; this one runs nothing either.
    computer = orm.Computer(
        label="aiida-jqmc-check",
        hostname="localhost",
        transport_type="core.local",
        scheduler_type="core.direct",
        workdir=str(workdir / "work"),
    ).store()
    computer.configure()
    executable = Path(sysconfig.get_path("scripts")) / "jqmc"
    if not executable.is_file():
        raise SystemExit(f"The jqmc dependency installed no jqmc executable at {executable}")
    code = orm.InstalledCode(computer=computer, filepath_executable=str(executable), with_mpi=True).store()

    _, node = run_get_node(
        CalculationFactory("jqmc.jqmc"),
        code=code,
        parameters=orm.Dict(MCMC),
        hamiltonian=orm.SinglefileData(hamiltonian),
        metadata={"options": OPTIONS, "dry_run": True, "store_provenance": False},
    )
    folder = Path(node.dry_run_info["folder"])
    control = tomllib.loads((folder / "aiida.toml").read_text())["control"]
    if control["job_type"] != "mcmc" or not (folder / control["hamiltonian_h5"]).is_file():
        raise SystemExit(f"Invalid MCMC input files in {folder}: {control}")
    print("JqmcCalculation (mcmc) input files, dry run: OK")

    calculation = orm.CalcJobNode(computer=computer, process_type="aiida.calculations:jqmc.jqmc")
    for name, value in {**OPTIONS, "output_filename": "aiida.out", "scheduler_stderr": "_scheduler-stderr.txt"}.items():
        calculation.set_option(name, value)
    parameters = orm.Dict({"control": {"job_type": "mcmc"}, "mcmc": {}}).store()
    calculation.base.links.add_incoming(parameters, link_type=LinkType.INPUT_CALC, link_label="parameters")
    calculation.store()
    retrieved = orm.FolderData()
    retrieved.base.repository.put_object_from_bytes(log.read_bytes(), "aiida.out")
    retrieved.base.links.add_incoming(calculation, link_type=LinkType.CREATE, link_label="retrieved")
    retrieved.store()
    results, parse = ParserFactory("jqmc.jqmc").parse_from_node(calculation, store_provenance=False)
    energy = results["output_parameters"].get_dict().get("energy") if "output_parameters" in results else None
    if parse.exit_status != 0 or energy is None or not math.isfinite(energy):
        raise SystemExit(f"Parsing the recorded MCMC log failed: exit {parse.exit_status} {parse.exit_message}")
    print("JqmcParser on a recorded MCMC log: OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hamiltonian", type=Path, help="a jQMC Hamiltonian (HDF5)")
    parser.add_argument("mcmc_log", type=Path, help="a recorded jqmc MCMC log")
    parser.add_argument("--commit", default="", help="the full commit the distribution must have been built from")
    args = parser.parse_args()
    hamiltonian, log = args.hamiltonian.resolve(), args.mcmc_log.resolve()
    with tempfile.TemporaryDirectory(prefix="aiida-jqmc-installed-") as directory:
        workdir = Path(directory)
        # Before aiida is imported: the profile lives in the temporary directory only.
        os.environ["AIIDA_PATH"] = str(workdir)
        os.chdir(workdir)
        check_installation(args.commit)
        verdi("presto")
        from aiida import load_profile

        load_profile()
        check_entry_points()
        check_calculation(hamiltonian, log, workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
