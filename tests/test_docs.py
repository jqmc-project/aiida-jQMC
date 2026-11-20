"""The hand-written tables in doc/ match the code they describe."""

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

import pathlib
import re
import tomllib
import urllib.parse

from aiida_jqmc.calculations.jqmc import JqmcCalculation
from aiida_jqmc.calculations.tool import SUBCOMMANDS, JqmcToolCalculation
from aiida_jqmc.workflows.restart import STEP_KEYS, JqmcRestartWorkChain

ROOT = pathlib.Path(__file__).parent.parent
DOC_DIR = ROOT / "doc"


def _table_rows(page: str, first_header: str) -> list[list[str]]:
    """Cells of the Markdown table whose first header cell is ``first_header``, backticks removed."""
    lines = (DOC_DIR / page).read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if re.match(rf"\|\s*{re.escape(first_header)}\s*\|", line))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip().replace("`", "") for cell in line.strip("|").split("|")])
    return rows


def test_calculation_exit_codes():
    """Every parser exit code (3xx) is in the table of calculations.md, with its status."""
    documented = {(int(row[0]), row[1]) for row in _table_rows("calculations.md", "status")}
    expected = {(code.status, label) for label, code in JqmcCalculation.exit_codes.items() if 300 <= code.status < 400}
    assert documented == expected


def test_tool_exit_codes():
    """Every JqmcToolCalculation exit code (3xx) is in the table of tools.md."""
    documented = {(int(row[0]), row[1]) for row in _table_rows("tools.md", "status")}
    expected = {(code.status, label) for label, code in JqmcToolCalculation.exit_codes.items() if 300 <= code.status < 400}
    assert documented == expected


def test_tool_subcommands():
    documented = set()
    for commands, *_ in _table_rows("tools.md", "command"):
        documented.update(commands.split(", "))
    assert documented == set(SUBCOMMANDS)


def test_workchain_exit_codes():
    """The work chain's own exit codes are named, with their status, in workflows.md."""
    text = (DOC_DIR / "workflows.md").read_text()
    own = {label: code for label, code in JqmcRestartWorkChain.exit_codes.items() if code.status >= 410}
    assert own
    for label, code in own.items():
        assert f"`{label}` ({code.status})" in text, label


def test_workchain_exit_codes_are_unique():
    """No exit code of the work chain reuses a status of BaseRestartWorkChain."""
    statuses = [code.status for code in JqmcRestartWorkChain.exit_codes.values()]
    assert len(statuses) == len(set(statuses))


def test_workchain_step_keys():
    documented = {}
    for job_types, steps, counter in _table_rows("workflows.md", "job type"):
        for job_type in job_types.split(", "):
            documented[job_type] = (steps, counter)
    assert documented == STEP_KEYS


def test_exit_codes_named_in_the_text():
    """Every ``LABEL`` (status) written in doc/ is an exit code with that status."""
    known = {label: code.status for label, code in JqmcCalculation.exit_codes.items()}
    known.update({label: code.status for label, code in JqmcRestartWorkChain.exit_codes.items() if code.status >= 410})
    mentions = [
        (label, int(status))
        for page in DOC_DIR.glob("*.md")
        for label, status in re.findall(r"`(ERROR_[A-Z_]+)`\s+\((\d+)\)", page.read_text())
    ]
    assert mentions
    for label, status in mentions:
        assert known.get(label) == status, (label, status)


def test_readme_aiida_badge_matches_the_dependency():
    """The README's AiiDA badge shows the bounds of the ``aiida-core`` dependency in pyproject.toml."""
    dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    (spec,) = [d for d in dependencies if re.match(r"aiida-core\b", d)]
    lower, upper = re.fullmatch(r"aiida-core>=([\d.]+),<([\d.]+)", spec.replace(" ", "")).groups()
    (badge,) = re.findall(r"https://img\.shields\.io/badge/AiiDA-([^)\s]+)-blue", (ROOT / "README.md").read_text())
    assert urllib.parse.unquote(badge) == f"\u2265{lower}, <{upper}"
