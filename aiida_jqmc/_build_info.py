"""Version and git commit of the running aiida-jqmc code (the same resolution as jQMC).

1. Live git: when the package is tracked by an enclosing git work tree (``pip install -e .``), ``git rev-parse HEAD``
   gives the commit and ``setuptools_scm`` the version, so both follow ``git pull`` without reinstalling.
2. Build time: otherwise (a wheel or an sdist install) the version and the full commit that ``setuptools_scm`` wrote into
   ``aiida_jqmc/_version.py`` when the distribution was built.
3. ``unknown``.

Nothing here raises. AiiDA records ``aiida_jqmc.__version__`` on every node it creates (``version.plugin``).
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

import functools
import re
import subprocess
from pathlib import Path

__all__ = ("build_info",)

_PACKAGE_DIR = Path(__file__).resolve().parent
_FULL_HASH = re.compile(r"[0-9a-f]{40}")


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(_PACKAGE_DIR), *args], capture_output=True, text=True, timeout=5.0, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


@functools.cache
def build_info() -> tuple[str, str]:
    """``(version, git_commit)`` of the running code."""
    commit = _git("rev-parse", "HEAD")
    root = _git("rev-parse", "--show-toplevel")
    if commit and root and _FULL_HASH.fullmatch(commit):
        try:
            from setuptools_scm import get_version

            version = get_version(root=root, local_scheme="no-local-version")
        except Exception:
            version = "unknown"
        return version, commit
    try:
        from aiida_jqmc._version import __git_commit_full__, __version__
    except ImportError:
        return "unknown", "unknown"
    node = str(__git_commit_full__ or "")
    node = node[1:] if node.startswith("g") else node
    return __version__, node if _FULL_HASH.fullmatch(node) else "unknown"
