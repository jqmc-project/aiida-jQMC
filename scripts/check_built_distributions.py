"""Check the sdist and the wheel that `python -m build` wrote to dist/.

Run from the repository root after `python -m build` (the wheel is built from
the sdist). The sdist must hold every tracked package, metadata and license
file; the wheel must be the pure-Python wheel of the same name and version,
with every package module, the generated version file and exactly the entry
points that pyproject.toml declares. On a release tag `vX.Y.Z` the version
must be `X.Y.Z`.
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

import os
import subprocess
import tarfile
import tomllib
import zipfile
from configparser import ConfigParser
from pathlib import Path

from packaging.utils import parse_sdist_filename, parse_wheel_filename


def main() -> None:
    (wheel,) = Path("dist").glob("*.whl")
    (source,) = Path("dist").glob("*.tar.gz")
    name, version, _, tags = parse_wheel_filename(wheel.name)
    if (name, version) != parse_sdist_filename(source.name) or name != "aiida-jqmc":
        raise SystemExit("Wheel and source distribution name/version differ")
    if {(tag.interpreter, tag.abi, tag.platform) for tag in tags} != {("py3", "none", "any")}:
        raise SystemExit(f"Not a pure-Python py3-none-any wheel: {wheel.name}")
    ref = os.environ.get("GITHUB_REF", "")
    if ref.startswith("refs/tags/v") and str(version) != ref.removeprefix("refs/tags/v"):
        raise SystemExit(f"Built version {version} does not match release tag {ref}")

    with tarfile.open(source) as archive:
        names = {member.partition("/")[2] for member in archive.getnames()}
    required = set(subprocess.check_output(["git", "ls-files", "aiida_jqmc"], text=True).splitlines())
    required.update({"pyproject.toml", "README.md", "LICENSE"})
    if missing := required - names:
        raise SystemExit(f"Source distribution is missing: {sorted(missing)}")

    with zipfile.ZipFile(wheel) as package:
        contents = set(package.namelist())
        dist_info = f"aiida_jqmc-{version}.dist-info"
        entry_points = ConfigParser(delimiters=("=",))
        entry_points.optionxform = str
        entry_points.read_string(package.read(f"{dist_info}/entry_points.txt").decode())
    modules = {path for path in required if path.startswith("aiida_jqmc/") and path.endswith(".py")}
    modules.add("aiida_jqmc/_version.py")
    if missing := modules - contents:
        raise SystemExit(f"Wheel is missing modules: {sorted(missing)}")
    declared = tomllib.loads(Path("pyproject.toml").read_text())["project"]["entry-points"]
    built = {group: {key: value.strip() for key, value in entry_points[group].items()} for group in entry_points.sections()}
    if built != declared:
        raise SystemExit(f"Wheel entry points {built} are not the declared {declared}")
    print(f"{source.name} and {wheel.name}: OK")


if __name__ == "__main__":
    main()
