"""Host-local cache of third-party test inputs that git does not carry.

``tests/data/pseudopotentials.toml`` pins every pseudopotential the Quantum
ESPRESSO fixtures read: its source URL, the sha256 of the upstream bytes and the
paths under ``tests/data`` where jqmc-tool expects it. This module resolves those
entries into a host-local cache and builds the mirror of ``tests/data`` that the
``data_with_pseudopotentials`` fixture of ``tests/conftest.py`` returns.

Cache
    ``$JQMC_TEST_DATA_CACHE`` (a leading ``~`` is expanded), default
    ``~/.cache/jqmc/test-data``; never inside the checkout. The directory and
    its layout are those of the jQMC test suite, so one host keeps one copy for
    both. A file lives at ``<cache>/<sha256>/<name>``. An exclusive ``flock`` on
    ``<cache>/.lock`` serializes the concurrent test sessions of one host. A
    missing file is downloaded into a temporary file in its cache directory,
    checked against the pinned sha256 and published with ``os.replace``. A
    cached file whose sha256 differs is an error, not a silent re-download.

Every failure raises :class:`TestDataCacheError`, which fails the tests that
need the file; nothing is skipped and there is no second source.

``python -m tests._test_data_cache`` pre-populates the cache.
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

import fcntl
import hashlib
import http.client
import os
import ssl
import tempfile
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "data"
MANIFEST = DATA_DIR / "pseudopotentials.toml"
CACHE_ENV = "JQMC_TEST_DATA_CACHE"
DEFAULT_CACHE = "~/.cache/jqmc/test-data"
DOWNLOAD_TIMEOUT_S = 60


class TestDataCacheError(RuntimeError):
    """A pinned test input cannot be provided; the tests that need it must fail."""

    __test__ = False  # not a pytest test class


@dataclass(frozen=True)
class ManifestEntry:
    """One pinned file: where it comes from and where the fixtures expect it."""

    name: str
    url: str
    sha256: str
    locations: tuple[str, ...]


def manifest_entries(manifest: Path = MANIFEST) -> tuple[ManifestEntry, ...]:
    """The pinned entries of ``manifest``, validated."""
    document = tomllib.loads(manifest.read_text(encoding="utf-8"))
    entries = []
    for row in document["pseudopotential"]:
        entry = ManifestEntry(name=row["name"], url=row["url"], sha256=row["sha256"].lower(), locations=tuple(row["locations"]))
        if len(entry.sha256) != 64 or any(char not in "0123456789abcdef" for char in entry.sha256):
            raise TestDataCacheError(f"{manifest}: {entry.name}: sha256 must be 64 hex digits")
        for location in entry.locations:
            if Path(location).name != entry.name or Path(location).is_absolute() or ".." in Path(location).parts:
                raise TestDataCacheError(f"{manifest}: {entry.name}: invalid location {location!r}")
        entries.append(entry)
    return tuple(entries)


def cache_dir() -> Path:
    """The host-local cache directory (not created)."""
    # Resolve before comparing: ROOT is resolved, so a relative sibling
    # (``../cache``) must not be read as inside it, and a symlink alias that
    # points into the checkout must be.
    directory = Path(os.environ.get(CACHE_ENV) or DEFAULT_CACHE).expanduser().resolve()
    if directory == ROOT or ROOT in directory.parents:
        raise TestDataCacheError(f"{CACHE_ENV}={directory} is inside the checkout {ROOT}; choose a host-local directory")
    return directory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(entry: ManifestEntry, target: Path) -> None:
    """Fetch ``entry.url`` into ``target`` through a verified temporary file in its directory."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{entry.name}.", suffix=".part", dir=target.parent)
    try:
        digest = hashlib.sha256()
        # urlopen follows redirects and honors the proxy environment variables.
        with (
            os.fdopen(descriptor, "wb") as output,
            urllib.request.urlopen(entry.url, timeout=DOWNLOAD_TIMEOUT_S) as response,
        ):
            for block in iter(lambda: response.read(1 << 16), b""):
                digest.update(block)
                output.write(block)
        if digest.hexdigest() != entry.sha256:
            raise TestDataCacheError(
                f"download of {entry.url} has sha256 {digest.hexdigest()}, the manifest {MANIFEST} pins {entry.sha256}"
            )
        os.chmod(temporary, 0o444)
        os.replace(temporary, target)
    except (OSError, http.client.HTTPException) as error:  # URLError, HTTPError and timeouts are OSError
        hint = (
            " Python's OpenSSL does not trust the certificate chain (an intercepting proxy?): "
            "point SSL_CERT_FILE at the host's CA bundle."
            if isinstance(getattr(error, "reason", error), ssl.SSLCertVerificationError)
            else ""
        )
        raise TestDataCacheError(
            f"cannot download {entry.url} into {target.parent}: {error}.{hint} "
            f"The tests need network access once per host to fill {CACHE_ENV}, "
            "or run `python -m tests._test_data_cache` where the network is available."
        ) from error
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def ensure_cached(entries: tuple[ManifestEntry, ...] | None = None) -> dict[str, Path]:
    """Cached path of every entry (by name), downloading the missing ones under the host lock."""
    entries = manifest_entries() if entries is None else entries
    directory = cache_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        lock = (directory / ".lock").open("a")
    except OSError as error:
        raise TestDataCacheError(f"cannot use the test-data cache {directory} ({CACHE_ENV}): {error}") from error
    cached: dict[str, Path] = {}
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for entry in entries:
            target = directory / entry.sha256 / entry.name
            if target.exists():
                actual = _sha256(target)
                if actual != entry.sha256:
                    raise TestDataCacheError(
                        f"cached {target} has sha256 {actual}, the manifest pins {entry.sha256}: "
                        "the cache is corrupted; remove the file to download it again"
                    )
            else:
                target.parent.mkdir(exist_ok=True)
                _download(entry, target)
            cached[entry.name] = target
    return cached


def build_data_mirror(destination: Path) -> Path:
    """Build the mirror of ``tests/data`` at ``destination`` (an empty directory).

    Directories are recreated; every checkout file becomes a symbolic link to the
    checkout, and every manifest location a symbolic link to the cached file.
    """
    entries = manifest_entries()
    cached = ensure_cached(entries)
    for directory, children, files in os.walk(DATA_DIR):
        relative = Path(directory).relative_to(DATA_DIR)
        children.sort()
        for child in children:
            (destination / relative / child).mkdir()
        for name in sorted(files):
            os.symlink(Path(directory) / name, destination / relative / name)
    for entry in entries:
        for location in entry.locations:
            link = destination / location
            if os.path.lexists(link):
                raise TestDataCacheError(
                    f"tests/data/{location} is in the checkout and pinned by {MANIFEST.name}; "
                    "third-party pseudopotentials are not kept in git"
                )
            if not link.parent.is_dir():
                raise TestDataCacheError(f"{MANIFEST.name}: {location}: no directory {link.parent.relative_to(destination)}")
            os.symlink(cached[entry.name], link)
    return destination


def main() -> int:
    for name, path in ensure_cached().items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
