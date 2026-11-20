"""Helpers for preparing JqmcCalculation inputs."""

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

import copy
import pathlib

import toml

from aiida_jqmc.calculations.jqmc import RESERVED_CONTROL_KEYS

__all__ = ("parameters_from_toml",)


def parameters_from_toml(source: str | pathlib.Path | dict) -> tuple[dict, dict]:
    """Turn a jqmc CLI input (TOML file or dict) into ``parameters`` for JqmcCalculation.

    The ``control`` keys that the plugin sets itself (``restart``,
    ``restart_chk``, ``hamiltonian_h5``, ``max_time``) are removed and
    returned separately, so that nothing is dropped silently: the
    Hamiltonian becomes the ``hamiltonian`` input, the checkpoint the
    ``parent_folder`` input, and ``max_time`` follows from the wall time.
    Every other key is kept unchanged.

    Returns:
        tuple: ``(parameters, removed)``, ``removed`` being ``{key: value}``.
    """
    data = toml.load(source) if isinstance(source, (str, pathlib.Path)) else copy.deepcopy(source)
    control = data.get("control", {})
    removed = {key: control.pop(key) for key in RESERVED_CONTROL_KEYS if key in control}
    return data, removed
