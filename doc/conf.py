"""Sphinx configuration of the aiida-jqmc documentation."""

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
import pathlib
import sys
from datetime import datetime

# Make aiida_jqmc importable for autodoc without installing it.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import aiida_jqmc


def _schema_rows(schema: dict, path: str = "") -> list[str]:
    """One Markdown table row per property of *schema*, nested properties dotted."""
    rows = []
    for key, value in schema.get("properties", {}).items():
        name = f"{path}.{key}" if path else key
        types = value.get("type", "")
        types = " or ".join(types) if isinstance(types, list) else types
        description = value.get("description", "").replace("|", "\\|")
        rows.append(f"| `{name}` | {types} | {value.get('unit', '')} | {description} |")
        rows.extend(_schema_rows(value, name))
        if isinstance(value.get("items"), dict):
            rows.extend(_schema_rows(value["items"], f"{name}[]"))
    return rows


def write_output_schema_page() -> None:
    """Render the output schemas of aiida_jqmc.schemas as doc/output_schema.md (not tracked)."""
    from aiida_jqmc.schemas import output_parameters_schema, tool_commands_with_parameters, tool_output_parameters_schema

    header = "| field | type | unit | meaning |\n|---|---|---|---|"
    lines = [
        "(output_schema_tags)=",
        "",
        "# Output schema",
        "",
        "Generated from {mod}`aiida_jqmc.schemas`. Get the JSON Schema itself with",
        "`output_parameters_schema(job_type)` or `tool_output_parameters_schema(command)`.",
        "`[]` marks the entries of a list. A field whose line was not printed is `null`.",
        "",
    ]
    for job_type in ("mcmc", "vmc", "lrdmc-bra", "lrdmc-tau"):
        lines += [f"## JqmcCalculation: `{job_type}`", "", header, *_schema_rows(output_parameters_schema(job_type)), ""]
    for command in tool_commands_with_parameters():
        lines += [
            f"## JqmcToolCalculation: `{command}`",
            "",
            header,
            *_schema_rows(tool_output_parameters_schema(command)),
            "",
        ]
    (pathlib.Path(__file__).parent / "output_schema.md").write_text("\n".join(lines))


write_output_schema_page()

# -- General configuration -----------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    # `aiida-calcjob` / `aiida-workchain` directives: inputs, outputs and exit
    # codes rendered from the process specs, so they cannot drift from the code.
    "aiida.sphinxext",
]
myst_enable_extensions = ["linkify", "colon_fence", "deflist"]
myst_heading_anchors = 4

napoleon_google_docstring = True
napoleon_numpy_docstring = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "aiida": ("https://aiida.readthedocs.io/projects/aiida-core/en/stable/", None),
    "python": ("https://docs.python.org/3", None),
}
# Build without network access (no cross-references to aiida-core and Python).
if os.environ.get("AIIDA_JQMC_DOCS_OFFLINE") == "1":
    intersphinx_mapping = {}

master_doc = "index"
exclude_patterns = ["_build", "README.md"]

project = "aiida-jqmc"
author = "Kosuke Nakano"
copyright = f"{datetime.now().year}, {author}"
release = aiida_jqmc.__version__
version = ".".join(release.split(".")[:2])

pygments_style = "sphinx"

# -- Options for HTML output ---------------------------------------------------

html_theme = "sphinx_book_theme"
html_theme_options = {
    "icon_links": [
        {"name": "GitHub", "url": "https://github.com/jqmc-project/aiida-jQMC", "icon": "fa-brands fa-github"},
    ],
}
html_title = f"aiida-jqmc v{release}"
htmlhelp_basename = "aiida-jqmcdoc"
