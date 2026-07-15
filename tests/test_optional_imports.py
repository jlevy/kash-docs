from __future__ import annotations

import subprocess
import sys


def test_base_package_does_not_import_optional_document_runtimes() -> None:
    code = """
import sys
import kash.kits.docs

optional_modules = ("boto3", "docx", "markitdown", "torch", "weasyprint")
loaded = sorted(
    name
    for name in sys.modules
    if any(name == module or name.startswith(f"{module}.") for module in optional_modules)
)
if loaded:
    raise RuntimeError(f"Base kash-docs import loaded optional modules: {loaded}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert not result.stderr
