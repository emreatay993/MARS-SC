"""
Export Mermaid diagrams from architecture.md to SVG and PNG files.

Output directory:
    docs/architecture_diagrams/
"""

from __future__ import annotations

import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE_MD = ROOT / "architecture.md"
OUTPUT_DIR = ROOT / "docs" / "architecture_diagrams"
KROKI_BASE_URL = "https://kroki.io/mermaid"

# Order matches the Mermaid blocks in architecture.md.
DIAGRAM_BASENAMES = [
    "component_map",
    "solve_pipeline",
    "solve_sequence",
]


def _extract_mermaid_blocks(markdown_text: str) -> list[str]:
    pattern = re.compile(r"```mermaid\s*\n(.*?)\n```", re.DOTALL)
    return [match.strip() for match in pattern.findall(markdown_text)]


def _render_with_kroki(diagram_source: str, fmt: str) -> bytes:
    url = f"{KROKI_BASE_URL}/{fmt}"
    request = urllib.request.Request(
        url=url,
        data=diagram_source.encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "text/plain",
            "User-Agent": "MARS-SC-architecture-export/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> int:
    if not ARCHITECTURE_MD.exists():
        print(f"ERROR: File not found: {ARCHITECTURE_MD}")
        return 1

    markdown_text = ARCHITECTURE_MD.read_text(encoding="utf-8")
    mermaid_blocks = _extract_mermaid_blocks(markdown_text)

    if len(mermaid_blocks) < len(DIAGRAM_BASENAMES):
        print(
            "ERROR: Not enough Mermaid blocks found in architecture.md. "
            f"Expected at least {len(DIAGRAM_BASENAMES)}, found {len(mermaid_blocks)}."
        )
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for basename, source in zip(DIAGRAM_BASENAMES, mermaid_blocks):
        mmd_path = OUTPUT_DIR / f"{basename}.mmd"
        mmd_path.write_text(source + "\n", encoding="utf-8")

        for fmt in ("svg", "png"):
            try:
                rendered = _render_with_kroki(source, fmt)
            except urllib.error.HTTPError as err:
                print(f"ERROR: HTTP {err.code} while rendering {basename}.{fmt}")
                return 1
            except urllib.error.URLError as err:
                print(f"ERROR: Network error while rendering {basename}.{fmt}: {err}")
                return 1

            out_path = OUTPUT_DIR / f"{basename}.{fmt}"
            out_path.write_bytes(rendered)
            print(f"Wrote {out_path.relative_to(ROOT)}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
