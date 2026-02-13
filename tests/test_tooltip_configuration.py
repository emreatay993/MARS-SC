"""
Static checks for tooltip configuration, wiring, and global toggle behavior.

These tests avoid importing PyQt runtime modules by validating source text.
"""

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TOOLTIPS_FILE = SRC_ROOT / "utils" / "tooltips.py"
APP_CONTROLLER_FILE = SRC_ROOT / "ui" / "application_controller.py"
STYLE_CONSTANTS_FILE = SRC_ROOT / "ui" / "styles" / "style_constants.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _all_python_files_under_src():
    return list(SRC_ROOT.rglob("*.py"))


def test_all_tooltip_constants_are_wired():
    """Every TOOLTIP_* constant in tooltips.py should be used in UI source."""
    tooltip_source = _read(TOOLTIPS_FILE)
    constants = set(
        re.findall(r"^TOOLTIP_[A-Z0-9_]+\s*=", tooltip_source, flags=re.MULTILINE)
    )
    constants = {entry.split("=")[0].strip() for entry in constants}

    wired = set()
    for py_file in _all_python_files_under_src():
        text = _read(py_file)
        wired.update(re.findall(r"setToolTip\((TOOLTIP_[A-Z0-9_]+)\)", text))
        wired.update(re.findall(r"setItemData\([^)]*(TOOLTIP_[A-Z0-9_]+)", text))

    missing = sorted(constants - wired)
    assert not missing, f"Tooltip constants defined but not wired: {missing}"


def test_global_tooltip_toggle_is_persistent_and_menu_driven():
    """View menu should include tooltip toggle and persist setting."""
    source = _read(APP_CONTROLLER_FILE)

    required_tokens = [
        "Enable Tooltips",
        "view/tooltips_enabled",
        "QSettings",
        "QEvent.ToolTip",
        "installEventFilter",
        "removeEventFilter",
    ]

    missing = [token for token in required_tokens if token not in source]
    assert not missing, f"Missing expected tooltip-toggle implementation tokens: {missing}"


def test_tooltip_style_matches_mars_palette():
    """Tooltip style should match MARS_ tooltip look and feel values."""
    source = _read(STYLE_CONSTANTS_FILE)

    expected_style_tokens = [
        "QToolTip",
        "background-color: #f7f9fc",
        "color: #1a1a2e",
        "border: 1px solid #5b9bd5",
        "border-radius: 4px",
        "font-size: 8pt",
    ]

    missing = [token for token in expected_style_tokens if token not in source]
    assert not missing, f"Tooltip style is missing expected values: {missing}"


def test_display_load_file_tooltip_documents_expected_csv_format():
    """Display file tooltip should explain required CSV columns and example header."""
    source = _read(TOOLTIPS_FILE)

    assert "TOOLTIP_DISPLAY_LOAD_FILE" in source
    assert "Required columns" in source
    assert "<b>X</b>, <b>Y</b>, <b>Z</b>" in source
    assert "NodeID" in source
    assert "Example header" in source
