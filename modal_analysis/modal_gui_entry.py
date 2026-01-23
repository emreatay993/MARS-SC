#!/usr/bin/env python
"""Entry point for the modal analysis GUI."""

import os
import sys


def _setup_paths() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def main() -> None:
    _setup_paths()
    try:
        from modal_gui.main_window import run
    except Exception as exc:
        print(f"ERROR: Failed to import modal GUI: {exc}")
        input("Press Enter to exit...")
        sys.exit(1)

    run()


if __name__ == "__main__":
    main()
