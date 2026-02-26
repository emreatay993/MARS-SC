"""Regression tests for Turkish UAT CSV text encoding."""

from __future__ import annotations

import codecs
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TR_UAT_FILE_HEADERS = {
    "MARS_SC_USER_ACCEPTANCE_TESTS_TR.csv": "Test Kimli\u011fi,Ba\u015fl\u0131k,\u00d6n Ko\u015fullar,Ad\u0131mlar,Beklenen Sonu\u00e7",
    "MARS_SC_USER_ACCEPTANCE_TESTS_GROUPED_TR.csv": "Test Grubu,Test Kimli\u011fi,Ba\u015fl\u0131k,\u00d6n Ko\u015fullar,Ad\u0131mlar,Beklenen Sonu\u00e7",
    "MARS_SC_UAT_Consolidated_Tests_TR.csv": "TEST NO,BA\u015eLIK,A\u00c7IKLAMA,G\u0130RD\u0130LER,TEST ADIMLARI,BEKLENEN SONU\u00c7LAR,GEREKS\u0130N\u0130M NO",
}


def test_turkish_uat_csv_files_use_utf8_bom_and_expected_headers() -> None:
    """Ensure Turkish acceptance-test CSVs render consistently on Windows viewers."""
    for file_name, expected_header in TR_UAT_FILE_HEADERS.items():
        file_path = PROJECT_ROOT / file_name
        raw_bytes = file_path.read_bytes()

        assert raw_bytes.startswith(codecs.BOM_UTF8), (
            f"{file_name} must be saved as UTF-8 with BOM for cross-machine rendering."
        )

        first_line = raw_bytes.decode("utf-8-sig").splitlines()[0]
        assert first_line == expected_header
