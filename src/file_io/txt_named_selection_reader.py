"""
Nodal named-selection reader for Mechanical-style TXT exports.

The expected format is a delimited text table with at least these columns:
``Node Number``, ``X Location (<unit>)``, ``Y Location (<unit>)``,
and ``Z Location (<unit>)``. Coordinate units are normalized to millimeters.
"""

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Tuple


@dataclass
class TXTNamedSelection:
    """One imported nodal named selection parsed from a TXT file."""

    name: str
    location: str
    ids: List[int]
    coordinates_mm: Dict[int, Tuple[float, float, float]]


class TXTNamedSelectionReader:
    """Parse one nodal named selection from a text table."""

    _VALID_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    _UNIT_FACTORS_TO_MM = {
        "m": 1000.0,
        "meter": 1000.0,
        "meters": 1000.0,
        "cm": 10.0,
        "centimeter": 10.0,
        "centimeters": 10.0,
        "mm": 1.0,
        "millimeter": 1.0,
        "millimeters": 1.0,
    }

    def __init__(self, txt_path: str, selection: TXTNamedSelection):
        self.txt_path = txt_path
        self.selections = {selection.name: selection}

    @classmethod
    def from_file(cls, txt_path: str, selection_name: str) -> "TXTNamedSelectionReader":
        """Parse one named selection from a TXT file."""
        if not cls.is_valid_selection_name(selection_name):
            raise ValueError(
                "Named selection names must start with a letter or underscore "
                "and contain only letters, numbers, and underscores."
            )

        try:
            with open(txt_path, "r", encoding="utf-8-sig", errors="ignore") as handle:
                lines = [line.rstrip("\n\r") for line in handle]
        except OSError as exc:
            raise ValueError(f"Failed to read TXT file: {exc}") from exc

        header_index, header = cls._find_header(lines)
        delimiter = cls._detect_delimiter(header)
        headers = cls._split_line(header, delimiter)
        column_map = cls._build_column_map(headers)

        node_ids: List[int] = []
        coordinates_mm: Dict[int, Tuple[float, float, float]] = {}
        seen = set()

        for line in lines[header_index + 1:]:
            if not line.strip():
                continue

            values = cls._split_line(line, delimiter)
            if len(values) < len(headers):
                continue

            try:
                node_id = int(float(values[column_map["node"]].strip()))
                x = float(values[column_map["x"]].strip()) * column_map["x_factor"]
                y = float(values[column_map["y"]].strip()) * column_map["y_factor"]
                z = float(values[column_map["z"]].strip()) * column_map["z_factor"]
            except (ValueError, IndexError):
                continue

            if node_id in seen:
                continue

            seen.add(node_id)
            node_ids.append(node_id)
            coordinates_mm[node_id] = (x, y, z)

        if not node_ids:
            raise ValueError("No valid node rows were found in the selected TXT file.")

        selection = TXTNamedSelection(
            name=selection_name,
            location="nodal",
            ids=node_ids,
            coordinates_mm=coordinates_mm,
        )
        return cls(txt_path=txt_path, selection=selection)

    @classmethod
    def is_valid_selection_name(cls, name: str) -> bool:
        """Return True when ``name`` is a valid imported named-selection name."""
        return bool(cls._VALID_NAME_RE.match((name or "").strip()))

    @staticmethod
    def _find_header(lines: List[str]) -> Tuple[int, str]:
        for index, line in enumerate(lines):
            if "node" in line.lower() and "location" in line.lower():
                return index, line
        raise ValueError(
            "TXT file must contain a header row with 'Node Number' and "
            "X/Y/Z location columns."
        )

    @staticmethod
    def _detect_delimiter(header: str) -> str:
        for delimiter in ("\t", ",", ";"):
            if delimiter in header:
                return delimiter
        return "multi_space"

    @staticmethod
    def _split_line(line: str, delimiter: str) -> List[str]:
        if delimiter == "multi_space":
            return [part.strip() for part in re.split(r"\s{2,}", line.strip()) if part.strip()]
        return [part.strip() for part in line.split(delimiter) if part.strip()]

    @classmethod
    def _build_column_map(cls, headers: List[str]) -> Dict[str, int]:
        column_map: Dict[str, int] = {}

        for index, header in enumerate(headers):
            normalized = cls._normalize_header(header)
            if normalized in ("nodenumber", "node", "nodeid"):
                column_map["node"] = index
            elif normalized.startswith("xlocation"):
                column_map["x"] = index
                column_map["x_factor"] = cls._unit_factor_from_header(header)
            elif normalized.startswith("ylocation"):
                column_map["y"] = index
                column_map["y_factor"] = cls._unit_factor_from_header(header)
            elif normalized.startswith("zlocation"):
                column_map["z"] = index
                column_map["z_factor"] = cls._unit_factor_from_header(header)

        missing = [name for name in ("node", "x", "y", "z") if name not in column_map]
        if missing:
            raise ValueError(
                "TXT file must include Node Number and X/Y/Z Location columns."
            )

        return column_map

    @staticmethod
    def _normalize_header(header: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", header.lower())

    @classmethod
    def _unit_factor_from_header(cls, header: str) -> float:
        match = re.search(r"\(([^)]+)\)", header)
        if not match:
            raise ValueError(
                f"Location column '{header}' must include a unit in parentheses "
                "(m, cm, or mm)."
            )

        unit = match.group(1).strip().lower()
        factor = cls._UNIT_FACTORS_TO_MM.get(unit)
        if factor is None:
            raise ValueError(
                f"Unsupported location unit '{unit}' in column '{header}'. "
                "Supported units are m, cm, and mm."
            )
        return factor

    @staticmethod
    def _unique_name(name: str, used_names: set) -> str:
        if name not in used_names:
            return name

        suffix = 2
        while f"{name}_{suffix}" in used_names:
            suffix += 1
        return f"{name}_{suffix}"

    def rename_conflicting_selections(self, reserved_names: Iterable[str]) -> None:
        """Rename the imported selection so it does not shadow reserved names."""
        used_names = set(reserved_names)
        renamed: Dict[str, TXTNamedSelection] = {}

        for name, selection in self.selections.items():
            unique_name = self._unique_name(name, used_names)
            renamed[unique_name] = TXTNamedSelection(
                name=unique_name,
                location=selection.location,
                ids=list(selection.ids),
                coordinates_mm=dict(selection.coordinates_mm),
            )

        self.selections = renamed

    def get_named_selections(self) -> List[str]:
        """Return imported TXT named-selection names."""
        return list(self.selections.keys())

    def get_named_selection_locations(self) -> Dict[str, str]:
        """Return imported named-selection location metadata."""
        return {name: "nodal" for name in self.selections}

    def get_named_selection_sources(self) -> Dict[str, str]:
        """Return imported named-selection source metadata."""
        return {name: "txt" for name in self.selections}

    def has_named_selection(self, ns_name: str) -> bool:
        """Return True when this TXT import contains the named selection."""
        return ns_name in self.selections

    def get_nodal_scoping_from_named_selection(self, ns_name: str, dpf_reader):
        """Create nodal DPF scoping for an imported TXT named selection."""
        selection = self.selections.get(ns_name)
        if selection is None:
            raise ValueError(f"Named selection '{ns_name}' was not found in TXT file.")
        return dpf_reader.create_nodal_scoping_from_node_ids(selection.ids)
