"""
Named-selection/component reader for ANSYS CDB files.

Mechanical exports named selections to CDB as APDL component blocks, commonly
``CMBLOCK`` entries. This reader extracts node and element components so they
can be used as solver scoping when the RST metadata does not expose named
selections reliably.
"""

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional


@dataclass
class CDBNamedSelection:
    """One named selection/component parsed from a CDB file."""

    name: str
    location: str
    ids: List[int]


class CDBNamedSelectionReader:
    """Parse named selections from APDL CDB ``CMBLOCK`` entries."""

    _SUPPORTED_LOCATIONS = {
        "NODE": "nodal",
        "NODES": "nodal",
        "ELEM": "elemental",
        "ELEMENT": "elemental",
        "ELEMENTS": "elemental",
    }

    def __init__(self, cdb_path: str, selections: Dict[str, CDBNamedSelection]):
        self.cdb_path = cdb_path
        self.selections = selections

    @classmethod
    def from_file(cls, cdb_path: str) -> "CDBNamedSelectionReader":
        """Parse supported named selections from a CDB file."""
        try:
            with open(cdb_path, "r", encoding="utf-8-sig", errors="ignore") as handle:
                lines = handle.readlines()
        except OSError as exc:
            raise ValueError(f"Failed to read CDB file: {exc}") from exc

        raw_selections: Dict[str, CDBNamedSelection] = {}
        index = 0

        while index < len(lines):
            line = cls._strip_comment(lines[index])
            if not line.upper().startswith("CMBLOCK"):
                index += 1
                continue

            header = cls._parse_cmblock_header(line)
            if header is None:
                index += 1
                continue

            name, location, expected_count = header
            index += 1

            if index < len(lines) and cls._strip_comment(lines[index]).startswith("("):
                index += 1

            ids: List[int] = []
            while index < len(lines) and (expected_count is None or len(ids) < expected_count):
                data_line = cls._strip_comment(lines[index])
                upper_line = data_line.upper()
                if upper_line.startswith("CMBLOCK"):
                    break

                ids.extend(cls._extract_ints(data_line))
                index += 1

            if expected_count is not None:
                ids = ids[:expected_count]

            if ids:
                cls._store_selection(raw_selections, name, location, ids)

        return cls(cdb_path=cdb_path, selections=raw_selections)

    @staticmethod
    def _strip_comment(line: str) -> str:
        """Remove APDL comments and surrounding whitespace."""
        return line.split("!", 1)[0].strip()

    @classmethod
    def _parse_cmblock_header(cls, line: str) -> Optional[tuple]:
        """Parse ``CMBLOCK,name,entity,count`` headers."""
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or parts[0].upper() != "CMBLOCK":
            return None

        name = cls._sanitize_selection_name(parts[1])
        if not name:
            return None

        location = cls._SUPPORTED_LOCATIONS.get(parts[2].upper())
        if location is None:
            return None

        expected_count = None
        if len(parts) >= 4 and parts[3]:
            try:
                expected_count = int(float(parts[3]))
            except ValueError:
                expected_count = None

        return name, location, expected_count

    @staticmethod
    def _extract_ints(line: str) -> List[int]:
        """Extract integer IDs from free- or fixed-format CDB data lines."""
        return [int(match) for match in re.findall(r"[-+]?\d+", line)]

    @staticmethod
    def _store_selection(
        selections: Dict[str, CDBNamedSelection],
        name: str,
        location: str,
        ids: List[int],
    ) -> None:
        """Store or merge a parsed component while preserving ID order."""
        existing = selections.get(name)
        if existing is None or existing.location != location:
            seen = set()
            unique_ids = []
            for item_id in ids:
                if item_id not in seen:
                    seen.add(item_id)
                    unique_ids.append(item_id)
            selections[name] = CDBNamedSelection(name=name, location=location, ids=unique_ids)
            return

        seen = set(existing.ids)
        for item_id in ids:
            if item_id not in seen:
                seen.add(item_id)
                existing.ids.append(item_id)

    @staticmethod
    def _sanitize_selection_name(name: str) -> str:
        """Return a UI/DPF-friendly name using only letters, digits, and underscores."""
        sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", name)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        if not sanitized:
            sanitized = "CDB"
        if sanitized[0].isdigit():
            sanitized = f"NS_{sanitized}"
        return sanitized

    @staticmethod
    def _unique_name(name: str, used_names: set) -> str:
        """Return name or name_N when the preferred alias is already used."""
        if name not in used_names:
            return name

        suffix = 2
        while f"{name}_{suffix}" in used_names:
            suffix += 1
        return f"{name}_{suffix}"

    def rename_conflicting_selections(self, reserved_names: Iterable[str]) -> None:
        """Rename CDB names in-place so they do not shadow reserved names."""
        used_names = set(reserved_names)
        renamed: Dict[str, CDBNamedSelection] = {}

        for name, selection in self.selections.items():
            unique_name = self._unique_name(name, used_names)
            used_names.add(unique_name)
            renamed[unique_name] = CDBNamedSelection(
                name=unique_name,
                location=selection.location,
                ids=list(selection.ids),
            )

        self.selections = renamed

    def get_named_selections(self) -> List[str]:
        """Return CMBLOCK named-selection names in file order."""
        return list(self.selections.keys())

    def get_named_selection_locations(self) -> Dict[str, str]:
        """Return component location metadata."""
        return {
            name: selection.location
            for name, selection in self.selections.items()
        }

    def get_named_selection_sources(self) -> Dict[str, str]:
        """Return component source metadata."""
        return {
            name: "cdb"
            for name in self.selections
        }

    def has_named_selection(self, ns_name: str) -> bool:
        """Return True when the CDB contains this component."""
        return ns_name in self.selections

    def get_nodal_scoping_from_named_selection(self, ns_name: str, dpf_reader):
        """Convert a CDB node or element component to nodal DPF scoping."""
        selection = self.selections.get(ns_name)
        if selection is None:
            raise ValueError(f"Named selection '{ns_name}' was not found in CDB file.")

        if selection.location == "nodal":
            return dpf_reader.create_nodal_scoping_from_node_ids(selection.ids)
        if selection.location == "elemental":
            return dpf_reader.create_nodal_scoping_from_element_ids(selection.ids)

        raise ValueError(
            f"Named selection '{ns_name}' has unsupported CDB location "
            f"'{selection.location}'."
        )
