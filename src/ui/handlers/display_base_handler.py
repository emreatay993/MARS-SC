"""
Base utilities for Display tab handler classes.
"""

from ui.handlers.display_state import DisplayState


class DisplayBaseHandler:
    """Common base class providing access to the tab widget and shared state."""

    def __init__(self, tab, state: DisplayState):
        self.tab = tab
        self.state = state

    def set_state_attr(self, attr_name: str, value) -> None:
        """
        Convenience helper to keep DisplayTab attributes in sync with state.

        Args:
            attr_name: Name of the attribute to set.
            value: Value to assign.
        """
        if not hasattr(self.state, attr_name):
            raise AttributeError(f"DisplayState has no attribute '{attr_name}'")
        setattr(self.state, attr_name, value)
        setattr(self.tab, attr_name, value)
