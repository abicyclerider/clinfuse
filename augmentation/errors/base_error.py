"""Base class for error transformations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class BaseError(ABC):
    """Abstract base class for demographic error transformations."""

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize error transformer.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(random_seed)

    @abstractmethod
    def apply(self, value: Any, context: Dict) -> Any:
        """
        Apply error transformation to a value.

        Args:
            value: Original value to transform
            context: Additional context (e.g., full patient record, field name)

        Returns:
            Transformed value with error applied
        """
        pass

    @abstractmethod
    def get_applicable_fields(self) -> List[str]:
        """
        Return list of CSV field names this error applies to.

        Returns:
            List of field names (e.g., ["FIRST", "LAST"])
        """
        pass

    @abstractmethod
    def get_error_type_name(self) -> str:
        """
        Return human-readable error type name for logging.

        Returns:
            Error type name (e.g., "nickname_substitution")
        """
        pass

    def should_apply(self, value: Any) -> bool:
        """
        Check if error should be applied to this value.

        Override this method to add custom logic for when to apply errors.
        For example, don't apply to null values, or only apply to certain formats.

        Args:
            value: Value to check

        Returns:
            True if error should be applied
        """
        # By default, don't apply to null/empty values
        if value is None or (isinstance(value, str) and not value.strip()):
            return False
        return True

    def _select_random_character(self, exclude: str = "") -> str:
        """
        Select a random alphanumeric character.

        Args:
            exclude: Characters to exclude from selection

        Returns:
            Random character
        """
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        chars = "".join(c for c in chars if c not in exclude)
        return self.rng.choice(list(chars))

    def _get_keyboard_adjacent(self, char: str) -> str:
        """
        Get a keyboard-adjacent character for typo simulation.

        Args:
            char: Original character

        Returns:
            Adjacent character on QWERTY keyboard
        """
        # QWERTY keyboard adjacency map
        adjacency = {
            "Q": ["W", "A"],
            "W": ["Q", "E", "S"],
            "E": ["W", "R", "D"],
            "R": ["E", "T", "F"],
            "T": ["R", "Y", "G"],
            "Y": ["T", "U", "H"],
            "U": ["Y", "I", "J"],
            "I": ["U", "O", "K"],
            "O": ["I", "P", "L"],
            "P": ["O", "L"],
            "A": ["Q", "S", "Z"],
            "S": ["W", "A", "D", "Z", "X"],
            "D": ["E", "S", "F", "X", "C"],
            "F": ["R", "D", "G", "C", "V"],
            "G": ["T", "F", "H", "V", "B"],
            "H": ["Y", "G", "J", "B", "N"],
            "J": ["U", "H", "K", "N", "M"],
            "K": ["I", "J", "L", "M"],
            "L": ["O", "K", "P"],
            "Z": ["A", "S", "X"],
            "X": ["S", "D", "Z", "C"],
            "C": ["D", "F", "X", "V"],
            "V": ["F", "G", "C", "B"],
            "B": ["G", "H", "V", "N"],
            "N": ["H", "J", "B", "M"],
            "M": ["J", "K", "N"],
        }

        char_upper = char.upper()
        if char_upper in adjacency and adjacency[char_upper]:
            adjacent = self.rng.choice(adjacency[char_upper])
            return adjacent.lower() if char.islower() else adjacent

        return char  # Return original if no adjacency found
