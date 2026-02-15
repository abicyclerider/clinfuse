"""Formatting error transformations (capitalization, whitespace)."""

import re
from typing import Any, Dict, List

from .base_error import BaseError


class CapitalizationError(BaseError):
    """Vary capitalization (ALLCAPS, lowercase, Title Case)."""

    def get_applicable_fields(self) -> List[str]:
        return ["FIRST", "LAST", "MAIDEN", "ADDRESS", "CITY"]

    def get_error_type_name(self) -> str:
        return "capitalization_error"

    def apply(self, value: Any, context: Dict) -> Any:
        if not self.should_apply(value):
            return value

        text = str(value)

        # Choose capitalization style
        style = self.rng.choice(["upper", "lower", "title", "mixed"])

        if style == "upper":
            return text.upper()
        elif style == "lower":
            return text.lower()
        elif style == "title":
            return text.title()
        elif style == "mixed":
            # Random capitalization of each character
            return "".join(
                c.upper() if self.rng.random() < 0.5 else c.lower() for c in text
            )

        return value


class ExtraWhitespace(BaseError):
    """Add extra whitespace within text."""

    def get_applicable_fields(self) -> List[str]:
        return ["FIRST", "LAST", "MAIDEN", "ADDRESS"]

    def get_error_type_name(self) -> str:
        return "extra_whitespace"

    def apply(self, value: Any, context: Dict) -> Any:
        if not self.should_apply(value):
            return value

        text = str(value)

        # Add extra spaces at random positions
        words = text.split()
        if len(words) < 2:
            return value

        # Join with multiple spaces
        num_spaces = self.rng.integers(2, 5)
        return (" " * num_spaces).join(words)


class MissingWhitespace(BaseError):
    """Remove whitespace between words."""

    def get_applicable_fields(self) -> List[str]:
        return ["ADDRESS"]

    def get_error_type_name(self) -> str:
        return "missing_whitespace"

    def apply(self, value: Any, context: Dict) -> Any:
        if not self.should_apply(value):
            return value

        text = str(value)

        # Remove all spaces
        return re.sub(r"\s+", "", text)


class LeadingTrailingWhitespace(BaseError):
    """Add leading or trailing whitespace."""

    def get_applicable_fields(self) -> List[str]:
        return ["FIRST", "LAST", "MAIDEN", "ADDRESS", "CITY", "SSN"]

    def get_error_type_name(self) -> str:
        return "leading_trailing_whitespace"

    def apply(self, value: Any, context: Dict) -> Any:
        if not self.should_apply(value):
            return value

        text = str(value)

        # Choose where to add whitespace
        choice = self.rng.choice(["leading", "trailing", "both"])

        num_spaces = self.rng.integers(1, 4)
        spaces = " " * num_spaces

        if choice == "leading":
            return spaces + text
        elif choice == "trailing":
            return text + spaces
        elif choice == "both":
            return spaces + text + spaces

        return value


class SpecialCharacterVariation(BaseError):
    """Vary special characters (periods, hyphens, apostrophes)."""

    def get_applicable_fields(self) -> List[str]:
        return ["FIRST", "LAST", "MAIDEN"]

    def get_error_type_name(self) -> str:
        return "special_character_variation"

    def apply(self, value: Any, context: Dict) -> Any:
        if not self.should_apply(value):
            return value

        text = str(value)

        # Add or remove common special characters
        if self.rng.random() < 0.5:
            # Add period (e.g., "John" -> "John.")
            if "." not in text:
                return text + "."
        else:
            # Remove special characters
            return re.sub(r"[.\-\']", "", text)

        return value
