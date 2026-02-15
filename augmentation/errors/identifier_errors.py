"""Identifier error transformations (SSN, driver's license, passport)."""

import re
from typing import Any, Dict, List

from .base_error import BaseError


class SSNTransposition(BaseError):
    """Transpose adjacent digits in SSN."""

    def get_applicable_fields(self) -> List[str]:
        return ["SSN"]

    def get_error_type_name(self) -> str:
        return "ssn_transposition"

    def apply(self, value: Any, context: Dict) -> Any:
        if not self.should_apply(value):
            return value

        ssn = str(value)

        # Extract digits only
        digits = re.sub(r"\D", "", ssn)

        if len(digits) < 2:
            return value

        # Choose random position to transpose
        pos = self.rng.integers(0, len(digits) - 1)

        # Transpose adjacent digits
        digits_list = list(digits)
        digits_list[pos], digits_list[pos + 1] = digits_list[pos + 1], digits_list[pos]
        new_digits = "".join(digits_list)

        # Preserve original format (with or without dashes)
        if "-" in ssn:
            # Format as XXX-XX-XXXX
            return f"{new_digits[:3]}-{new_digits[3:5]}-{new_digits[5:]}"
        else:
            return new_digits


class SSNDigitError(BaseError):
    """Single digit error in SSN."""

    def get_applicable_fields(self) -> List[str]:
        return ["SSN"]

    def get_error_type_name(self) -> str:
        return "ssn_digit_error"

    def apply(self, value: Any, context: Dict) -> Any:
        if not self.should_apply(value):
            return value

        ssn = str(value)

        # Extract digits only
        digits = re.sub(r"\D", "", ssn)

        if len(digits) < 1:
            return value

        # Choose random position
        pos = self.rng.integers(0, len(digits))

        # Replace with different random digit
        new_digit = str(self.rng.integers(0, 10))
        while new_digit == digits[pos]:
            new_digit = str(self.rng.integers(0, 10))

        new_digits = digits[:pos] + new_digit + digits[pos + 1 :]

        # Preserve original format
        if "-" in ssn:
            return f"{new_digits[:3]}-{new_digits[3:5]}-{new_digits[5:]}"
        else:
            return new_digits


class SSNFormatVariation(BaseError):
    """Vary SSN format (with/without dashes)."""

    def get_applicable_fields(self) -> List[str]:
        return ["SSN"]

    def get_error_type_name(self) -> str:
        return "ssn_format_variation"

    def apply(self, value: Any, context: Dict) -> Any:
        if not self.should_apply(value):
            return value

        ssn = str(value)

        # Extract digits only
        digits = re.sub(r"\D", "", ssn)

        if len(digits) != 9:
            return value

        # Toggle format
        if "-" in ssn:
            # Remove dashes
            return digits
        else:
            # Add dashes
            return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"


class DriversLicenseError(BaseError):
    """Introduce errors in driver's license number."""

    def get_applicable_fields(self) -> List[str]:
        return ["DRIVERS"]

    def get_error_type_name(self) -> str:
        return "drivers_license_error"

    def apply(self, value: Any, context: Dict) -> Any:
        if not self.should_apply(value):
            return value

        license_num = str(value)

        if len(license_num) < 2:
            return value

        # Choose random position
        pos = self.rng.integers(0, len(license_num))

        # Replace character
        if license_num[pos].isdigit():
            # Replace digit with different digit
            new_char = str(self.rng.integers(0, 10))
            while new_char == license_num[pos]:
                new_char = str(self.rng.integers(0, 10))
        elif license_num[pos].isalpha():
            # Replace letter with different letter
            new_char = self._select_random_character(exclude=license_num[pos])
        else:
            # Non-alphanumeric, return original
            return value

        return license_num[:pos] + new_char + license_num[pos + 1 :]


class PassportError(BaseError):
    """Introduce errors in passport number."""

    def get_applicable_fields(self) -> List[str]:
        return ["PASSPORT"]

    def get_error_type_name(self) -> str:
        return "passport_error"

    def apply(self, value: Any, context: Dict) -> Any:
        if not self.should_apply(value):
            return value

        passport = str(value)

        if len(passport) < 2:
            return value

        # Choose random position
        pos = self.rng.integers(0, len(passport))

        # Replace character (similar to driver's license logic)
        if passport[pos].isdigit():
            new_char = str(self.rng.integers(0, 10))
            while new_char == passport[pos]:
                new_char = str(self.rng.integers(0, 10))
        elif passport[pos].isalpha():
            new_char = self._select_random_character(exclude=passport[pos])
        else:
            return value

        return passport[:pos] + new_char + passport[pos + 1 :]
