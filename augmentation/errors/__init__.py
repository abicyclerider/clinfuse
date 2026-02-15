"""Error transformation modules."""

from .base_error import BaseError
from .demographic_errors import (
    AddressAbbreviation,
    ApartmentFormatVariation,
    DateOffByOne,
    MaidenNameUsage,
    NameTypo,
    NicknameSubstitution,
)
from .formatting_errors import (
    CapitalizationError,
    ExtraWhitespace,
    LeadingTrailingWhitespace,
    MissingWhitespace,
    SpecialCharacterVariation,
)
from .identifier_errors import (
    DriversLicenseError,
    PassportError,
    SSNDigitError,
    SSNFormatVariation,
    SSNTransposition,
)

__all__ = [
    "BaseError",
    # Demographic
    "NicknameSubstitution",
    "NameTypo",
    "AddressAbbreviation",
    "ApartmentFormatVariation",
    "DateOffByOne",
    "MaidenNameUsage",
    # Identifier
    "SSNTransposition",
    "SSNDigitError",
    "SSNFormatVariation",
    "DriversLicenseError",
    "PassportError",
    # Formatting
    "CapitalizationError",
    "ExtraWhitespace",
    "MissingWhitespace",
    "LeadingTrailingWhitespace",
    "SpecialCharacterVariation",
]
