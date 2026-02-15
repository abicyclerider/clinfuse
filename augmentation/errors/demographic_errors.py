"""Demographic error transformations (names, addresses, dates)."""

import re
from datetime import timedelta
from typing import Any, Dict, List

from .base_error import BaseError


class NicknameSubstitution(BaseError):
    """Substitute formal names with common nicknames."""

    # Common nickname mappings
    NICKNAME_MAP = {
        "WILLIAM": "BILL",
        "ROBERT": "BOB",
        "RICHARD": "DICK",
        "JAMES": "JIM",
        "JOHN": "JACK",
        "MICHAEL": "MIKE",
        "DAVID": "DAVE",
        "JOSEPH": "JOE",
        "THOMAS": "TOM",
        "CHARLES": "CHUCK",
        "CHRISTOPHER": "CHRIS",
        "DANIEL": "DAN",
        "MATTHEW": "MATT",
        "ANTHONY": "TONY",
        "DONALD": "DON",
        "KENNETH": "KEN",
        "STEVEN": "STEVE",
        "EDWARD": "ED",
        "TIMOTHY": "TIM",
        "JEFFREY": "JEFF",
        "NICHOLAS": "NICK",
        "JONATHAN": "JON",
        "BENJAMIN": "BEN",
        "SAMUEL": "SAM",
        "ALEXANDER": "ALEX",
        "ANDREW": "ANDY",
        "JOSHUA": "JOSH",
        "ELIZABETH": "LIZ",
        "PATRICIA": "PAT",
        "JENNIFER": "JENNY",
        "LINDA": "LYNN",
        "BARBARA": "BARB",
        "SUSAN": "SUE",
        "JESSICA": "JESS",
        "MARGARET": "PEGGY",
        "SARAH": "SALLY",
        "KIMBERLY": "KIM",
        "DEBORAH": "DEB",
        "REBECCA": "BECKY",
        "STEPHANIE": "STEPH",
        "CATHERINE": "CATHY",
        "CHRISTINE": "CHRIS",
        "SAMANTHA": "SAM",
        "AMANDA": "MANDY",
        "MELISSA": "MEL",
        "MICHELLE": "SHELLY",
        "KATHLEEN": "KATHY",
        "DOROTHY": "DOT",
    }

    def get_applicable_fields(self) -> List[str]:
        return ["FIRST"]

    def get_error_type_name(self) -> str:
        return "nickname_substitution"

    def apply(self, value: Any, context: Dict) -> Any:
        if not self.should_apply(value):
            return value

        name_upper = str(value).upper().strip()

        # Numbers already stripped at CSV load time, but keep this line for safety
        name_clean = re.sub(r"\d+", "", name_upper).strip()

        if name_clean in self.NICKNAME_MAP:
            return self.NICKNAME_MAP[name_clean]

        return value


class NameTypo(BaseError):
    """Introduce typos in names (keyboard adjacency or character substitution)."""

    def get_applicable_fields(self) -> List[str]:
        return ["FIRST", "LAST", "MAIDEN"]

    def get_error_type_name(self) -> str:
        return "name_typo"

    def apply(self, value: Any, context: Dict) -> Any:
        if not self.should_apply(value):
            return value

        name = str(value)
        if len(name) < 2:
            return value

        # Choose a random position (avoid first character to maintain recognizability)
        pos = self.rng.integers(1, len(name))

        # 50/50: keyboard adjacent or random substitution
        if self.rng.random() < 0.5:
            # Keyboard adjacent
            new_char = self._get_keyboard_adjacent(name[pos])
        else:
            # Random character
            new_char = self._select_random_character(exclude=name[pos])

        return name[:pos] + new_char + name[pos + 1 :]


class AddressAbbreviation(BaseError):
    """Vary address abbreviations (Street vs St, Avenue vs Ave)."""

    ABBREVIATIONS = {
        "STREET": ["ST", "ST.", "STR"],
        "AVENUE": ["AVE", "AVE.", "AV"],
        "ROAD": ["RD", "RD."],
        "DRIVE": ["DR", "DR."],
        "LANE": ["LN", "LN."],
        "COURT": ["CT", "CT."],
        "PLACE": ["PL", "PL."],
        "BOULEVARD": ["BLVD", "BLVD.", "BLV"],
        "PARKWAY": ["PKWY", "PKWY.", "PKY"],
        "CIRCLE": ["CIR", "CIR."],
        "TERRACE": ["TER", "TER.", "TERR"],
        "APARTMENT": ["APT", "APT.", "#", "UNIT"],
    }

    def get_applicable_fields(self) -> List[str]:
        return ["ADDRESS"]

    def get_error_type_name(self) -> str:
        return "address_abbreviation"

    def apply(self, value: Any, context: Dict) -> Any:
        if not self.should_apply(value):
            return value

        address = str(value).upper()

        # Try to find and replace street types
        for full_form, abbrevs in self.ABBREVIATIONS.items():
            if full_form in address:
                replacement = self.rng.choice(abbrevs)
                return address.replace(full_form, replacement)

            # Also check if abbreviated form exists and expand/change it
            for abbrev in abbrevs:
                if abbrev in address:
                    # 50% chance to expand, 50% to use different abbreviation
                    if self.rng.random() < 0.5:
                        return address.replace(abbrev, full_form)
                    else:
                        other_abbrevs = [a for a in abbrevs if a != abbrev]
                        if other_abbrevs:
                            replacement = self.rng.choice(other_abbrevs)
                            return address.replace(abbrev, replacement)

        return value


class ApartmentFormatVariation(BaseError):
    """Vary apartment/unit format (Apt 5 vs Unit 5 vs #5)."""

    APT_PATTERNS = [
        (
            r"APT\.?\s*(\d+)",
            ["APARTMENT {}", "APT {}", "APT. {}", "UNIT {}", "#{}", "NO. {}"],
        ),
        (r"APARTMENT\s*(\d+)", ["APT {}", "APT. {}", "UNIT {}", "#{}", "NO. {}"]),
        (r"UNIT\s*(\d+)", ["APT {}", "APT. {}", "APARTMENT {}", "#{}", "NO. {}"]),
        (r"#(\d+)", ["APT {}", "APT. {}", "UNIT {}", "APARTMENT {}", "NO. {}"]),
    ]

    def get_applicable_fields(self) -> List[str]:
        return ["ADDRESS"]

    def get_error_type_name(self) -> str:
        return "apartment_format_variation"

    def apply(self, value: Any, context: Dict) -> Any:
        if not self.should_apply(value):
            return value

        address = str(value).upper()

        for pattern, formats in self.APT_PATTERNS:
            match = re.search(pattern, address)
            if match:
                apt_number = match.group(1)
                new_format = self.rng.choice(formats).format(apt_number)
                return re.sub(pattern, new_format, address)

        return value


class DateOffByOne(BaseError):
    """Introduce off-by-one errors in dates (day, month, or year)."""

    def get_applicable_fields(self) -> List[str]:
        return ["BIRTHDATE"]

    def get_error_type_name(self) -> str:
        return "date_off_by_one"

    def apply(self, value: Any, context: Dict) -> Any:
        if not self.should_apply(value):
            return value

        # Value should be a datetime object (parsed by csv_handler)
        if not hasattr(value, "year"):
            return value

        # Choose which component to modify
        choice = self.rng.choice(["day", "month", "year"])

        try:
            if choice == "day":
                # Off by 1 day
                delta = timedelta(days=int(self.rng.choice([-1, 1])))
                return value + delta
            elif choice == "month":
                # Off by 1 month (approximate with 30 days)
                delta = timedelta(days=int(self.rng.choice([-30, 30])))
                return value + delta
            elif choice == "year":
                # Off by 1 year
                delta = timedelta(days=int(self.rng.choice([-365, 365])))
                return value + delta
        except (ValueError, OverflowError):
            # If date operation fails, return original
            return value

        return value


class MaidenNameUsage(BaseError):
    """Use maiden name instead of last name (or vice versa)."""

    def get_applicable_fields(self) -> List[str]:
        return ["LAST"]

    def get_error_type_name(self) -> str:
        return "maiden_name_usage"

    def apply(self, value: Any, context: Dict) -> Any:
        if not self.should_apply(value):
            return value

        # Check if maiden name exists in context
        patient_record = context.get("patient_record", {})
        maiden_name = patient_record.get("MAIDEN")

        if maiden_name and isinstance(maiden_name, str) and maiden_name.strip():
            # Swap last name with maiden name
            return maiden_name

        return value
