"""Custom Presidio anonymization operators and registration."""

import inspect
import random
import re
from datetime import datetime, timedelta
from typing import Any

import dateparser
from faker import Faker
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.operators import Operator, OperatorType


class FakePhoneOperator(Operator):
    """Replace a phone number with a fake Chilean mobile number."""

    def __init__(self) -> None:
        self.fake = Faker("es_CL")

    def operate(
        self,
        text: str,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Generate a fake nine-digit Chilean mobile number."""
        self.fake.seed_instance(hash(text))
        return f"9{self.fake.numerify('########')}"

    def validate(self, params: dict[str, Any] | None = None) -> None:
        """No parameters are required."""

    def operator_name(self) -> str:
        return "fake_phone_number"

    def operator_type(self) -> OperatorType:
        return OperatorType.Anonymize


class FakeRutOperator(Operator):
    """Replace a RUT with a fictional, checksum-valid RUT."""

    def __init__(self) -> None:
        self.fake = Faker("es_CL")

    def operate(
        self,
        text: str,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Generate a fictional RUT with a valid modulo-11 check digit."""
        self.fake.seed_instance(hash(text))
        body = self.fake.random_int(min=10_000_000, max=24_999_999)
        check_digit = self._calculate_check_digit(str(body))

        return f"{body}-{check_digit}"

    def validate(self, params: dict[str, Any] | None = None) -> None:
        """No parameters are required."""

    def operator_name(self) -> str:
        return "fake_rut"

    def operator_type(self) -> OperatorType:
        return OperatorType.Anonymize

    @staticmethod
    def _calculate_check_digit(rut_body: str) -> str:
        """Calculate the Chilean RUT modulo-11 verification digit."""
        total = 0
        multiplier = 2

        for digit in reversed(rut_body):
            total += int(digit) * multiplier
            multiplier = 2 if multiplier == 7 else multiplier + 1

        remainder = 11 - (total % 11)

        if remainder == 11:
            return "0"

        if remainder == 10:
            return "K"

        return str(remainder)


class ShiftSpanishDateOperator(Operator):
    """Parse Spanish DATE_TIME expressions and shift them by a number of days."""

    MONTHS = (
        "enero",
        "febrero",
        "marzo",
        "abril",
        "mayo",
        "junio",
        "julio",
        "agosto",
        "septiembre",
        "octubre",
        "noviembre",
        "diciembre",
    )

    RELATIVE_DATE_PATTERN = re.compile(
        r"\b(?:"
        r"hoy|mañana|manana|ayer|anteayer|"
        r"pasado mañana|pasado manana|"
        r"la semana pasada|"
        r"la próxima semana|la proxima semana|"
        r"el próximo lunes|el proximo lunes|"
        r"el próximo martes|el proximo martes|"
        r"el próximo miércoles|el proximo miércoles|"
        r"el próximo jueves|el proximo jueves|"
        r"el próximo viernes|el proximo viernes|"
        r"el próximo sábado|el proximo sábado|"
        r"el próximo domingo|el proximo domingo|"
        r"hace \d+ (?:día|dias|días|semana|semanas|mes|meses|año|años)"
        r")\b",
        re.IGNORECASE,
    )

    TIME_PATTERN = re.compile(
        r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b",
    )

    def operate(
        self,
        text: str,
        params: dict[str, Any] | None = None,
    ) -> str:
        """
        Parse a Spanish date expression and apply temporal noise.

        Optional params:
        - days_shift: Integer shift in days. Use the same value for every
          DATE_TIME entity in a document to preserve chronology.
        - reference_date: ISO-8601 datetime used to resolve expressions such
          as "mañana". Example: "2026-10-02T00:00:00".
        """
        params = params or {}

        random.seed(hash(text))

        days_shift = int(
            params.get(
                "days_shift",
                random.randint(-365, 365),
            )
        )
        
        reference_date = self._get_reference_date(
            params.get("reference_date"),
        )

        parsed_date = dateparser.parse(
            text,
            languages=["es"],
            settings={
                "RELATIVE_BASE": reference_date,
                "DATE_ORDER": "DMY",
                "PREFER_DAY_OF_MONTH": "first",
                "PREFER_DATES_FROM": "past",
                "RETURN_AS_TIMEZONE_AWARE": False,
            },
        )

        if parsed_date is None:
            return "<FECHA>"

        shifted_date = parsed_date + timedelta(days=days_shift)

        return self._format_like_original(
            original=text,
            date_value=shifted_date,
        )

    def validate(self, params: dict[str, Any] | None = None) -> None:
        """Validate optional operator parameters."""
        if not params:
            return

        if "days_shift" in params:
            int(params["days_shift"])

        if "reference_date" in params:
            datetime.fromisoformat(str(params["reference_date"]))

    def operator_name(self) -> str:
        return "shift_date"

    def operator_type(self) -> OperatorType:
        return OperatorType.Anonymize

    @staticmethod
    def _get_reference_date(reference_date: Any) -> datetime:
        """Return an explicit or current reference date."""
        if reference_date:
            return datetime.fromisoformat(str(reference_date))

        return datetime.now()

    @classmethod
    def _format_like_original(
        cls,
        original: str,
        date_value: datetime,
    ) -> str:
        """Format the shifted value similarly to the original expression."""
        original_lower = original.lower().strip()

        has_time = bool(cls.TIME_PATTERN.search(original_lower))
        has_year = bool(re.search(r"\b(?:19|20)\d{2}\b", original_lower))

        is_time_only = bool(
            cls.TIME_PATTERN.fullmatch(original_lower)
            or re.fullmatch(
                r"(?:a\s+las\s+)?(?:[01]?\d|2[0-3]):[0-5]\d",
                original_lower,
            )
        )

        if is_time_only:
            return date_value.strftime("%H:%M")

        # Replace relative expressions with an explicit date. This prevents
        # retaining the original relationship, such as "mañana" or "ayer".
        if cls.RELATIVE_DATE_PATTERN.search(original_lower):
            result = cls._natural_date(date_value, include_year=True)
            return cls._append_time(result, date_value, has_time)

        # Example: 26/11/2024, 26-11-2024, 26.11.2024.
        numeric_date = re.search(
            r"\b\d{1,2}([/.\-])\d{1,2}\1\d{2,4}\b",
            original,
        )

        if numeric_date:
            separator = numeric_date.group(1)

            if has_year:
                result = date_value.strftime(
                    f"%d{separator}%m{separator}%Y"
                )
            else:
                result = date_value.strftime(f"%d{separator}%m")

            return cls._append_time(result, date_value, has_time)

        # Example: 2024-11-26.
        if re.search(
            r"\b(?:19|20)\d{2}-\d{1,2}-\d{1,2}\b",
            original,
        ):
            result = date_value.strftime("%Y-%m-%d")
            return cls._append_time(result, date_value, has_time)

        result = cls._natural_date(
            date_value,
            include_year=has_year,
        )

        return cls._append_time(result, date_value, has_time)

    @classmethod
    def _natural_date(
        cls,
        date_value: datetime,
        include_year: bool,
    ) -> str:
        """Format a datetime as a natural Spanish date."""
        month = cls.MONTHS[date_value.month - 1]

        if include_year:
            return f"{date_value.day} de {month} de {date_value.year}"

        return f"{date_value.day} de {month}"

    @staticmethod
    def _append_time(
        date_text: str,
        date_value: datetime,
        has_time: bool,
    ) -> str:
        """Append time only when the original expression had a time."""
        if not has_time:
            return date_text

        return f"{date_text} a las {date_value:%H:%M}"


def register_anonymizers(engine: AnonymizerEngine) -> None:
    """Find and register every custom Operator declared in this module."""
    current_module = __import__(__name__, fromlist=["*"])

    for _, operator_class in inspect.getmembers(
        current_module,
        inspect.isclass,
    ):
        if (
            issubclass(operator_class, Operator)
            and operator_class is not Operator
            and operator_class.__module__ == __name__
        ):
            engine.add_anonymizer(operator_class)