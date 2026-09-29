"""Validate the supplied provider contract without coercion or remote lookups."""

import json

from jsonschema import Draft202012Validator
from referencing import Registry


class PayloadValidator:
    def __init__(self, schema: dict):
        Draft202012Validator.check_schema(schema)
        self.validator = Draft202012Validator(schema, registry=Registry())

    def errors(self, payload) -> tuple[str, ...]:
        # JSON Schema alone can accept Python NaN as a number. Enforce JSON too.
        try:
            json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError):
            return ("$: response must be JSON-serializable with finite numbers",)
        return tuple(
            sorted(
                f"/{'/'.join(map(str, error.absolute_path))}: violates {error.validator}; expected {error.validator_value!r}"
                for error in self.validator.iter_errors(payload)
            )
        )
