"""ImportValidator — validates CSV files before import ingestion."""

import csv
import io
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of CSV validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    row_count: int = 0
    columns: list[str] = field(default_factory=list)


REQUIRED_COLUMNS = {"x", "y"}
VALUE_COLUMNS = {"gradient_nt", "value"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


class ImportValidator:
    """Validates CSV content for survey data import."""

    def validate_csv(self, content: str) -> ValidationResult:
        """Validate CSV content. Returns a ValidationResult."""
        if not content or not content.strip():
            return ValidationResult(valid=False, errors=["File is empty"])

        # Strip comment lines (metadata headers starting with #)
        lines = content.strip().splitlines()
        data_lines = [line for line in lines if not line.startswith("#")]

        if not data_lines:
            return ValidationResult(valid=False, errors=["No data rows found"])

        try:
            reader = csv.DictReader(io.StringIO("\n".join(data_lines)))
            if reader.fieldnames is None:
                return ValidationResult(valid=False, errors=["No header row found"])

            columns = [c.strip() for c in reader.fieldnames]
        except csv.Error as e:
            return ValidationResult(valid=False, errors=[f"CSV parse error: {e}"])

        errors: list[str] = []

        # Check required columns
        col_set = set(columns)
        missing_required = REQUIRED_COLUMNS - col_set
        if missing_required:
            errors.append(f"Missing required columns: {', '.join(sorted(missing_required))}")

        has_value_col = bool(col_set & VALUE_COLUMNS)
        if not has_value_col:
            errors.append(
                f"Missing value column: need one of {', '.join(sorted(VALUE_COLUMNS))}"
            )

        if errors:
            return ValidationResult(valid=False, errors=errors, columns=columns)

        # Validate data rows
        row_count = 0
        for i, row in enumerate(reader, start=2):  # line 2 is first data row
            row_count += 1
            for col in list(REQUIRED_COLUMNS) + list(col_set & VALUE_COLUMNS):
                val = row.get(col, "")
                if val is None or val.strip() == "":
                    errors.append(f"Row {i}: empty value in column '{col}'")
                    continue
                try:
                    float(val)
                except ValueError:
                    errors.append(f"Row {i}: non-numeric value '{val}' in column '{col}'")

            # Cap error messages to avoid huge reports
            if len(errors) > 20:
                errors.append("... additional errors truncated")
                break

        if row_count < 3:
            errors.append(f"Too few data rows: {row_count} (minimum 3 required)")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            row_count=row_count,
            columns=columns,
        )
