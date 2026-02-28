"""Tests for ImportValidator service."""

from pathlib import Path

from app.services.import_validator import ImportValidator

FIXTURES = Path(__file__).parent / "fixtures"

validator = ImportValidator()


def test_valid_survey_csv():
    content = (FIXTURES / "valid_survey.csv").read_text()
    result = validator.validate_csv(content)
    assert result.valid is True
    assert result.row_count == 20
    assert "x" in result.columns
    assert "y" in result.columns
    assert "gradient_nt" in result.columns
    assert result.errors == []


def test_valid_grid_csv():
    content = (FIXTURES / "valid_grid.csv").read_text()
    result = validator.validate_csv(content)
    assert result.valid is True
    assert result.row_count == 12
    assert result.errors == []


def test_missing_columns():
    content = (FIXTURES / "invalid_missing_cols.csv").read_text()
    result = validator.validate_csv(content)
    assert result.valid is False
    assert any("Missing required columns" in e for e in result.errors) or any(
        "Missing value column" in e for e in result.errors
    )


def test_bad_values():
    content = (FIXTURES / "invalid_bad_values.csv").read_text()
    result = validator.validate_csv(content)
    assert result.valid is False
    assert any("non-numeric" in e for e in result.errors)


def test_too_few_rows():
    content = "x,y,gradient_nt\n0.0,0.0,1.2\n1.0,0.0,1.5\n"
    result = validator.validate_csv(content)
    assert result.valid is False
    assert any("Too few data rows" in e for e in result.errors)


def test_empty_file():
    result = validator.validate_csv("")
    assert result.valid is False
    assert any("empty" in e.lower() for e in result.errors)
