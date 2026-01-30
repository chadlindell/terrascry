"""
HIRT PDF Generator Library

Shared components for generating publication-quality scientific PDFs.
"""

from .styles import (
    COLOR_PALETTE,
    get_styles,
    PAGE_WIDTH,
    PAGE_HEIGHT,
    MARGIN,
    CONTENT_WIDTH,
    PRIMARY,
    SECONDARY,
    ACCENT,
    SUCCESS,
    WARNING,
    LIGHT_BG,
    GROUND_TAN,
)

from .pdf_builder import SectionPDFBuilder

__all__ = [
    'COLOR_PALETTE',
    'get_styles',
    'PAGE_WIDTH',
    'PAGE_HEIGHT',
    'MARGIN',
    'CONTENT_WIDTH',
    'PRIMARY',
    'SECONDARY',
    'ACCENT',
    'SUCCESS',
    'WARNING',
    'LIGHT_BG',
    'GROUND_TAN',
    'SectionPDFBuilder',
]
