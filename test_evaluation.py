
"""Unit tests for the evaluation utility."""

import pytest
from src.evaluation import calculate_metrics


def test_perfect_detection():
    """Perfect detection should produce scores of one."""
    result = calculate_metrics(10, 0, 0)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_no_detections():
    """Zero counts should be handled without division errors."""
    result = calculate_metrics(0, 0, 0)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_known_values():
    """Check metrics against manually calculated values."""
    result = calculate_metrics(8, 2, 2)
    assert result["precision"] == pytest.approx(0.8)
    assert result["recall"] == pytest.approx(0.8)
    assert result["f1"] == pytest.approx(0.8)


def test_invalid_counts():
    """Negative detection counts must be rejected."""
    with pytest.raises(ValueError):
        calculate_metrics(-1, 2, 3)
