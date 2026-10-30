
"""Utilities for evaluating frozen-meat packaging defect detection.

This module provides input validation and precision, recall and F1
calculations for object-detection evaluation.
"""

def calculate_metrics(tp: int, fp: int, fn: int) -> dict:
    """Calculate precision, recall and F1 score.

    Args:
        tp: Number of true-positive detections.
        fp: Number of false-positive detections.
        fn: Number of false-negative detections.

    Returns:
        Dictionary containing precision, recall and F1 score.

    Raises:
        ValueError: If a count is negative or not an integer.
    """
    for name, value in {"tp": tp, "fp": fp, "fn": fn}.items():
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
