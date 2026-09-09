"""Automated technical quality checks for wearable sensor data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FLATLINE_LIMIT_SECONDS = {
    "HR": 60,
    "EDA": 60,
    "ACC": 30,
}

MIN_COMPLETE_COVERAGE_PERCENT = 95.0
MAX_START_END_DIFFERENCE_SECONDS = 15.0


def make_result(
    subject_id: str,
    session_id: str,
    signal: str,
    check_name: str,
    severity: str,
    detected: bool,
    metric_value: Any,
    threshold: Any,
    message: str,
) -> dict:
    """Return one consistently structured quality-check result."""
    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "signal": signal,
        "check_name": check_name,
        "severity": severity,
        "detected": detected,
        "metric_value": metric_value,
        "threshold": threshold,
        "message": message,
    }


def longest_flatline_seconds(
    dataframe: pd.DataFrame,
    value_columns: list[str],
    sampling_rate: float,
) -> float:
    """Calculate the longest run of unchanged consecutive observations."""
    values = dataframe[value_columns]

    changed = values.ne(values.shift()).any(axis=1)
    run_groups = changed.cumsum()
    longest_run = int(run_groups.value_counts().max())

    return longest_run / sampling_rate


def detect_repeated_recording(
    dataframe: pd.DataFrame,
    value_columns: list[str],
    sampling_rate: float,
) -> tuple[bool, float]:
    """
    Detect whether a substantial recording was appended for a second time.

    The algorithm searches for a second occurrence of the recording's
    opening 10-second sequence, then measures the exact match percentage.
    """
    values = dataframe[value_columns].to_numpy()
    row_hashes = pd.util.hash_pandas_object(
        dataframe[value_columns],
        index=False,
    ).to_numpy()

    minimum_probe = int(sampling_rate * 10)

    if len(values) < minimum_probe * 4:
        return False, 0.0

    search_start = int(len(values) * 0.30)
    search_end = int(len(values) * 0.70)

    candidate_positions = np.flatnonzero(
        row_hashes[search_start:search_end] == row_hashes[0]
    ) + search_start

    for position in candidate_positions:
        available_length = min(position, len(values) - position)

        if available_length < minimum_probe:
            continue

        if not np.array_equal(
            values[:minimum_probe],
            values[position : position + minimum_probe],
            equal_nan=True,
        ):
            continue

        first_section = values[:available_length]
        repeated_section = values[
            position : position + available_length
        ]

        matching_rows = np.all(
            np.isclose(
                first_section,
                repeated_section,
                equal_nan=True,
            ),
            axis=1,
        )

        match_percent = float(matching_rows.mean() * 100)

        if match_percent >= 95:
            repeated_seconds = available_length / sampling_rate
            return True, repeated_seconds

    return False, 0.0


def run_raw_quality_checks(
    dataframe: pd.DataFrame,
    signal_name: str,
    audit_record: dict,
) -> list[dict]:
    """Run checks directly on one reconstructed raw signal."""
    subject_id = audit_record["subject_id"]
    session_id = audit_record["session_id"]
    sampling_rate = audit_record["observed_sampling_rate_hz"]

    value_columns = [
        column
        for column in dataframe.columns
        if column != "timestamp"
    ]

    results = []

    rate_invalid = not audit_record["sampling_rate_valid"]

    results.append(
        make_result(
            subject_id,
            session_id,
            signal_name,
            "sampling_rate",
            "FAIL",
            rate_invalid,
            sampling_rate,
            audit_record["expected_sampling_rate_hz"],
            (
                "Observed sampling rate does not match expectation"
                if rate_invalid
                else "Sampling rate validated"
            ),
        )
    )

    invalid_count = audit_record["invalid_value_count"]

    results.append(
        make_result(
            subject_id,
            session_id,
            signal_name,
            "invalid_values",
            "FAIL",
            invalid_count > 0,
            invalid_count,
            0,
            (
                f"{invalid_count} invalid values detected"
                if invalid_count > 0
                else "All sensor values are numeric"
            ),
        )
    )

    flatline_seconds = longest_flatline_seconds(
        dataframe,
        value_columns,
        sampling_rate,
    )
    flatline_threshold = FLATLINE_LIMIT_SECONDS[signal_name]
    flatline_detected = flatline_seconds >= flatline_threshold

    results.append(
        make_result(
            subject_id,
            session_id,
            signal_name,
            "flatline",
            "REVIEW",
            flatline_detected,
            round(flatline_seconds, 2),
            flatline_threshold,
            (
                "Prolonged unchanged signal detected"
                if flatline_detected
                else "No prolonged flatline detected"
            ),
        )
    )

    duplicate_detected, repeated_seconds = detect_repeated_recording(
        dataframe,
        value_columns,
        sampling_rate,
    )

    results.append(
        make_result(
            subject_id,
            session_id,
            signal_name,
            "repeated_recording",
            "REVIEW",
            duplicate_detected,
            round(repeated_seconds, 2),
            0,
            (
                "Large repeated recording segment detected"
                if duplicate_detected
                else "No large repeated recording detected"
            ),
        )
    )

    return results


def run_session_quality_checks(
    merged: pd.DataFrame,
    audit_records: list[dict],
) -> list[dict]:
    """Run checks after HR, EDA, and ACC have been time-aligned."""
    session_id = merged["session_id"].iloc[0]
    subject_id = merged["subject_id"].iloc[0]

    results = []

    received_signals = {
        record["signal"]
        for record in audit_records
        if record.get("source_rows", 0) > 0
    }
    expected_signals = {"HR", "EDA", "ACC"}
    missing_signals = sorted(expected_signals - received_signals)

    results.append(
        make_result(
            subject_id,
            session_id,
            "SESSION",
            "missing_signals",
            "FAIL",
            bool(missing_signals),
            "|".join(missing_signals) or "None",
            "No missing signals",
            (
                f"Missing signals: {', '.join(missing_signals)}"
                if missing_signals
                else "All required signals received"
            ),
        )
    )

    coverage = float(merged["complete_second"].mean() * 100)
    low_coverage = coverage < MIN_COMPLETE_COVERAGE_PERCENT

    results.append(
        make_result(
            subject_id,
            session_id,
            "SESSION",
            "complete_coverage",
            "REVIEW",
            low_coverage,
            round(coverage, 2),
            MIN_COMPLETE_COVERAGE_PERCENT,
            (
                "Complete signal coverage is below threshold"
                if low_coverage
                else "Complete signal coverage meets threshold"
            ),
        )
    )

    valid_audits = [
        record
        for record in audit_records
        if record.get("start_time") is not None
    ]

    start_times = [record["start_time"] for record in valid_audits]
    end_times = [record["end_time"] for record in valid_audits]

    start_difference = (
        max(start_times) - min(start_times)
    ).total_seconds()

    end_difference = (
        max(end_times) - min(end_times)
    ).total_seconds()

    for name, difference in [
        ("start_time_alignment", start_difference),
        ("end_time_alignment", end_difference),
    ]:
        detected = difference > MAX_START_END_DIFFERENCE_SECONDS

        results.append(
            make_result(
                subject_id,
                session_id,
                "SESSION",
                name,
                "REVIEW",
                detected,
                round(difference, 2),
                MAX_START_END_DIFFERENCE_SECONDS,
                (
                    "Sensor timing difference exceeds threshold"
                    if detected
                    else "Sensor timing alignment is acceptable"
                ),
            )
        )

    split_session = "_" in session_id

    results.append(
        make_result(
            subject_id,
            session_id,
            "SESSION",
            "split_session",
            "REVIEW",
            split_session,
            session_id,
            "Single recording",
            (
                "Session appears to be one segment of a split recording"
                if split_session
                else "Session is stored as one recording"
            ),
        )
    )

    return results