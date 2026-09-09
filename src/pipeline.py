"""Pipeline for Empatica E4 stress-session data."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from quality_checks import (
    run_raw_quality_checks,
    run_session_quality_checks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = DATA_ROOT / "processed"

EXPECTED_SIGNALS = {
    "HR": {
        "columns": ["heart_rate_bpm"],
        "sampling_rate": 1.0,
    },
    "EDA": {
        "columns": ["eda_microsiemens"],
        "sampling_rate": 4.0,
    },
    "ACC": {
        "columns": ["acc_x_raw", "acc_y_raw", "acc_z_raw"],
        "sampling_rate": 32.0,
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def locate_stress_directory() -> Path:
    """Locate the STRESS directory."""
    matches = [
        path
        for path in DATA_ROOT.rglob("STRESS")
        if path.is_dir() and path.parent.name == "Wearable_Dataset"
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one STRESS directory, found {len(matches)}."
        )

    return matches[0]


def parse_subject_id(session_id: str) -> str:
    """Return f14 from split-session names such as f14_a."""
    match = re.match(r"([A-Za-z]\d+)", session_id)

    if match is None:
        raise ValueError(f"Invalid session folder name: {session_id}")

    return match.group(1)


def get_protocol_version(subject_id: str) -> str:
    """Map official participant prefixes to protocol versions."""
    if subject_id.startswith("S"):
        return "V1"

    if subject_id.startswith("f"):
        return "V2"

    raise ValueError(f"Unknown participant prefix: {subject_id}")


def read_e4_signal(
    file_path: Path,
    signal_name: str,
) -> tuple[pd.DataFrame, dict]:
    """Read, validate, and timestamp one fixed-frequency E4 signal."""
    specification = EXPECTED_SIGNALS[signal_name]
    expected_columns = specification["columns"]
    expected_rate = specification["sampling_rate"]

    raw = pd.read_csv(file_path, header=None)

    if len(raw) < 3:
        raise ValueError(f"{file_path} contains no sensor observations.")

    if raw.shape[1] != len(expected_columns):
        raise ValueError(
            f"{file_path} has {raw.shape[1]} columns; "
            f"expected {len(expected_columns)}."
        )

    start_values = raw.iloc[0].dropna().astype(str).unique()
    rate_values = pd.to_numeric(
        raw.iloc[1],
        errors="coerce",
    ).dropna().unique()

    if len(start_values) != 1:
        raise ValueError(f"Inconsistent start time in {file_path}")

    if len(rate_values) != 1:
        raise ValueError(f"Inconsistent sampling rate in {file_path}")

    start_time = pd.to_datetime(start_values[0], utc=True)
    observed_rate = float(rate_values[0])

    values = raw.iloc[2:].copy()
    values.columns = expected_columns
    values = values.apply(pd.to_numeric, errors="coerce")

    invalid_value_count = int(values.isna().sum().sum())

    offsets = np.arange(len(values), dtype=float) / observed_rate
    values.insert(
        0,
        "timestamp",
        start_time + pd.to_timedelta(offsets, unit="s"),
    )

    end_time = values["timestamp"].iloc[-1]

    audit_record = {
        "subject_id": parse_subject_id(file_path.parent.name),
        "session_id": file_path.parent.name,
        "signal": signal_name,
        "source_rows": len(values),
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": (
            end_time - start_time
        ).total_seconds(),
        "expected_sampling_rate_hz": expected_rate,
        "observed_sampling_rate_hz": observed_rate,
        "sampling_rate_valid": bool(
            np.isclose(observed_rate, expected_rate)
        ),
        "invalid_value_count": invalid_value_count,
        "source_file": str(file_path.relative_to(PROJECT_ROOT)),
    }

    return values, audit_record


def aggregate_to_one_second(
    dataframe: pd.DataFrame,
    signal_name: str,
) -> pd.DataFrame:
    """Convert each signal to a consistent one-second time grid."""
    dataframe = dataframe.set_index("timestamp")

    if signal_name == "HR":
        result = dataframe.resample("1s").agg(
            heart_rate_bpm=("heart_rate_bpm", "mean")
        )

    elif signal_name == "EDA":
        result = dataframe.resample("1s").agg(
            eda_microsiemens=("eda_microsiemens", "mean")
        )

    elif signal_name == "ACC":
        # Official ACC unit is 1/64 g.
        dataframe["acc_x_g"] = dataframe["acc_x_raw"] / 64.0
        dataframe["acc_y_g"] = dataframe["acc_y_raw"] / 64.0
        dataframe["acc_z_g"] = dataframe["acc_z_raw"] / 64.0

        dataframe["acc_magnitude_g"] = np.sqrt(
            dataframe["acc_x_g"] ** 2
            + dataframe["acc_y_g"] ** 2
            + dataframe["acc_z_g"] ** 2
        )

        result = dataframe.resample("1s").agg(
            movement_mean_g=("acc_magnitude_g", "mean"),
            movement_variability_g=("acc_magnitude_g", "std"),
        )

    else:
        raise ValueError(f"Unsupported signal: {signal_name}")

    return result.reset_index()


def read_event_tags(session_directory: Path) -> pd.DataFrame:
    """Read physical event-button timestamps for one session."""
    tags_path = session_directory / "tags.csv"

    if not tags_path.exists() or tags_path.stat().st_size == 0:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "session_id",
                "protocol_version",
                "event_number",
                "timestamp",
            ]
        )

    tags = pd.read_csv(tags_path, header=None, names=["timestamp"])
    tags["timestamp"] = pd.to_datetime(
        tags["timestamp"],
        utc=True,
        errors="coerce",
    )
    tags = tags.dropna(subset=["timestamp"]).reset_index(drop=True)

    session_id = session_directory.name
    subject_id = parse_subject_id(session_id)

    tags.insert(0, "event_number", np.arange(1, len(tags) + 1))
    tags.insert(0, "protocol_version", get_protocol_version(subject_id))
    tags.insert(0, "session_id", session_id)
    tags.insert(0, "subject_id", subject_id)

    return tags


def process_session(
    session_directory: Path,
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Process all selected signals belonging to one stress session."""
    session_id = session_directory.name
    subject_id = parse_subject_id(session_id)
    protocol = get_protocol_version(subject_id)

    signal_frames = []
    audit_records = []

    quality_results = []

    for signal_name in EXPECTED_SIGNALS:
        file_path = session_directory / f"{signal_name}.csv"

        if not file_path.exists():
            audit_records.append(
                {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "signal": signal_name,
                    "source_rows": 0,
                    "sampling_rate_valid": False,
                    "invalid_value_count": None,
                    "issue": "Missing source file",
                }
            )
            continue

        raw_signal, audit_record = read_e4_signal(
            file_path,
            signal_name,
        )
        quality_results.extend(
            run_raw_quality_checks(
                dataframe=raw_signal,
                signal_name=signal_name,
                audit_record=audit_record,
            )
        )
        
        

        audit_record["issue"] = (
            "None"
            if audit_record["sampling_rate_valid"]
            and audit_record["invalid_value_count"] == 0
            else "Format or value validation failed"
        )

        audit_records.append(audit_record)

        one_second_signal = aggregate_to_one_second(
            raw_signal,
            signal_name,
        )
        signal_frames.append(one_second_signal)

    if not signal_frames:
        raise RuntimeError(f"No usable signals in session {session_id}")

    merged = signal_frames[0]

    for frame in signal_frames[1:]:
        merged = merged.merge(
            frame,
            on="timestamp",
            how="outer",
            validate="one_to_one",
        )

    merged = merged.sort_values("timestamp").reset_index(drop=True)

    merged.insert(0, "protocol", "STRESS")
    merged.insert(0, "protocol_version", protocol)
    merged.insert(0, "session_id", session_id)
    merged.insert(0, "subject_id", subject_id)

    required_columns = [
        "heart_rate_bpm",
        "eda_microsiemens",
        "movement_mean_g",
    ]

    available_columns = [
        column for column in required_columns if column in merged.columns
    ]

    merged["available_signal_count"] = (
        merged[available_columns].notna().sum(axis=1)
    )
    merged["complete_second"] = (
        merged["available_signal_count"] == len(required_columns)
    )
    quality_results.extend(
        run_session_quality_checks(
            merged=merged,
            audit_records=audit_records,
            )
            )
    events = read_event_tags(session_directory)

    return merged, audit_records, events,quality_results


def create_session_summary(
    timeseries: pd.DataFrame,
    audit: pd.DataFrame,
    quality_checks: pd.DataFrame,
) -> pd.DataFrame:
    """Create one operational quality record per session."""
    summaries = []

    for session_id, session_data in timeseries.groupby("session_id"):
        session_audit = audit[audit["session_id"] == session_id]

        detected_checks = quality_checks[
            (quality_checks["session_id"] == session_id)
            & (quality_checks["detected"] == True)
        ]

        expected_signal_count = len(EXPECTED_SIGNALS)
        present_signal_count = int(
            (session_audit["source_rows"].fillna(0) > 0).sum()
        )

        complete_percentage = float(
            session_data["complete_second"].mean() * 100
        )

        has_failure = (
            detected_checks["severity"].eq("FAIL").any()
        )
        has_review = (
            detected_checks["severity"].eq("REVIEW").any()
        )

        if has_failure:
            quality_status = "FAIL"
        elif has_review:
            quality_status = "REVIEW"
        else:
            quality_status = "PASS"

        issue_descriptions = [
            f"{row.signal}: {row.message}"
            for row in detected_checks.itertuples()
        ]

        summaries.append(
            {
                "subject_id": session_data["subject_id"].iloc[0],
                "session_id": session_id,
                "protocol_version": session_data[
                    "protocol_version"
                ].iloc[0],
                "start_time": session_data["timestamp"].min(),
                "end_time": session_data["timestamp"].max(),
                "duration_minutes": round(
                    (
                        session_data["timestamp"].max()
                        - session_data["timestamp"].min()
                    ).total_seconds()
                    / 60,
                    2,
                ),
                "signals_received": (
                    f"{present_signal_count}/{expected_signal_count}"
                ),
                "complete_seconds_percent": round(
                    complete_percentage,
                    2,
                ),
                "detected_issue_count": len(detected_checks),
                "quality_status": quality_status,
                "detected_issues": (
                    "; ".join(issue_descriptions)
                    if issue_descriptions
                    else "No automated issues detected"
                ),
            }
        )

    return pd.DataFrame(summaries)


def run_pipeline() -> None:
    """Run the complete raw-to-processed transformation pipeline."""
    stress_directory = locate_stress_directory()
    session_directories = sorted(
        {
            file_path.parent
            for file_path in stress_directory.rglob("HR.csv")
        }
    )

    if not session_directories:
        raise RuntimeError("No stress sessions were found.")

    LOGGER.info("Found %d stress sessions", len(session_directories))

    all_timeseries = []
    all_audits = []
    all_events = []
    all_quality_results = []

    for session_directory in session_directories:
        LOGGER.info("Processing %s", session_directory.name)

        timeseries, audit_records, events, quality_results = (
            process_session(session_directory)
            )
       
        all_timeseries.append(timeseries)
        all_audits.extend(audit_records)
        all_quality_results.extend(quality_results)

        if not events.empty:
            all_events.append(events)

    timeseries_output = pd.concat(
        all_timeseries,
        ignore_index=True,
    )
    audit_output = pd.DataFrame(all_audits)
    quality_output = pd.DataFrame(all_quality_results)
    events_output = (
        pd.concat(all_events, ignore_index=True)
        if all_events
        else pd.DataFrame()
    )

    session_summary = create_session_summary(
        timeseries_output,
        audit_output,
        quality_output,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    timeseries_output.to_csv(
        OUTPUT_ROOT / "stress_timeseries_1s.csv",
        index=False,
    )
    audit_output.to_csv(
        OUTPUT_ROOT / "signal_audit.csv",
        index=False,
    )
    session_summary.to_csv(
        OUTPUT_ROOT / "session_summary.csv",
        index=False,
    )
    events_output.to_csv(
        OUTPUT_ROOT / "session_events.csv",
        index=False,
    )
    quality_output.to_csv(
        OUTPUT_ROOT / "quality_checks.csv",
        index=False,
    )
    LOGGER.info(
        "Pipeline complete: %d one-second observations",
        len(timeseries_output),
    )
    LOGGER.info("Outputs written to %s", OUTPUT_ROOT)


if __name__ == "__main__":
    run_pipeline()