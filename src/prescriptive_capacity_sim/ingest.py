"""Read and de-identify the Technion anonymous-bank call logs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SERVICE_CLASS_MAP = {
    "PS": "regular",
    "PE": "regular",
    "NW": "regular",
    "IN": "specialist",
    "NE": "specialist",
    "TT": "callback_special",
}

_RAW_COLUMNS = [
    "source_row", "vru_line", "call_id", "customer_id", "priority", "type",
    "date", "vru_entry", "vru_exit", "vru_time", "q_start", "q_exit",
    "q_time", "outcome", "ser_start", "ser_exit", "ser_time", "server",
    "day_of_week",
]


@dataclass(frozen=True)
class DataQualityCounts:
    total_rows: int = 0
    phantom_rows: int = 0
    prequeue_exit_rows: int = 0
    unknown_type_rows: int = 0
    invalid_time_rows: int = 0

    def __add__(self, other: "DataQualityCounts") -> "DataQualityCounts":
        return DataQualityCounts(**{
            name: getattr(self, name) + getattr(other, name)
            for name in self.__dataclass_fields__
        })


@dataclass(frozen=True)
class IngestedCalls:
    calls: pd.DataFrame
    quality: DataQualityCounts
    source_files: tuple[str, ...]


def load_month(path: str | Path) -> tuple[pd.DataFrame, DataQualityCounts]:
    path = Path(path)
    raw = pd.read_csv(
        path,
        sep="\t",
        names=_RAW_COLUMNS,
        skiprows=1,
        dtype=str,
        keep_default_na=False,
        engine="python",
    )
    # Finder metadata files and malformed blank lines must not become calls.
    raw = raw.loc[raw["date"].str.fullmatch(r"\d{6}", na=False)].copy()
    total = len(raw)
    raw["type"] = raw["type"].str.strip().str.upper()
    raw["outcome"] = raw["outcome"].str.strip().str.upper()
    for column in ("priority", "vru_time", "q_time", "ser_time"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    raw["call_date"] = pd.to_datetime(raw["date"], format="%y%m%d", errors="coerce")
    raw["arrival_ts"] = _combine_time(raw["call_date"], raw["vru_entry"])
    raw["queue_start_ts"] = _combine_time(raw["call_date"], raw["q_start"], raw["arrival_ts"])
    raw["queue_exit_ts"] = _combine_time(raw["call_date"], raw["q_exit"], raw["queue_start_ts"])
    raw["service_start_ts"] = _combine_time(raw["call_date"], raw["ser_start"], raw["arrival_ts"])
    raw["service_exit_ts"] = _combine_time(raw["call_date"], raw["ser_exit"], raw["service_start_ts"])

    invalid_time = raw["arrival_ts"].isna() | raw["call_date"].isna()
    unknown_type = ~raw["type"].isin(SERVICE_CLASS_MAP)
    phantom = raw["outcome"].eq("PHANTOM")
    prequeue = raw["outcome"].eq("HANG") & raw["q_start"].isin({"0:00:00", "00:00:00"})

    quality = DataQualityCounts(
        total_rows=int(total),
        phantom_rows=int(phantom.sum()),
        prequeue_exit_rows=int((prequeue & ~phantom).sum()),
        unknown_type_rows=int(unknown_type.sum()),
        invalid_time_rows=int(invalid_time.sum()),
    )

    include = ~(invalid_time | unknown_type | phantom | prequeue)
    clean = raw.loc[include].copy()
    clean["service_class"] = clean["type"].map(SERVICE_CLASS_MAP)
    clean["priority"] = clean["priority"].fillna(0).astype(int).gt(1).astype(int)
    clean["service_seconds"] = clean["ser_time"].where(
        clean["outcome"].eq("AGENT") & clean["ser_time"].gt(0), np.nan
    )
    clean["patience_seconds"] = clean["q_time"].where(
        clean["outcome"].eq("HANG") & clean["q_time"].gt(0), np.nan
    )
    clean["queue_seconds"] = clean["q_time"].fillna(0).clip(lower=0)
    clean["vru_seconds"] = clean["vru_time"].fillna(0).clip(lower=0)
    clean["weekday"] = clean["call_date"].dt.day_name().str.lower()
    columns = [
        "call_date", "weekday", "arrival_ts", "queue_start_ts", "queue_exit_ts",
        "service_start_ts", "service_exit_ts", "service_class", "priority",
        "outcome", "vru_seconds", "queue_seconds", "service_seconds",
        "patience_seconds",
    ]
    clean = clean[columns].sort_values("arrival_ts", kind="stable").reset_index(drop=True)
    return clean, quality


def load_raw_directory(path: str | Path) -> IngestedCalls:
    root = Path(path)
    files = tuple(sorted(
        p for p in root.glob("*.txt")
        if p.is_file() and not p.name.startswith("._")
    ))
    if not files:
        raise FileNotFoundError(f"no monthly .txt files found in {root}")
    frames: list[pd.DataFrame] = []
    quality = DataQualityCounts()
    for source in files:
        frame, counts = load_month(source)
        frames.append(frame)
        quality = quality + counts
    calls = pd.concat(frames, ignore_index=True).sort_values(
        "arrival_ts", kind="stable"
    ).reset_index(drop=True)
    return IngestedCalls(calls, quality, tuple(p.name for p in files))


def _combine_time(
    dates: pd.Series,
    values: pd.Series,
    references: pd.Series | None = None,
) -> pd.Series:
    text = values.astype(str).str.strip()
    missing = text.isin({"", "0:00:00", "00:00:00", "nan"})
    delta = pd.to_timedelta(text.where(~missing), errors="coerce")
    result = dates + delta
    if references is not None:
        rollover = result.notna() & references.notna() & (result < references)
        result = result.where(~rollover, result + pd.Timedelta(days=1))
    return result
