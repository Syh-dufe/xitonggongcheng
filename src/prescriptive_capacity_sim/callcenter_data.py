"""Privacy-preserving ingestion for the Technion Anonymous Bank call logs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DOCUMENTED_COLUMNS = (
    "vru.line",
    "call_id",
    "customer_id",
    "priority",
    "type",
    "date",
    "vru_entry",
    "vru_exit",
    "vru_time",
    "q_start",
    "q_exit",
    "q_time",
    "outcome",
    "ser_start",
    "ser_exit",
    "ser_time",
    "server",
    "day.of.week",
)


@dataclass(frozen=True)
class QueueCall:
    """A de-identified queue observation suitable for calibration."""

    call_key: str
    queue_entry: datetime
    call_type: str
    priority: int
    observed_wait_seconds: float
    outcome: str
    observed_service_seconds: float | None


def _is_zero_time(value: str) -> bool:
    return value.strip() in {"0:00:00", "00:00:00"}


def _timestamp(date_text: str, time_text: str, *, row_number: int) -> datetime:
    try:
        date = datetime.strptime(date_text.strip(), "%y%m%d").date()
        time = datetime.strptime(time_text.strip(), "%H:%M:%S").time()
    except ValueError as exc:
        raise ValueError(
            f"invalid date or time in row {row_number}: {date_text!r} {time_text!r}"
        ) from exc
    return datetime.combine(date, time)


def load_callcenter_month(
    path: Path, included_types: set[str] | None = None
) -> list[QueueCall]:
    """Read one Bern-mirror month without exposing customer or agent identifiers.

    The Bern mirror stores an unlabeled row number before the 18 documented
    columns. IVR-only abandonments and ``PHANTOM`` records never enter the
    service queue and are therefore excluded.
    """

    path = Path(path)
    calls: list[QueueCall] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise ValueError(f"empty call-center file: {path}") from exc
        if header != DOCUMENTED_COLUMNS:
            raise ValueError(
                f"unexpected call-center columns in {path.name}: {header!r}"
            )

        for physical_row, values in enumerate(reader, start=2):
            if not values or all(not value.strip() for value in values):
                continue
            if len(values) != len(DOCUMENTED_COLUMNS) + 1:
                raise ValueError(
                    f"row {physical_row} has {len(values)} fields; expected 19"
                )
            row = dict(zip(DOCUMENTED_COLUMNS, values[1:], strict=True))
            call_type = row["type"].strip()
            if included_types is not None and call_type not in included_types:
                continue

            try:
                wait_seconds = float(row["q_time"])
                service_seconds = float(row["ser_time"])
                priority = int(row["priority"])
            except ValueError as exc:
                raise ValueError(f"invalid numeric value in row {physical_row}") from exc
            if wait_seconds < 0 or service_seconds < 0:
                raise ValueError(f"negative duration in row {physical_row}")

            outcome = row["outcome"].strip().upper()
            if outcome == "PHANTOM":
                continue
            queued = not _is_zero_time(row["q_start"])
            if outcome == "HANG" and not queued:
                continue
            if outcome not in {"AGENT", "HANG"}:
                continue

            if queued:
                entry_time = row["q_start"]
            else:
                entry_time = row["vru_exit"]
                if _is_zero_time(entry_time):
                    entry_time = row["ser_start"]
                wait_seconds = 0.0

            observed_service = service_seconds if outcome == "AGENT" else None
            calls.append(
                QueueCall(
                    call_key=f'{row["vru.line"].strip()}:{row["call_id"].strip()}',
                    queue_entry=_timestamp(
                        row["date"], entry_time, row_number=physical_row
                    ),
                    call_type=call_type,
                    priority=priority,
                    observed_wait_seconds=wait_seconds,
                    outcome=outcome,
                    observed_service_seconds=observed_service,
                )
            )
    return calls
