from __future__ import annotations

from datetime import datetime

import pytest

from prescriptive_capacity_sim.callcenter_data import load_callcenter_month


HEADER = (
    "vru.line\tcall_id\tcustomer_id\tpriority\ttype\tdate\tvru_entry\t"
    "vru_exit\tvru_time\tq_start\tq_exit\tq_time\toutcome\tser_start\t"
    "ser_exit\tser_time\tserver\tday.of.week"
)


def _row(
    row_number: int,
    call_id: str,
    *,
    priority: int = 2,
    call_type: str = "PS",
    date: str = "990101",
    vru_exit: str = "7:00:05",
    q_start: str = "7:00:05",
    q_time: int = 10,
    outcome: str = "AGENT",
    ser_start: str = "7:00:15",
    ser_time: int = 60,
    server: str = "ALICE",
) -> str:
    fields = [
        row_number,
        "AA0101",
        call_id,
        "private-customer-id",
        priority,
        call_type,
        date,
        "7:00:00",
        vru_exit,
        5,
        q_start,
        "7:00:15" if q_start != "0:00:00" else "0:00:00",
        q_time,
        outcome,
        ser_start,
        "7:01:15" if outcome == "AGENT" else "0:00:00",
        ser_time,
        server,
        "friday",
    ]
    return "\t".join(map(str, fields))


def test_loader_keeps_queue_calls_and_removes_private_fields(tmp_path) -> None:
    path = tmp_path / "January.txt"
    rows = [
        _row(1, "101", q_time=10, outcome="AGENT", ser_time=60),
        _row(2, "102", q_time=30, outcome="HANG", ser_start="0:00:00", ser_time=0),
        _row(
            3,
            "103",
            q_start="0:00:00",
            q_time=0,
            outcome="HANG",
            ser_start="0:00:00",
            ser_time=0,
        ),
        _row(4, "104", q_time=12, outcome="PHANTOM", ser_start="0:00:00", ser_time=0),
        _row(5, "105", q_start="0:00:00", q_time=0, outcome="AGENT", ser_start="7:00:06", ser_time=45),
    ]
    path.write_text("\n".join([HEADER, *rows]), encoding="utf-8")

    calls = load_callcenter_month(path)

    assert [call.call_key for call in calls] == ["AA0101:101", "AA0101:102", "AA0101:105"]
    assert calls[0].queue_entry == datetime(1999, 1, 1, 7, 0, 5)
    assert calls[0].observed_wait_seconds == 10
    assert calls[0].observed_service_seconds == 60
    assert calls[1].outcome == "HANG"
    assert calls[1].observed_service_seconds is None
    assert calls[2].queue_entry == datetime(1999, 1, 1, 7, 0, 5)
    assert calls[2].observed_wait_seconds == 0
    assert "customer_id" not in calls[0].__dataclass_fields__
    assert "server" not in calls[0].__dataclass_fields__


def test_loader_filters_call_types(tmp_path) -> None:
    path = tmp_path / "January.txt"
    path.write_text(
        "\n".join([HEADER, _row(1, "101", call_type="PS"), _row(2, "102", call_type="NE")]),
        encoding="utf-8",
    )

    calls = load_callcenter_month(path, included_types={"NE"})

    assert [call.call_type for call in calls] == ["NE"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"date": "not-a-date"}, "date"),
        ({"q_time": -1}, "negative"),
        ({"ser_time": -1}, "negative"),
    ],
)
def test_loader_rejects_invalid_records(tmp_path, overrides, message) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("\n".join([HEADER, _row(1, "101", **overrides)]), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_callcenter_month(path)
