from __future__ import annotations

from prescriptive_capacity_sim.callcenter_simulator import (
    BusyCall,
    CallCenterState,
    SimulatedCall,
    simulate_interval,
)


def _call(
    key: str,
    arrival: float,
    *,
    service: float = 10,
    patience: float = 100,
    priority: int = 0,
) -> SimulatedCall:
    return SimulatedCall(
        call_key=key,
        arrival_second=arrival,
        service_seconds=service,
        patience_seconds=patience,
        priority=priority,
        call_type="PS",
    )


def test_free_agent_starts_service_immediately() -> None:
    result = simulate_interval(
        CallCenterState.empty(), [_call("a", 3, service=5)], staffing=1, interval_seconds=10
    )

    assert result.started_call_keys == ("a",)
    assert result.completed_call_keys == ("a",)
    assert result.mean_wait_seconds == 0
    assert result.busy_seconds == 5
    assert result.next_state.now_second == 10


def test_waiting_calls_use_priority_then_fifo_order() -> None:
    state = CallCenterState(
        now_second=0,
        waiting=[],
        busy=[
            BusyCall(
                call=_call("incumbent", -5, service=15),
                service_start_second=-5,
                completion_second=10,
                wait_seconds=0,
            )
        ],
    )
    arrivals = [
        _call("regular-first", 1, service=1, priority=0),
        _call("regular-second", 2, service=1, priority=0),
        _call("priority", 3, service=1, priority=2),
    ]

    result = simulate_interval(state, arrivals, staffing=1, interval_seconds=20)

    assert result.started_call_keys == ("priority", "regular-first", "regular-second")


def test_call_abandons_at_patience_deadline() -> None:
    result = simulate_interval(
        CallCenterState.empty(),
        [_call("a", 0, service=20), _call("b", 1, patience=4)],
        staffing=1,
        interval_seconds=10,
    )

    assert result.abandoned_call_keys == ("b",)
    assert result.abandoned == 1
    assert result.next_state.busy[0].call.call_key == "a"


def test_service_can_complete_in_a_later_interval() -> None:
    first = simulate_interval(
        CallCenterState.empty(),
        [_call("a", 0, service=15)],
        staffing=1,
        interval_seconds=10,
    )
    second = simulate_interval(
        first.next_state, [], staffing=0, interval_seconds=10
    )

    assert first.completed_service == 0
    assert len(first.next_state.busy) == 1
    assert second.completed_call_keys == ("a",)
    assert len(second.next_state.busy) == 0


def test_interval_obeys_flow_conservation() -> None:
    state = CallCenterState(
        now_second=0,
        waiting=[],
        busy=[
            BusyCall(
                call=_call("old", -5, service=10),
                service_start_second=-5,
                completion_second=5,
                wait_seconds=0,
            )
        ],
    )
    result = simulate_interval(
        state,
        [
            _call("served", 1, service=3),
            _call("hang", 2, patience=1),
            _call("left", 9, service=20),
        ],
        staffing=1,
        interval_seconds=10,
    )

    entering = len(state.waiting) + len(state.busy) + result.arrivals
    leaving_or_remaining = (
        result.completed_service
        + result.abandoned
        + len(result.next_state.waiting)
        + len(result.next_state.busy)
    )
    assert entering == leaving_or_remaining


def test_replay_is_deterministic_and_more_staffing_does_not_raise_abandonment() -> None:
    arrivals = [
        _call(f"c{i}", i, service=10, patience=3 + (i % 2)) for i in range(8)
    ]

    low_first = simulate_interval(
        CallCenterState.empty(), arrivals, staffing=1, interval_seconds=30
    )
    low_second = simulate_interval(
        CallCenterState.empty(), arrivals, staffing=1, interval_seconds=30
    )
    high = simulate_interval(
        CallCenterState.empty(), arrivals, staffing=3, interval_seconds=30
    )

    assert low_first == low_second
    assert high.abandoned <= low_first.abandoned


def test_invalid_calls_and_staffing_are_rejected() -> None:
    invalid = _call("bad", 0, service=-1)

    try:
        simulate_interval(CallCenterState.empty(), [invalid], staffing=1)
    except ValueError as exc:
        assert "service" in str(exc)
    else:
        raise AssertionError("negative service time was accepted")

    try:
        simulate_interval(CallCenterState.empty(), [], staffing=-1)
    except ValueError as exc:
        assert "staffing" in str(exc)
    else:
        raise AssertionError("negative staffing was accepted")
