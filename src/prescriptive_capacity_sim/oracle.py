"""Common-random-number counterfactual action evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from .environment import CallCenterEnvironment
from .state import ExogenousDraw, PeriodResult, SystemState


@dataclass(frozen=True)
class OracleEvaluation:
    results: tuple[PeriodResult, ...]
    oracle_action: int


def evaluate_actions(
    environment: CallCenterEnvironment,
    state: SystemState,
    draw: ExogenousDraw,
) -> OracleEvaluation:
    results = tuple(environment.transition(state, action, draw) for action in range(4))
    oracle_action = min(range(4), key=lambda action: results[action].total_cost)
    return OracleEvaluation(results, oracle_action)
