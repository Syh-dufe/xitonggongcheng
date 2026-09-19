"""Counterfactual action branching with common exogenous randomness."""

from __future__ import annotations

from dataclasses import dataclass

from .environment import CapacityEnvironment
from .state import ExogenousShock, PeriodResult, SystemState


@dataclass(frozen=True)
class OracleEvaluation:
    results: tuple[PeriodResult, ...]
    oracle_action: int


def evaluate_actions(
    environment: CapacityEnvironment,
    state: SystemState,
    shock: ExogenousShock,
) -> OracleEvaluation:
    results = tuple(
        environment.transition(state, action, shock)
        for action in range(len(environment.config.behavior.action_levels))
    )
    oracle_action = min(range(len(results)), key=lambda action: results[action].total_cost)
    return OracleEvaluation(results=results, oracle_action=oracle_action)

