from dataclasses import replace

from prescriptive_capacity_sim.config import SimulationConfig
from prescriptive_capacity_sim.environment import CapacityEnvironment
from prescriptive_capacity_sim.state import ExogenousShock, SystemState


def fixed_shock(arrivals=(4, 2, 1), next_state=0) -> ExogenousShock:
    return ExogenousShock(
        arrivals=arrivals,
        next_disruption_state=next_state,
        service_multiplier=1.0,
        manager_alarm=0.0,
    )


def test_reset_is_reproducible_and_empty() -> None:
    cfg = SimulationConfig.default()
    env = CapacityEnvironment(cfg)
    assert env.reset(episode_id=7) == env.reset(episode_id=7)
    assert env.reset(episode_id=7).backlog == (0, 0, 0)


def test_transition_obeys_flow_conservation_and_service_bounds() -> None:
    cfg = SimulationConfig.default()
    env = CapacityEnvironment(cfg)
    state = SystemState.from_backlog(cfg, 0, 3, (5, 3, 2), disruption_state=0)
    result = env.transition(state, action=1, shock=fixed_shock())
    for prior, arrived, completed, remaining in zip(
        state.backlog, result.arrivals, result.completed, result.next_state.backlog
    ):
        assert remaining == prior + arrived - completed
        assert 0 <= completed <= prior + arrived
    assert all(value >= 0 for value in result.next_state.backlog)


def test_higher_action_has_higher_staffing_cost_for_same_shock() -> None:
    cfg = SimulationConfig.default()
    env = CapacityEnvironment(cfg)
    state = SystemState.from_backlog(cfg, 0, 3, (100, 50, 20), disruption_state=1)
    low = env.transition(state, action=0, shock=fixed_shock((0, 0, 0), 1))
    high = env.transition(state, action=3, shock=fixed_shock((0, 0, 0), 1))
    assert high.staffing_cost > low.staffing_cost
    assert sum(high.completed) >= sum(low.completed)


def test_unserved_work_ages_and_can_trigger_safety_violation() -> None:
    base = SimulationConfig.default()
    cfg = replace(base, base_staff=0, senior_staff=0)
    cfg.validate()
    env = CapacityEnvironment(cfg)
    state = SystemState.from_backlog(cfg, 0, 0, (0, 0, 4), disruption_state=2)
    result = env.transition(state, action=0, shock=fixed_shock((0, 0, 0), 2))
    assert result.next_state.max_ages[2] == 1
    assert result.safety_violation
    assert result.total_cost >= result.safety_cost > 0
