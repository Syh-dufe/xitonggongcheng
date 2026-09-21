from prescriptive_capacity_sim.config import SimulationConfig
from prescriptive_capacity_sim.environment import CallCenterEnvironment
from prescriptive_capacity_sim.state import ExogenousDraw, QueuedCall, SystemState


def _draw(*calls):
    return ExogenousDraw(
        new_calls=tuple(calls),
        next_demand_state=0,
        regular_availability=1.0,
        specialist_availability=1.0,
        service_efficiency=1.0,
        manager_alarm=0.0,
    )


def test_transition_conserves_each_queue():
    env = CallCenterEnvironment(SimulationConfig.default())
    state = SystemState(
        episode_id=0,
        period=0,
        weekday="monday",
        demand_state=0,
        waiting=(QueuedCall(0, False, 2.0, 3, 1),),
    )
    result = env.transition(state, action=1, draw=_draw(QueuedCall(1, False, 3.0, 2)))
    prior = state.queue_counts
    for klass in range(3):
        assert result.next_state.queue_counts[klass] == (
            prior[klass] + result.arrivals[klass]
            - result.served[klass] - result.abandoned[klass]
        )


def test_specialists_cross_serve_regular_at_reduced_efficiency():
    cfg = SimulationConfig.default().with_overrides(
        resources={"regular_agents": 0, "specialist_agents": 1,
                   "cross_skill_efficiency": 0.5}
    )
    env = CallCenterEnvironment(cfg)
    state = SystemState.empty(0, "monday")
    result = env.transition(state, 0, _draw(QueuedCall(0, False, 10.0, 2)))
    assert result.specialist_minutes_cross_served == 20.0
    assert result.served[0] == 1


def test_more_capacity_does_not_increase_abandonment():
    cfg = SimulationConfig.default().with_overrides(
        resources={"regular_agents": 1, "specialist_agents": 0,
                   "minutes_per_period": 1}
    )
    env = CallCenterEnvironment(cfg)
    state = SystemState.empty(0, "monday")
    calls = tuple(QueuedCall(0, False, 1.0, 1) for _ in range(3))
    low = env.transition(state, 0, _draw(*calls))
    high = env.transition(state, 3, _draw(*calls))
    assert sum(high.abandoned) <= sum(low.abandoned)


def test_priority_precedes_nonpriority_within_class():
    cfg = SimulationConfig.default().with_overrides(
        resources={"regular_agents": 1, "specialist_agents": 0,
                   "minutes_per_period": 1}
    )
    env = CallCenterEnvironment(cfg)
    calls = (
        QueuedCall(0, False, 1.0, 2),
        QueuedCall(0, True, 1.0, 2),
    )
    result = env.transition(SystemState.empty(0, "monday"), 0, _draw(*calls))
    assert result.served_priority[0] == 1
    assert result.served_nonpriority[0] == 0
