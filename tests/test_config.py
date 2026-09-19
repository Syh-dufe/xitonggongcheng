from dataclasses import replace

import pytest

from prescriptive_capacity_sim.config import SimulationConfig


def test_default_configuration_is_valid() -> None:
    cfg = SimulationConfig.default()
    cfg.validate()
    assert cfg.periods_per_day == 48
    assert cfg.behavior.action_levels == (0.0, 0.1, 0.2, 0.3)


def test_partial_yaml_overrides_defaults(tmp_path) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text("behavior:\n  bias_strength: 2.5\n", encoding="utf-8")
    cfg = SimulationConfig.from_yaml(path)
    assert cfg.behavior.bias_strength == 2.5
    assert cfg.periods_per_day == 48


@pytest.mark.parametrize(
    "cfg",
    [
        replace(
            SimulationConfig.default(),
            demand=replace(
                SimulationConfig.default().demand,
                transition_matrix=((0.8, 0.3, 0.0), (0.2, 0.7, 0.1), (0.1, 0.2, 0.7)),
            ),
        ),
        replace(
            SimulationConfig.default(),
            costs=replace(SimulationConfig.default().costs, temporary_unit_cost=-1.0),
        ),
        replace(
            SimulationConfig.default(),
            behavior=replace(SimulationConfig.default().behavior, action_levels=(0.0, 0.2, 0.1, 0.3)),
        ),
    ],
)
def test_invalid_configuration_is_rejected(cfg) -> None:
    with pytest.raises(ValueError):
        cfg.validate()
