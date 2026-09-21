from pathlib import Path

import pytest

from prescriptive_capacity_sim.config import SimulationConfig


def test_defaults_use_documented_agent_pools():
    cfg = SimulationConfig.default()
    assert cfg.resources.regular_agents == 8
    assert cfg.resources.specialist_agents == 5
    assert cfg.behavior.action_levels == (0.0, 0.1, 0.2, 0.3)


def test_missing_calibration_is_rejected(tmp_path: Path):
    cfg = SimulationConfig.default().with_overrides(
        calibration_path=tmp_path / "missing.json"
    )
    with pytest.raises(FileNotFoundError):
        cfg.validate_for_simulation()


def test_invalid_cross_skill_efficiency_is_rejected():
    with pytest.raises(ValueError):
        SimulationConfig.default().with_overrides(
            resources={"cross_skill_efficiency": 1.5}
        )
