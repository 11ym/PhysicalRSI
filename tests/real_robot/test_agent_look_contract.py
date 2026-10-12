"""Invalid public policy inputs must be actionable feedback, before sensing/motion."""
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PhysicalRSI_baselines.real_robot.capabilities import API
from PhysicalRSI_baselines.real_robot.contracts import CapabilityUnavailable
from PhysicalRSI_baselines.real_robot.controllers import L1Controller
from PhysicalRSI_baselines.real_robot.visual_perception import validate_features


@pytest.mark.parametrize("features,detail", [
    ([str(i) for i in range(18)], "received 18"),
    (["bottle", "bottle"], "unique"),
    (["x" * 161], "160"),
])
def test_bad_look_arguments_reach_learning_without_sensor_or_action(tmp_path, features, detail):
    class Runtime:
        def execute(self, source, memory, *, handlers, output):
            handlers["look"]([features], {}, deadline=time.monotonic() + 10)
    context = SimpleNamespace(episode=SimpleNamespace(root=tmp_path), robot=Mock(), checkpoint=Mock())
    perception = Mock()
    with pytest.raises(CapabilityUnavailable, match=detail):
        L1Controller(dict(source="unused", memory={}), Runtime(), perception=perception).run(context)
    context.checkpoint.assert_not_called()
    assert context.robot.mock_calls == []
    assert perception.mock_calls == []


def test_published_limit_matches_enforced_limit():
    rule = API["methods"]["look"]["feature_constraints"]
    assert rule == dict(min_count=1, max_count=16, unique=True, max_name_characters=160)
    validate_features([str(i) for i in range(rule["max_count"])])
    with pytest.raises(ValueError):
        validate_features([str(i) for i in range(rule["max_count"] + 1)])
