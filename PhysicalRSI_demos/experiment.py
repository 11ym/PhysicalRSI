"""Runnable CPU example of independently verified experiments."""
from PhysicalRSI_core.experiments import Budget, ExperimentRuntime


class CounterEnvironment:
    def identity(self):
        return {"name": "counter", "revision": "1", "scope": "software"}

    def reset(self, case, context):
        self.position, self.target = case["initial"], case["target"]
        return {"ready": True, "observation": {"position": self.position}}

    def step(self, action, context):
        self.position += action
        return {"observation": {"position": self.position}, "terminated": self.position >= self.target}


class CounterPolicy:
    def __init__(self, increment=1):
        self.increment = increment

    def identity(self):
        return {"name": "increment", "increment": self.increment}

    def act(self, observation, context):
        return self.increment


class CounterVerifier:
    def identity(self):
        return {"name": "measured-counter-position", "revision": "1"}

    def verify(self, case, trace):
        measured = trace[-1]["observation"]["position"]
        return {"outcome": "success" if measured == case["target"] else "failure",
                "reason": "Compare measured final position with the declared target",
                "measurements": {"position": measured, "target": case["target"]}}


def run(workspace, run_id="counter-example"):
    return ExperimentRuntime(workspace).run(
        run_id, task="counter", case={"initial": 0, "target": 3},
        scope="CPU software contract example; no physical qualification",
        environment=CounterEnvironment(), policy=CounterPolicy(),
        verifier=CounterVerifier(), budget=Budget(max_steps=5, seconds=30))
