"""Executable CPU contract service, not a robot simulator or benchmark."""

import argparse
from pathlib import Path

import numpy as np

from PhysicalRSI_core.infra.hosts import EnvironmentHost, ModelHost
from PhysicalRSI_core.infra.rpc.journal import RequestJournal

ENVIRONMENT = {"kind": "local-counter", "revision": "1", "target": 2}
MODEL = {"kind": "local-increment-policy", "revision": "1"}


class Counter:
    def __init__(self):
        self.value = 0.0

    def observe(self):
        return {"state": np.array([self.value], dtype=np.float32)}

    def reset(self):
        self.value = 0.0
        return self.observe()

    def step(self, action):
        self.value += float(action[0])
        solved = self.value == ENVIRONMENT["target"]
        return dict(
            observation=self.observe(),
            reward=float(solved),
            terminated=solved,
            truncated=False,
            success=solved,
        )

    def chunk_step(self, actions, return_all_frames=False):
        if not len(actions):
            raise ValueError("Nonempty chunk required")
        frames = []
        for action in actions:
            result = self.step(action)
            if return_all_frames:
                frames.append(result["observation"])
            if result["terminated"] or result["truncated"]:
                break
        return dict(result, frames=frames)

    def close(self):
        pass


class IncrementPolicy:
    def __init__(self):
        self.calls = {}

    def predict(self, observation, options, *, session_id):
        self.calls[session_id] = self.calls.get(session_id, 0) + 1
        target = options["goal"]["target"]
        step = float(np.sign(target - observation["state"][0]))
        return np.array([[step], [step]], dtype=np.float32)

    def reset(self, session_id):
        self.calls.pop(session_id, None)

    def close(self):
        self.calls.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["environment", "model"])
    parser.add_argument("--journal", required=True, type=Path)
    args = parser.parse_args()
    journal = RequestJournal(args.journal)
    if args.kind == "environment":
        host = EnvironmentHost(Counter(), metadata=ENVIRONMENT, journal=journal)
    else:
        host = ModelHost(IncrementPolicy(), metadata=MODEL, journal=journal)
    host.serve(
        transport="http",
        host="127.0.0.1",
        port=0,
        parent_watch=True,
        session_sweep_s=0.1,
    )


if __name__ == "__main__":
    main()
