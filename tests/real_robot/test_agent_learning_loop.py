"""Full offline RGB / rescue / code evolution / selection integration.

Models and human teleoperation are deterministic fixtures. This verifies the
software experiment, not GPT reasoning quality or real-world grasp success.
"""
import base64
from concurrent.futures import Future
from contextlib import nullcontext
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import h5py
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PhysicalRSI_demos.real_robot.contracts import Level, Task
from PhysicalRSI_demos.real_robot.controllers import L1Controller, L2Controller
from PhysicalRSI_demos.real_robot.coordinator import Coordinator
from PhysicalRSI_demos.real_robot.demonstrations import CAMERA_NAMES, TerminalTeleoperation
from PhysicalRSI_demos.real_robot.evidence import Episode, metrics, write_new
from PhysicalRSI_demos.real_robot.evolution import DevelopmentProposer, PairedEpisodeEvaluator, make_harness
from PhysicalRSI_demos.real_robot.learning import read_development_episode, selected_policy
from PhysicalRSI_demos.real_robot.library import PolicyLibrary
from PhysicalRSI_demos.real_robot.monitor import MonitoringSystem
from PhysicalRSI_demos.real_robot.policy_runtime import PolicyRuntime
from test_visual_servo import ImagePlant, PixelDetector


def offsets(images):
    errors = []
    for image in images:
        pixels = np.asarray(image)
        points = []
        for channel in (1, 0):
            yy, xx = np.nonzero(pixels[:, :, channel] > 200)
            if not len(xx):
                return None
            points.append(np.array([xx.mean(), yy.mean()]) / (pixels.shape[0]-1))
        errors.append(points[0]-points[1])
    return np.asarray(errors)


class RGBJudgeModel:
    def __init__(self):
        self.config = SimpleNamespace(protocol="chat_completions")
        self.histories = []

    def complete(self, history, tools):
        self.histories.append(history)
        content = history[-1]["content"]
        payload = json.loads(content[0]["text"])
        assert "qpos26" not in json.dumps(payload)
        images = [Image.open(io.BytesIO(base64.b64decode(c["image_url"]["url"].split(",", 1)[1]))).convert("RGB")
                  for c in content if c["type"] == "image_url"][:3]
        error = offsets(images)
        outcome = "uncertain" if error is None else "success" if np.max(abs(error)) < .03 else "ongoing"
        answer = dict(observation_id=payload["current"]["observation_id"], outcome=outcome,
                      progress="unknown", evidence="PRIVATE RGB JUDGE EXPLANATION")
        return json.dumps(answer), [], []


class SynchronousMonitor:
    """Keep this large integration deterministic; async races have separate tests."""
    def __init__(self, task):
        self.engine = MonitoringSystem(RGBJudgeModel(), task)
        self.previous = None

    def submit(self, observation):
        future = Future()
        future.set_result(self.engine._assess(observation, self.previous))
        self.previous = observation
        return future

    def close(self):
        self.engine.close()


class RGBDirectModel:
    """Deterministic L2 tool response fixture, consuming only its RGB payload.

    Its camera projection convention is public and fixed for development only.
    The generated L1 is evaluated on changed, undisclosed projection conventions.
    """
    def __init__(self, give_up=False):
        self.config = SimpleNamespace(protocol="chat_completions")
        self.histories, self.give_up = [], give_up

    def complete(self, history, tools):
        self.histories.append(json.loads(json.dumps(history)))
        if self.give_up:
            name, args = "yield_control", dict(reason="Fixture L2 cannot identify the next action")
        else:
            initial = json.loads(history[1]["content"][0]["text"])["observation"]
            obs = initial
            for row in history:
                if row["role"] == "tool":
                    obs = json.loads(row["content"]).get("observation", obs)
            content = next(row["content"] for row in reversed(history) if row["role"] == "user")
            images = [Image.open(io.BytesIO(base64.b64decode(c["image_url"]["url"].split(",", 1)[1]))).convert("RGB")
                      for c in content if c["type"] == "image_url"]
            error = offsets(images)
            delta = -np.array([error[0, 0], error[1, 0], error[0, 1]]) / 1.5
            delta *= min(1, .03/max(np.linalg.norm(delta), 1e-9))
            targets = {s:dict(p, hand=None) for s,p in obs["eef"].items()}
            targets["right"]["position"] = (np.asarray(targets["right"]["position"])+delta).tolist()
            name, args = "act", dict(observation_id=obs["id"], targets=targets, steps=5,
                                     reason="Fixture uses RGB landmark error under public sensor geometry")
        call = dict(id="tool-"+str(len(self.histories)), name=name, arguments=json.dumps(args))
        native = dict(role="assistant", content="", tool_calls=[dict(id=call["id"], type="function",
                      function=dict(name=name, arguments=call["arguments"]))])
        return "", [call], [native]

    def tool_result(self, call_id, value):
        return dict(role="tool", tool_call_id=call_id, content=json.dumps(value))


class Plant(ImagePlant):
    def stop(self):
        self.stopped = True

    def release_for_human(self):
        self.stop()
        self.released = True


class NeverCalled:
    def run(self, context):
        raise AssertionError("Unexpected fallback during independent L1 validation")


class FixtureProposerModel:
    def __init__(self):
        self.config = SimpleNamespace(protocol="chat_completions", public=lambda: dict(model="offline-fixture"))
        self.requests = []
        self.unchanged = False

    def complete(self, history, tools):
        request = json.loads(history[-1]["content"][0]["text"])
        self.requests.append(request)
        rows = request["development_feedback"]
        assert {"L2", "L3"} <= {r["solved_by"] for r in rows}
        assert all(r["split"] == "development" for r in rows)
        assert "PRIVATE RGB JUDGE" not in json.dumps(request)
        assert any(e["kind"] == "human_demonstration" for r in rows for e in r["execution"])
        if self.unchanged:
            return json.dumps(dict(source=request["current_code"], memory=request["memory"],
                                   rationale="No supported improvement in this fixture")), [], []
        source = '''def policy(robot, memory):
    result = robot.align(memory["arm"], memory["moving"], memory["target"], memory["views"])
    if result["status"] != "image_aligned":
        robot.yield_control(result["reason"])
'''
        if len(self.requests) > 1:
            source += "    robot.reobserve()\n"
        return json.dumps(dict(source=source, memory=dict(arm="right", moving="hand marker",
            target="object marker", views=["front", "left"]), rationale="Fixture candidate tests reusable visual control")), [], []


class LearningLoopTests(unittest.TestCase):
    def test_rgb_rescues_evolution_and_reduced_fallback_on_new_layout(self):
        runtime = Path(__import__("os").environ.get("PHYSICALRSI_TEST_RUNTIME", "/tmp/physicalrsi-real-robot-runtime"))
        if not (runtime / "manifest.json").is_file():
            self.skipTest("Prepare isolated Python runtime")
        evidence_root = os.environ.get("ROBOT_AGENT_OFFLINE_EVIDENCE_DIR")
        if evidence_root:
            Path(evidence_root).mkdir(parents=True, exist_ok=False)
        storage = nullcontext(evidence_root) if evidence_root else tempfile.TemporaryDirectory()
        with storage as temporary:
            root = Path(temporary)
            task = Task("rgb-marker-alignment", "Align the visible hand and object markers in all views",
                        "Independent RGB check of marker alignment; this is not a grasp task")
            model = FixtureProposerModel()
            library = PolicyLibrary(root / "library", foundation=model.config.public())
            parent = library.freeze(task=task, source='def policy(robot, memory):\n    robot.yield_control("No skill yet")\n', memory={})
            feedback, before = [], []

            for level in (Level.L2, Level.L3):
                directory = root / ("development-"+level.value)
                directory.mkdir()
                plant = Plant(directory, [.085, -.065, .055])
                episode = Episode(directory / "episode", task, policy_id=parent["id"])
                direct = RGBDirectModel(give_up=level == Level.L3)

                def human_record(_):
                    # Simulated human drives only from rendered camera evidence,
                    # after ownership handoff, and writes real collector format.
                    self.assertTrue(plant.released)
                    observations = []
                    for _ in range(15):
                        obs = plant.observe()
                        observations.append(obs)
                        error = offsets([Image.open(f.path).convert("RGB") for f in obs.frames])
                        if np.max(abs(error)) < .015:
                            break
                        delta = -np.array([error[0, 0], error[1, 0], error[0, 1]])/1.5
                        plant.position += delta*min(1, .03/np.linalg.norm(delta))
                    path = directory / "human.hdf5"
                    with h5py.File(path, "w") as data:
                        for key, value in dict(hand_stack="manus_revo2", qpos_layout="interleaved_LhRh",
                            action_source="armcmd_handcmd", qpos_arm_source="measured", qpos_hand_source="measured",
                            measured_fallback_frames=0, arm_cmd_matched_frames=len(observations)).items():
                            data.attrs[key] = value
                        q = np.zeros((len(observations), 26))
                        for i, obs in enumerate(observations):
                            q[i, :3] = obs.eef["left"]["position"]
                            q[i, 13:16] = obs.eef["right"]["position"]
                        data["observations/qpos"] = q
                        data["action"] = q.copy()
                        data["timestamp_ros_nsec"] = [o.captured_ns for o in observations]
                        for index, name in enumerate(CAMERA_NAMES.values()):
                            data["observations/images/"+name] = np.stack([
                                np.asarray(Image.open(o.frames[index].path).convert("RGB")) for o in observations])
                    return str(path)

                fk = lambda q: {s:dict(position=q[i:i+3], quaternion_wxyz=[1, 0, 0, 0])
                                for s,i in (("left", 0), ("right", 13))}
                controllers = {Level.L1:L1Controller(library.load_for_execution(parent), PolicyRuntime(runtime, timeout_s=20)),
                    Level.L2:L2Controller(direct), Level.L3:TerminalTeleoperation(fk=fk, read=human_record, write=lambda _: None)}
                result = Coordinator(task=task, robot=plant, monitor=SynchronousMonitor(task), judge=None,
                                     episode=episode, controllers=controllers).run()
                self.assertEqual(result["solved_by"], level.value, result)
                self.assertEqual(result["status"], "completed")
                self.assertNotIn("PRIVATE RGB JUDGE", json.dumps(direct.histories))
                feedback.append(read_development_episode(episode.root)[1])
                before.append(result)

            cases = [dict(target=[-.075, .05, -.06], flip=True, gain=.7),
                     dict(target=[.06, .08, -.045], flip=False, gain=1)]
            def trial(task, case, candidate, directory, *, split="validation"):
                image_dir = directory.parent / (directory.name+"-cameras")
                image_dir.mkdir(parents=True)
                plant = Plant(image_dir, **case)
                episode = Episode(directory, task, split=split, policy_id=candidate["id"])
                controller = L1Controller(library.load_for_execution(candidate), PolicyRuntime(runtime, timeout_s=20),
                                          perception=PixelDetector())
                return Coordinator(task=task, robot=plant, monitor=SynchronousMonitor(task), judge=None, episode=episode,
                    controllers={Level.L1:controller, Level.L2:NeverCalled(), Level.L3:NeverCalled()}, allowed_levels=(Level.L1,)).run()

            evaluator = PairedEpisodeEvaluator(task=task, scene_provider=lambda task, n: cases, trial_runner=trial,
                                               identity=dict(scope="synthetic RGB software integration only"), episodes=2)
            proposer = DevelopmentProposer(library=library, task=task, model=model, feedback=feedback)
            result = make_harness(root / "comparison", state_root=root / "lineage", parent=parent,
                                  proposer=proposer, evaluator=evaluator).run()
            self.assertEqual(result["decision"]["decision"], "inherit_child")
            self.assertNotIn("validation-scenes", json.dumps(model.requests))
            selected = selected_policy(root / "lineage", task=task, qualification="offline")
            # A third layout appears only after selection, never in the proposal.
            followup = trial(task, dict(target=[-.055, -.075, .05], flip=True, gain=.8), selected,
                             root / "next-episode", split="development")
            self.assertTrue(followup["l1_success"], followup)
            pre, post = metrics(before), metrics([followup])
            self.assertEqual((pre["overall_success_rate"], post["overall_success_rate"]), (1, 1))
            self.assertEqual((pre["l1_success_rate"], post["l1_success_rate"]), (0, 1))
            self.assertEqual((pre["l2_episode_rate"], post["l2_episode_rate"]), (1, 0))
            self.assertEqual((pre["l3_episode_rate"], post["l3_episode_rate"]), (.5, 0))
            with self.assertRaises(ValueError):
                selected_policy(root / "lineage", task=task, qualification="physical")
            # Continue the SAME lineage with new development data and fresh
            # validation layouts. A tied candidate must retain the incumbent.
            second_feedback = feedback+[read_development_episode(root / "next-episode")[1]]
            second_proposer = DevelopmentProposer(library=library, task=task, model=model, feedback=second_feedback)
            cases = [dict(target=[.075, -.04, -.05], flip=True, gain=.8),
                     dict(target=[-.06, .065, .045], flip=False, gain=.7)]
            second_loop = make_harness(root / "comparison-2", state_root=root / "lineage", parent=selected,
                                       proposer=second_proposer, evaluator=evaluator)
            second = second_loop.run()
            self.assertEqual(second["harness"]["id"], selected["id"])
            decision = json.loads((root / "comparison-2/selection.json").read_text())["output"]
            self.assertEqual(decision["decision"], "retain_parent")
            count = len(model.requests)
            self.assertEqual(second_loop.run(), second)
            self.assertEqual(len(model.requests), count)
            # Input changes within a completed round are never a new experiment.
            with self.assertRaisesRegex(ValueError, "different development input"):
                make_harness(root / "comparison-2", state_root=root / "lineage", parent=selected,
                    proposer=DevelopmentProposer(library=library, task=task, model=model, feedback=feedback), evaluator=evaluator)
            original_outcome = second_proposer.feedback[0]["outcome"]
            second_proposer.feedback[0]["outcome"] = "failure"
            with self.assertRaisesRegex(ValueError, "development input changed"):
                second_loop.run()
            second_proposer.feedback[0]["outcome"] = original_outcome
            with self.assertRaisesRegex(ValueError, "current lineage parent"):
                make_harness(root / "stale-parent", state_root=root / "lineage", parent=parent,
                             proposer=second_proposer, evaluator=evaluator)
            self.assertEqual(len(model.requests), count)
            # A third round with unchanged code/memory retains the incumbent
            # without inventing a comparison or reusing validation layouts.
            model.unchanged = True
            third_loop = make_harness(root / "comparison-3", state_root=root / "lineage", parent=selected,
                                      proposer=second_proposer, evaluator=evaluator)
            third = third_loop.run()
            self.assertEqual(third["harness"]["id"], selected["id"])
            self.assertFalse((root / "comparison-3/validation.json").exists())
            self.assertEqual(json.loads((root / "comparison-3/retained.json").read_text())["reason"], "no_changed_candidate")
            count = len(model.requests)
            self.assertEqual(third_loop.run(), third)
            self.assertEqual(len(model.requests), count)
            write_new(root / "summary.json", dict(
                scope="software integration: synthetic marker alignment, not physical grasp",
                qualification="offline", models="deterministic fixtures, not a GPT capability benchmark",
                before=pre, after=post, selected_policy_id=selected["id"],
                decision=result["decision"]["decision"], held_out_layouts=2,
                post_selection_new_layouts=1, physical_qualification=False,
                second_round_decision=decision["decision"], completed_rounds=3,
                third_round_reason="no_changed_candidate",
                checks=["L2 tool control", "L3 HDF5 import", "RGB-only independent monitoring",
                        "private judgment isolation", "CodeProposer source/memory update",
                        "isolated L1 visual control", "SelfHarness paired comparison",
                        "lineage selection", "lower fallback rates", "offline/physical separation"]))


if __name__ == "__main__":
    unittest.main()
