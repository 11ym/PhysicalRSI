"""New dual-system contracts with fake devices; no SDK, ROS, model API or simulator."""
import base64
from concurrent.futures import Future
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from queue import Queue
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PhysicalRSI_baselines.real_robot.capabilities import validate_direct_action
from PhysicalRSI_baselines.real_robot.contracts import (Assessment, CapabilityUnavailable, ExecutionUncertain,
                                    Frame, HumanDecisionPending, Level, Observation, Outcome, Progress, SensorUnavailable, Task)
from PhysicalRSI_baselines.real_robot.controllers import L2Controller
from PhysicalRSI_baselines.real_robot.controllers import L1Controller
from PhysicalRSI_baselines.real_robot.coordinator import Coordinator
from PhysicalRSI_baselines.real_robot.evidence import Episode, metrics
from PhysicalRSI_baselines.real_robot.monitor import MonitoringSystem, TerminalJudge
from PhysicalRSI_baselines.real_robot.policy_runtime import PolicyRuntime, validate_source
from PhysicalRSI_baselines.real_robot.demonstrations import import_hdf5
from PhysicalRSI_baselines.real_robot.library import PolicyLibrary

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j7x8AAAAASUVORK5CYII=")


class Model:
    def __init__(self, replies):
        self.replies, self.histories = iter(replies), []
        self.config = SimpleNamespace(protocol="chat_completions", public=lambda: dict(model="test-only"))

    def complete(self, history, tools):
        self.histories.append(json.loads(json.dumps(history)))
        return next(self.replies)

    def tool_result(self, call_id, value):
        return dict(role="tool", tool_call_id=call_id, content=json.dumps(value))


class ScriptedMonitor:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)

    def submit(self, obs):
        outcome = next(self.outcomes, Outcome.ONGOING)
        f = Future()
        f.set_result(Assessment(obs.id, outcome, Progress.UNKNOWN, "PRIVATE MONITOR EXPLANATION"))
        return f

    def close(self):
        pass


class FakeRobot:
    def __init__(self, observation):
        self.observation = observation
        self.actions, self.stops, self.releases = [], 0, 0
        self.count = 0

    def observe(self):
        self.count += 1
        return replace(self.observation, id=f"obs-{self.count}")

    def hold(self):
        pass

    def stop(self):
        self.stops += 1

    def release_for_human(self):
        self.releases += 1

    def act_eef(self, targets, steps, *, checkpoint, observation=None):
        self.actions.append(dict(targets=targets, steps=steps))
        return dict(executed_ticks=steps)


class Unavailable:
    def run(self, context):
        raise CapabilityUnavailable("No reusable policy for this task")


class ActionController:
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def run(self, context):
        self.calls += 1
        def invoke():
            if self.fail:
                raise OSError("Unknown physical effect")
            return dict(executed_ticks=1)
        context.action("test_action", {}, invoke)


class SystemTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        frames = []
        now = time.time_ns()
        for camera in ("front", "left", "right"):
            p = self.root / (camera + ".png")
            p.write_bytes(PNG)
            frames.append(Frame(camera, str(p), now, hashlib.sha256(PNG).hexdigest()))
        pose = dict(position=[0, 0, 0], quaternion_wxyz=[1, 0, 0, 0])
        self.obs = Observation("obs-1", now, tuple(frames), (0.0,)*26,
                               dict(left=dict(pose), right=dict(pose)))
        self.task = Task("insert", "将杆插入孔", "视觉确认插入；无法确认接触时由人判断")

    def episode(self):
        return Episode(self.root / "episode", self.task)

    def coordinator(self, monitor, controllers, judge=None):
        return Coordinator(task=self.task, robot=FakeRobot(self.obs), monitor=monitor,
                           judge=judge, episode=self.episode(), controllers=controllers)

    def test_three_views_and_hashes_are_required(self):
        self.obs.validate()
        with self.assertRaises(ValueError):
            replace(self.obs, frames=self.obs.frames[:2]).validate()
        Path(self.obs.frames[0].path).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.obs.public()

    def test_saved_rescue_learning_reopens_only_public_development_evidence(self):
        from PhysicalRSI_baselines.real_robot.learning import read_development_episode, learn_from_episodes
        episode = self.episode()
        episode.observation(self.obs)
        episode.event("controller_enter", level="L1")
        episode.event("controller_enter", level="L2")
        episode.event("controller_enter", level="L3")
        episode.event("human_demonstration", level="L3", trajectory=dict(measured_qpos26=[[0.]*26]),
                      manifest=dict(replay_allowed=False))
        episode.assessment(Assessment(self.obs.id, Outcome.SUCCESS, Progress.IMPROVED, "PRIVATE SECRET HUMAN LABEL", "human"))
        episode.finish(outcome=Outcome.SUCCESS, solved_by=Level.L3)
        task, feedback, result = read_development_episode(episode.root)
        self.assertEqual(task, self.task)
        self.assertEqual(feedback["solved_by"], "L3")
        self.assertNotIn("PRIVATE SECRET", json.dumps(feedback))
        # Private files are not needed by the offline importer/proposer.
        for p in episode.private.iterdir():
            p.unlink()
        source = "def policy(robot, memory):\n    robot.look(['rod center', 'thumb tip'])\n"
        model = Model([(json.dumps(dict(source=source, memory=dict(lesson="observe before acting"),
                                        rationale="Learn a public observation precondition")), [], [])])
        library = PolicyLibrary(self.root / "library", foundation=model.config.public())
        record = learn_from_episodes([episode.root], library=library, model=model, output=self.root / "learning")
        self.assertFalse(record["promoted"])
        self.assertEqual(record["metrics"]["l3_calls"], 1)
        self.assertEqual(library.load(record["candidate"])["source"], source)
        self.assertNotIn("PRIVATE SECRET", json.dumps(model.histories))
        self.assertTrue((self.root / "learning/development-public.json").is_file())

    def test_saved_learning_rejects_validation_changed_events_and_images(self):
        from PhysicalRSI_baselines.real_robot.learning import read_development_episode
        episode = self.episode()
        episode.observation(self.obs)
        episode.event("controller_enter", level="L2")
        episode.finish(outcome=Outcome.UNCERTAIN, solved_by=None, status="interrupted")
        _, row, _ = read_development_episode(episode.root)
        self.assertEqual(row["outcome"], "uncertain")
        event_path = episode.public / "event-000000.json"
        original = event_path.read_text()
        event_path.write_text(original.replace('"L2"', '"L1"'))
        with self.assertRaisesRegex(ValueError, "history differs"):
            read_development_episode(episode.root)
        event_path.write_text(original)
        meta = episode.root / "episode.json"
        saved = meta.read_text()
        meta.write_text(saved.replace('"development"', '"validation"'))
        with self.assertRaisesRegex(ValueError, "Only development"):
            read_development_episode(episode.root)
        meta.write_text(saved)
        next((episode.public / "images").iterdir()).write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            read_development_episode(episode.root)

    def test_saved_learning_rejects_private_image_paths_and_duplicate_input(self):
        from PhysicalRSI_baselines.real_robot.learning import read_development_episode, learn_from_episodes
        episode = self.episode()
        episode.observation(self.obs)
        episode.event("controller_enter", level="L2")
        episode.finish(outcome=Outcome.FAILURE, solved_by=None)
        with self.assertRaisesRegex(ValueError, "distinct"):
            learn_from_episodes([episode.root, episode.root], library=None, model=None, output=self.root / "bad")
        path = next((episode.public / "observations").iterdir())
        packet = json.loads(path.read_text())
        private_image = episode.private / "privileged.png"
        private_image.write_bytes(PNG)
        packet["frames"][0]["path"] = str(private_image)
        path.write_text(json.dumps(packet))
        with self.assertRaisesRegex(ValueError, "retained execution"):
            read_development_episode(episode.root)

    def test_execution_cannot_reuse_validation_after_adapter_changes(self):
        from PhysicalRSI_baselines.real_robot.library import dependency_files
        library = PolicyLibrary(self.root / "library", foundation=dict(model="test-only"))
        candidate = library.freeze(task=self.task, source="def policy(robot, memory):\n    return None\n", memory={})
        library.load_for_execution(candidate)
        changed = dependency_files()
        changed.pop(next(iter(changed)))
        with patch("PhysicalRSI_baselines.real_robot.library.dependency_files", return_value=changed):
            with self.assertRaisesRegex(ValueError, "adapter/API changed"):
                library.load_for_execution(candidate)
        # Its immutable source remains available for development, not execution qualification.
        self.assertIn("policy", library.load(candidate)["source"])

    def test_model_change_invalidates_comparison_but_mailbox_routing_does_not(self):
        foundation = dict(kind="external-model-mailbox", model="test-only", session="run-a")
        library = PolicyLibrary(self.root / "library", foundation=foundation)
        candidate = library.freeze(task=self.task, source="def policy(robot, memory):\n    return None\n", memory={})
        library.load_for_execution(candidate, foundation=dict(foundation, session="run-b"))
        other_run = PolicyLibrary(self.root / "library", foundation=dict(foundation, session="run-b"))
        other_candidate = other_run.freeze(task=self.task, source="def policy(robot, memory):\n    return None\n", memory={})
        self.assertEqual(candidate["id"], other_candidate["id"])
        with self.assertRaisesRegex(ValueError, "foundation changed"):
            library.load_for_execution(candidate, foundation=dict(foundation, model="different-model"))


    def test_evolution_method_identity_excludes_changing_round_data_and_reply_routing(self):
        from PhysicalRSI_baselines.real_robot.evolution import DevelopmentProposer
        def model(session):
            return SimpleNamespace(config=SimpleNamespace(public=lambda: dict(
                kind="external-model-mailbox", model="fixture", session=session)))
        a = DevelopmentProposer(library=None, task=self.task, model=model("first"),
                                feedback=[dict(task=self.task.id, split="development", outcome="failure")])
        b = DevelopmentProposer(library=None, task=self.task, model=model("second"),
                                feedback=[dict(task=self.task.id, split="development", outcome="success")])
        self.assertEqual(a.identity(), b.identity())
        self.assertNotEqual(a.feedback_sha256, b.feedback_sha256)
        changed_criterion = replace(self.task, success_criteria="Different success criterion")
        c = DevelopmentProposer(library=None, task=changed_criterion, model=model("third"), feedback=b.feedback)
        self.assertNotEqual(b.identity(), c.identity())

    def test_paired_evaluation_rejects_mixed_qualification_and_unearned_success(self):
        from PhysicalRSI_baselines.real_robot.evolution import PairedEpisodeEvaluator
        from PhysicalRSI_baselines.real_robot.evidence import write_new
        from PhysicalRSI_core.infra.storage import file_digest
        for index, overrides in enumerate((dict(qualification="physical"), dict(solved_by=None))):
            directory = self.root / str(index)
            directory.mkdir()
            scenes = directory / "validation-scenes.json"
            write_new(scenes, {"case":{}})
            def trial(task, case, candidate, output):
                result = dict(split="validation", policy_id=candidate["id"], task=task.id, status="completed",
                              outcome="success", l1_success=True, solved_by="L1", qualification="offline",
                              l2_calls=0, l3_calls=0)
                result.update(overrides)
                write_new(output / "result.json", result)
                return result
            evaluator = PairedEpisodeEvaluator(task=self.task, scene_provider=lambda *a: [{}], trial_runner=trial,
                                               identity=dict(scope="test only"), episodes=1)
            cohort = dict(admission_evidence={str(scenes):file_digest(scenes)}, layouts={self.task.id:["case"]})
            with self.assertRaises(ValueError):
                evaluator.evaluate(dict(id="candidate"), {}, cohort, directory)

    def visual_reply(self):
        return dict(observation_id=self.obs.id, views={camera: {
            "rod center": dict(visibility="visible", uv=[.4, .3], confidence=.8),
            "thumb tip": dict(visibility="visible", uv=[.6, .7], confidence=.9)}
            for camera in ("front", "left", "right")})

    def test_visual_features_use_rgb_only_and_compute_offsets(self):
        from PhysicalRSI_baselines.real_robot.visual_perception import VisualPerception
        raw = self.visual_reply()
        raw["views"]["left"]["rod center"] = dict(visibility="occluded", uv=None, confidence=.9)
        model = Model([(json.dumps(raw), [], [])])
        result = VisualPerception(model).look(self.obs, ["rod center", "thumb tip"],
                                             deadline=time.monotonic()+10)
        front = result["views"]["front"]["offsets"][0]
        self.assertAlmostEqual(front["delta_uv"][0], .2)
        self.assertAlmostEqual(front["delta_uv"][1], .4)
        self.assertEqual(front["confidence"], .8)
        self.assertIsNone(result["views"]["left"]["offsets"][0]["delta_uv"])
        content = model.histories[0][1]["content"]
        payload = json.loads(content[0]["text"])
        self.assertEqual(set(payload), {"observation_id", "features", "cameras"})
        self.assertEqual(len([c for c in content if c["type"] == "image_url"]), 3)
        self.assertEqual(result["frames"][0]["sha256"], self.obs.frames[0].sha256)
        self.assertNotIn("PRIVATE MONITOR", json.dumps(model.histories))
        self.assertNotIn("qpos26", json.dumps(model.histories))

    def test_visual_features_reject_privileged_or_invalid_outputs(self):
        from PhysicalRSI_baselines.real_robot.visual_perception import project_features
        cases = []
        raw = self.visual_reply(); raw["success"] = True; cases.append(raw)
        raw = self.visual_reply(); raw["observation_id"] = "old"; cases.append(raw)
        raw = self.visual_reply(); del raw["views"]["left"]; cases.append(raw)
        raw = self.visual_reply(); raw["views"]["front"]["rod center"]["pose"] = [1, 2, 3]; cases.append(raw)
        for field, value in [("uv", [1.1, .3]), ("uv", [True, .2]), ("uv", [float('nan'), .2]),
                             ("visibility", "occluded"), ("confidence", True), ("confidence", float('inf'))]:
            raw = self.visual_reply(); raw["views"]["front"]["rod center"][field] = value; cases.append(raw)
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                project_features(raw, self.obs, ["rod center", "thumb tip"])

    def test_visual_features_reject_tools_late_output_and_changed_images(self):
        from PhysicalRSI_baselines.real_robot.visual_perception import VisualPerception
        raw = self.visual_reply()
        model = Model([(json.dumps(raw), [dict(name="act")], [])])
        with self.assertRaises(CapabilityUnavailable):
            VisualPerception(model).look(self.obs, ["rod center", "thumb tip"], deadline=time.monotonic()+10)
        model = Model([(json.dumps(raw), [], [])])
        with patch("PhysicalRSI_baselines.real_robot.visual_perception.time.monotonic", side_effect=[0, 2]):
            with self.assertRaises(CapabilityUnavailable):
                VisualPerception(model).look(self.obs, ["rod center", "thumb tip"], deadline=1)
        def corrupt(history, tools):
            Path(self.obs.frames[0].path).write_bytes(b"replaced during inference")
            return json.dumps(raw), [], []
        model.complete = corrupt
        with self.assertRaises(CapabilityUnavailable):
            VisualPerception(model).look(self.obs, ["rod center", "thumb tip"], deadline=time.monotonic()+10)

    def test_isolated_l1_branches_on_visual_relation_without_metric_scene(self):
        from PhysicalRSI_baselines.real_robot.visual_perception import VisualPerception
        runtime = Path(__import__("os").environ.get("PHYSICALRSI_TEST_RUNTIME", "/tmp/physicalrsi-real-robot-runtime"))
        if not (runtime / "manifest.json").is_file():
            self.skipTest("Prepare the isolated Python runtime")
        source = '''def policy(robot, memory):
    view = robot.look(["rod center", "thumb tip"])["views"]["front"]
    delta = view["offsets"][0]["delta_uv"]
    if delta is None:
        return
    if delta[0] > 0:
        robot.gripper("right", memory["a"])
    else:
        robot.gripper("right", memory["b"])
'''
        # A branch/transport integration test, not a grasp controller or real
        # localization benchmark. The same source handles changed image input.
        for index, (thumb_u, expected) in enumerate([(.6, "open"), (.2, "close"), (None, None)]):
            raw = self.visual_reply()
            point = raw["views"]["front"]["thumb tip"]
            point.update(uv=None if thumb_u is None else [thumb_u, .7],
                         visibility="occluded" if thumb_u is None else "visible")
            model = Model([(json.dumps(raw), [], [])])
            episode = Episode(self.root / str(index), self.task)
            robot = FakeRobot(self.obs)
            robot.gripper = lambda arm, action, mode, checkpoint: robot.actions.append((arm, action)) or {}
            context = SimpleNamespace(robot=robot, episode=episode,
                                      checkpoint=lambda **kw: self.obs,
                                      action=lambda method, args, invoke: invoke())
            L1Controller(dict(source=source, memory=dict(a="open", b="close")),
                         PolicyRuntime(runtime, timeout_s=10),
                         perception=VisualPerception(model)).run(context)
            self.assertEqual(robot.actions, [] if expected is None else [("right", expected)])
            self.assertEqual(episode.events[0]["kind"], "visual_features")

    def test_privileged_keys_cannot_enter_sensor_observation(self):
        packet = self.obs.public()
        packet["judge_answer"] = "secret"
        with self.assertRaises(ValueError):
            Observation.from_public(packet)
        with self.assertRaises(ValueError):
            replace(self.obs, eef=dict(self.obs.eef, oracle_pose={})).validate()

    def test_camera_timestamp_skew_and_live_age(self):
        bad = replace(self.obs.frames[0], captured_ns=self.obs.captured_ns-10**9)
        with self.assertRaises(ValueError):
            replace(self.obs, frames=(bad, *self.obs.frames[1:])).validate()
        with self.assertRaises(ValueError):
            self.obs.validate(live=True, max_age_s=0)

    def test_monitor_gets_actual_three_images_not_judgment_or_robot_control(self):
        answer = dict(observation_id=self.obs.id, outcome="ongoing", progress="unknown", evidence="Unclear contact")
        model = Model([(json.dumps(answer), [], [])])
        monitor = MonitoringSystem(model, self.task)
        try:
            self.assertEqual(monitor.submit(self.obs).result().outcome, Outcome.ONGOING)
        finally:
            monitor.close()
        message = model.histories[0][-1]["content"]
        self.assertEqual(sum(p["type"] == "image_url" for p in message), 3)
        self.assertNotIn("qpos26", message[0]["text"])

    def test_human_assessment_is_terminal_dialogue_not_l3(self):
        judge = TerminalJudge(read=lambda _: "完成：我从现场确认已插入", write=lambda _: None)
        c = self.coordinator(ScriptedMonitor([Outcome.UNCERTAIN]), {l: Unavailable() for l in Level}, judge)
        result = c.run()
        self.assertEqual(result["outcome"], "success")
        self.assertEqual(result["human_judgments"], 1)
        self.assertEqual(result["l3_calls"], 0)
        self.assertIsNone(result["solved_by"])
        feedback = json.dumps(c.episode.development_feedback(result), ensure_ascii=False)
        self.assertNotIn("我从现场确认", feedback)
        self.assertNotIn("PRIVATE MONITOR", feedback)

    def test_uncertain_human_assessment_does_not_pause_exploration(self):
        judge = TerminalJudge(read=lambda _: "无法判断：看不到接触点", write=lambda _: None)
        action = ActionController()
        c = self.coordinator(ScriptedMonitor([Outcome.UNCERTAIN]), {l: action for l in Level}, judge)
        result = c.run()
        self.assertEqual(result["status"], "exploration_unfinished")
        self.assertEqual(result["outcome"], "uncertain")
        self.assertEqual(action.calls, 3)

    def test_pending_human_judgment_does_not_block_l2(self):
        future = Future()
        judge = SimpleNamespace(submit=lambda *args: future)
        action = ActionController()
        c = self.coordinator(ScriptedMonitor([Outcome.UNCERTAIN, Outcome.SUCCESS]),
                             {Level.L1: Unavailable(), Level.L2: action, Level.L3: Unavailable()}, judge)
        result = c.run()
        self.assertFalse(future.done())
        self.assertEqual(action.calls, 1)
        self.assertEqual(result["solved_by"], "L2")

    def test_validation_waits_for_final_images_after_delayed_old_assessment(self):
        # Model calls complete only on result(), making the stale-image race
        # deterministic without timing a worker thread or sleeping.
        seen = []
        class Late:
            def __init__(self, answer):
                self.answer = answer
            def done(self):
                return False
            def result(self):
                return self.answer
        class Monitor:
            def submit(self, obs):
                seen.append(obs.id)
                return Late(Assessment(obs.id, Outcome.ONGOING if len(seen) == 1 else Outcome.SUCCESS,
                                       Progress.UNKNOWN, "PRIVATE FINAL IMAGE TEST"))
            def close(self):
                pass
        c = self.coordinator(Monitor(), {Level.L1:ActionController(), Level.L2:Unavailable(), Level.L3:Unavailable()})
        c.allowed_levels = (Level.L1,)
        result = c.run()
        self.assertTrue(result["l1_success"], result)
        self.assertEqual(len(seen), 2)
        self.assertNotEqual(seen[0], seen[1])
        self.assertEqual(result["l2_calls"], 0)

    def test_uncertain_final_validation_is_not_scored_as_failure(self):
        c = self.coordinator(ScriptedMonitor([Outcome.UNCERTAIN]*8),
                             {Level.L1:ActionController(), Level.L2:Unavailable(), Level.L3:Unavailable()})
        c.allowed_levels = (Level.L1,)
        result = c.run()
        self.assertEqual(result["outcome"], "uncertain")
        self.assertEqual(result["status"], "evaluation_uncertain")
        self.assertFalse(result["l1_success"])

    def test_final_validation_waits_for_human_and_reasks_after_stale_answer(self):
        requests = []
        def submit(task, observed, assessment):
            future = Future()
            requests.append((observed, future))
            return future
        c = self.coordinator(ScriptedMonitor([Outcome.UNCERTAIN]*12),
                             {Level.L1:ActionController(), Level.L2:Unavailable(), Level.L3:Unavailable()},
                             SimpleNamespace(submit=submit))
        c.allowed_levels = (Level.L1,)
        replies = []
        def reply_during_wait(_):
            observed, future = requests[-1]
            self.assertFalse(future.done())
            replies.append(observed.id)
            # The pre-action success must be saved, but cannot finish the trial.
            future.set_result(Assessment(observed.id, Outcome.SUCCESS, Progress.IMPROVED,
                                         "PRIVATE HUMAN CONTACT INFORMATION", "human"))
        with patch("PhysicalRSI_baselines.real_robot.coordinator.time.sleep", side_effect=reply_during_wait):
            result = c.run()
        self.assertEqual(len(requests), 2)
        self.assertNotEqual(replies[0], replies[1])
        self.assertEqual(requests[-1][0].id, c.last_observation.id)
        self.assertEqual(result["human_judgments"], 2)
        self.assertTrue(result["l1_success"], result)
        self.assertEqual(result["l2_calls"] + result["l3_calls"], 0)
        self.assertNotIn("PRIVATE HUMAN", json.dumps(c.episode.development_feedback(result)))

    def test_final_human_timeout_is_uncertain_and_does_not_call_l2_or_l3(self):
        future = Future()
        c = self.coordinator(ScriptedMonitor([Outcome.UNCERTAIN]*12),
                             {Level.L1:ActionController(), Level.L2:Unavailable(), Level.L3:Unavailable()},
                             SimpleNamespace(submit=lambda *args: future))
        c.allowed_levels = (Level.L1,)
        c.human_judgment_timeout_s = .1
        clock = [0.]
        def advance(seconds):
            clock[0] += seconds
        with patch("PhysicalRSI_baselines.real_robot.coordinator.time.monotonic", side_effect=lambda: clock[0]), \
                patch("PhysicalRSI_baselines.real_robot.coordinator.time.sleep", side_effect=advance):
            result = c.run()
        self.assertEqual(result["outcome"], "uncertain")
        self.assertEqual(result["status"], "awaiting_human_judgment")
        self.assertFalse(future.done())
        self.assertEqual(result["l2_calls"] + result["l3_calls"], 0)
        self.assertEqual(c.robot.stops, 1)

    def test_explicit_unable_to_judge_does_not_keep_reasking_at_validation_end(self):
        judge = TerminalJudge(read=lambda _: "无法判断：接触点不可见", write=lambda _: None)
        c = self.coordinator(ScriptedMonitor([Outcome.UNCERTAIN]*12),
                             {Level.L1:ActionController(), Level.L2:Unavailable(), Level.L3:Unavailable()}, judge)
        c.allowed_levels = (Level.L1,)
        with patch("PhysicalRSI_baselines.real_robot.coordinator.time.sleep", side_effect=AssertionError("Unnecessary wait")):
            result = c.run()
        self.assertEqual(result["outcome"], "uncertain")
        self.assertEqual(result["status"], "evaluation_uncertain")

    def test_stale_human_answer_does_not_replace_current_monitor_assessment(self):
        c = self.coordinator(ScriptedMonitor([]), {l: Unavailable() for l in Level})
        current = Assessment("current", Outcome.ONGOING, Progress.IMPROVED, "current")
        c.action_revision = 1
        c._accept_assessment(current, 1, Level.L1)
        c._accept_assessment(Assessment("old", Outcome.UNCERTAIN, Progress.UNKNOWN, "old", "human"),
                             0, Level.L1)
        self.assertIs(c.last_assessment, current)

    def test_human_question_identifies_the_exact_images_without_sending_them_to_execution(self):
        captured = []
        dialogue = SimpleNamespace(request=lambda *args: captured.append(args))
        judge = TerminalJudge(dialogue=dialogue)
        judge.submit(self.task, self.obs, Assessment(self.obs.id, Outcome.UNCERTAIN, Progress.UNKNOWN,
                                                    "PRIVATE MONITOR CONTACT QUESTION"))
        kind, prompt, parse = captured[0]
        self.assertEqual(kind, "J")
        self.assertIn(self.obs.id, prompt)
        for frame in self.obs.frames:
            self.assertIn(frame.path, prompt)
        self.assertEqual(parse("没改善：PRIVATE HUMAN ANSWER").observation_id, self.obs.id)

    def test_terminal_routes_judgment_and_takeover_without_crossing_answers(self):
        from PhysicalRSI_baselines.real_robot.dialogue import TerminalDialogue
        lines, output = Queue(), Queue()
        def read(_):
            value = lines.get()
            if value is None:
                raise EOFError()
            return value
        dialogue = TerminalDialogue(read=read, write=output.put)
        try:
            judged = TerminalJudge(dialogue=dialogue).submit(self.task, self.obs, Assessment(
                self.obs.id, Outcome.UNCERTAIN, Progress.UNKNOWN, "PRIVATE CONTACT QUESTION"))
            takeover = dialogue.request("T", "Provide demonstration directory")
            self.assertIn("[J1]", output.get(timeout=1))
            self.assertIn("[T2]", output.get(timeout=1))
            lines.put("完成")
            self.assertIn("请用问题编号", output.get(timeout=1))
            self.assertFalse(judged.done())
            self.assertFalse(takeover.done())
            lines.put("J1 不是合法判定")
            self.assertIn("请重新回答", output.get(timeout=1))
            self.assertFalse(takeover.done())
            lines.put("J1 完成：PRIVATE CONTACT ANSWER")
            answer = judged.result(timeout=1)
            self.assertEqual(answer.outcome, Outcome.SUCCESS)
            self.assertEqual(answer.source, "human")
            self.assertFalse(takeover.done())
            lines.put("T2 /recorded/demo")
            self.assertEqual(takeover.result(timeout=1), "/recorded/demo")
        finally:
            dialogue.close()
            lines.put(None)

    def test_terminal_eof_leaves_judgment_pending_instead_of_inventing_failure(self):
        from PhysicalRSI_baselines.real_robot.dialogue import TerminalDialogue
        def ended(_):
            raise EOFError()
        dialogue = TerminalDialogue(read=ended, write=lambda _: None)
        self.addCleanup(dialogue.close)
        future = dialogue.request("J", "Judge task")
        with self.assertRaises(HumanDecisionPending):
            future.result(timeout=1)

    def test_sequential_episodes_cancel_old_questions_and_share_one_terminal_reader(self):
        from PhysicalRSI_baselines.real_robot.dialogue import TerminalDialogue
        import threading
        lines, readers = Queue(), []
        def read(_):
            readers.append(threading.get_ident())
            value = lines.get()
            if value is None:
                raise EOFError()
            return value
        dialogue = TerminalDialogue(read=read, write=lambda _: None)
        try:
            old = dialogue.request("J", "Old unresolved judgment")
            dialogue.cancel_pending()
            with self.assertRaises(HumanDecisionPending):
                old.result(timeout=1)
            self.assertFalse(dialogue.closed)
            new = dialogue.request("S", "Restore next scene")
            lines.put("S2 就绪")
            self.assertEqual(new.result(timeout=1), "就绪")
            self.assertEqual(len(set(readers)), 1)
        finally:
            dialogue.close()
            lines.put(None)

    def test_monitor_failure_means_keep_exploring_in_development(self):
        action = ActionController()
        c = self.coordinator(ScriptedMonitor([Outcome.FAILURE, Outcome.SUCCESS]),
                             {Level.L1: action, Level.L2: Unavailable(), Level.L3: Unavailable()})
        result = c.run()
        self.assertEqual(action.calls, 1)
        self.assertEqual(result["outcome"], "success")

    def test_delayed_success_from_before_action_cannot_end_later_controller(self):
        pending = Future()
        observations = []
        class Monitor:
            def submit(self, obs):
                observations.append(obs)
                return pending if len(observations) == 1 else Future()
            def close(self):
                pass
        class Action:
            def run(self, context):
                context.action("test", {}, lambda: pending.set_result(Assessment(
                    observations[0].id, Outcome.SUCCESS, Progress.IMPROVED, "OLD VIEW")))
        c = self.coordinator(Monitor(), {Level.L1: Action(), Level.L2: Unavailable(), Level.L3: Unavailable()})
        result = c.run()
        self.assertEqual(result["status"], "exploration_unfinished")
        self.assertIsNone(result["solved_by"])

    def test_episode_retains_pixels_after_camera_cache_is_removed(self):
        episode = self.episode()
        episode.observation(self.obs)
        episode.observation(self.obs)
        for frame in self.obs.frames:
            Path(frame.path).unlink()
        retained = Observation.from_public(episode.observations[self.obs.id])
        retained.validate()
        self.assertTrue(all(Path(f.path).is_relative_to(episode.root) for f in retained.frames))

    def test_l2_rescue_success_is_not_l1_success(self):
        l2, l3 = ActionController(), ActionController()
        c = self.coordinator(ScriptedMonitor([Outcome.ONGOING, Outcome.SUCCESS]),
                             {Level.L1: Unavailable(), Level.L2: l2, Level.L3: l3})
        result = c.run()
        self.assertEqual(result["solved_by"], "L2")
        self.assertFalse(result["l1_success"])
        self.assertEqual((l2.calls, l3.calls), (1, 0))
        feedback = c.episode.development_feedback(result)
        self.assertTrue(any(e["kind"] == "action_completed" for e in feedback["execution"]))
        self.assertNotIn("PRIVATE MONITOR", json.dumps(feedback))

    def test_unknown_physical_effect_does_not_fall_back(self):
        l2 = ActionController()
        c = self.coordinator(ScriptedMonitor([Outcome.ONGOING]),
                             {Level.L1: ActionController(fail=True), Level.L2: l2, Level.L3: l2})
        result = c.run()
        self.assertEqual(result["status"], "needs_reconciliation")
        self.assertEqual(l2.calls, 0)
        self.assertEqual(c.robot.stops, 1)

    def test_stop_during_success_handoff_is_not_reported_as_completed_success(self):
        c = self.coordinator(ScriptedMonitor([Outcome.ONGOING, Outcome.SUCCESS]),
                             {level: ActionController() for level in Level})
        def interrupted_handoff():
            self.assertEqual(c.robot.stops, 0)
            raise ExecutionUncertain("operator stop while holding object")
        c.before_success_stop = interrupted_handoff
        result = c.run()
        self.assertEqual((result["outcome"], result["status"]), ("uncertain", "needs_reconciliation"))
        self.assertEqual(c.robot.stops, 1)

    def test_camera_loss_after_confirmed_action_is_not_unknown_actuation(self):
        from PhysicalRSI_baselines.real_robot.physical_port import PhysicalRobot
        job_id = "a" * 32
        owner = self.root / "owner"
        job = owner / "jobs" / job_id
        job.mkdir(parents=True)
        receipt = dict(id=job_id, status="completed", ticks=5)
        (job / "receipt.json").write_text(json.dumps(receipt))
        port = PhysicalRobot.__new__(PhysicalRobot)
        port.workspace, port.evidence = owner, self.root / "retained"
        port.last_observation = self.obs
        requests = []
        results = iter([dict(id=job_id), dict(status="running"), receipt])
        def rpc(method, arguments):
            requests.append((method, arguments))
            return next(results)
        port.rpc = rpc
        def lost_camera():
            raise SensorUnavailable("RGB publisher disappeared")
        class Action:
            def run(self, context):
                context.action("act_eef", {}, lambda: port._job("direct", {}, 5, lost_camera))
        fallback = ActionController()
        c = self.coordinator(ScriptedMonitor([Outcome.ONGOING]),
                             {Level.L1: Action(), Level.L2: fallback, Level.L3: fallback})
        with patch("PhysicalRSI_baselines.real_robot.physical_port.uuid.uuid4", return_value=SimpleNamespace(hex=job_id)):
            result = c.run()
        self.assertEqual(result["status"], "awaiting_sensor_recovery")
        self.assertEqual(result["outcome"], "uncertain")
        self.assertEqual(fallback.calls, 0)
        self.assertEqual([m for m, _ in requests], ["submit", "job", "job"])
        self.assertTrue(all(a["id"] == job_id for m, a in requests))
        events = c.episode.events
        self.assertTrue(any(e["kind"] == "action_completed" for e in events))
        self.assertFalse(any(e["kind"] == "action_effect_uncertain" for e in events))
        self.assertTrue((port.evidence / "hardware" / job_id / "receipt.json").is_file())
        self.assertTrue((c.episode.private / "interruption.json").is_file())
        self.assertNotIn("RGB publisher disappeared", json.dumps(c.episode.development_feedback(result)))

    def test_sensor_loss_without_completion_proof_still_requires_reconciliation(self):
        class Action:
            def run(self, context):
                def invoke():
                    raise SensorUnavailable("RGB lost with no receipt")
                context.action("act_eef", {}, invoke)
        fallback = ActionController()
        c = self.coordinator(ScriptedMonitor([Outcome.ONGOING]),
                             {Level.L1: Action(), Level.L2: fallback, Level.L3: fallback})
        result = c.run()
        self.assertEqual(result["status"], "needs_reconciliation")
        self.assertEqual(fallback.calls, 0)

    def test_observation_distinguishes_camera_loss_from_robot_connection_loss(self):
        from PhysicalRSI_baselines.real_robot.physical_port import PhysicalRobot
        port = PhysicalRobot.__new__(PhysicalRobot)
        port.released, port.heartbeat_error = False, None
        def unavailable_camera():
            raise TimeoutError("No frames")
        port.cameras = SimpleNamespace(capture=unavailable_camera)
        with self.assertRaises(SensorUnavailable):
            port.observe()
        port.cameras = SimpleNamespace(capture=lambda: self.obs.frames)
        def lost_hardware(*args):
            raise ExecutionUncertain("Snapshot IPC failed")
        port.rpc = lost_hardware
        with self.assertRaises(ExecutionUncertain):
            port.observe()

    def test_validation_cannot_be_exported_to_learning(self):
        episode = Episode(self.root / "validation", self.task, split="validation")
        with self.assertRaises(ValueError):
            episode.development_feedback({})

    def test_metrics_include_uncertain_and_separate_human_judgments(self):
        base = dict(qualification="offline", outcome="success", l1_success=False,
                    l2_calls=1, l3_calls=0, human_judgments=2)
        result = metrics([base, dict(base, outcome="uncertain", l2_calls=0)])
        self.assertEqual(result["overall_success_rate"], .5)
        self.assertEqual(result["l1_success_rate"], 0)
        self.assertEqual(result["l3_episode_rate"], 0)
        with self.assertRaises(ValueError):
            metrics([base, dict(base, qualification="physical")])

    def test_direct_action_keeps_reference_loop_and_hand_adaptation(self):
        targets = {a: dict(p, hand=None) for a, p in self.obs.eef.items()}
        action = dict(observation_id=self.obs.id, targets=targets, steps=5, reason="Align the visible tip")
        validate_direct_action(action, self.obs)
        for bad in (True, 0, 6):
            with self.assertRaises(ValueError):
                validate_direct_action(dict(action, steps=bad), self.obs)
        with self.assertRaises(ValueError):
            validate_direct_action(dict(action, observation_id="old"), self.obs)
        targets["left"]["position"] = [.051, 0, 0]
        with self.assertRaises(ValueError):
            validate_direct_action(action, self.obs)

    def test_l2_semantic_gripper_binds_observation_and_receives_measured_result(self):
        from PhysicalRSI_baselines.real_robot.controllers import direct_tools
        calls = []
        def reply(oid):
            return "", [dict(id=oid, name="gripper", arguments=json.dumps(dict(
                observation_id=oid, arm="right", action="close", mode="wrap", reason="Rod within the hand")))], []
        end = ("", [dict(id="end", name="yield_control", arguments=json.dumps(dict(reason="Need another grasp")))], [])
        model = Model([reply("stale"), reply(self.obs.id), end])
        result = dict(gripper_result=dict(state="stable_residual", contact="unknown", grasp_success=None))
        def grip(*args, **kwargs):
            calls.append((args, kwargs))
            return result
        context = SimpleNamespace(task=self.task, episode=self.episode(), robot=SimpleNamespace(hold=lambda:None, gripper=grip),
                                  checkpoint=lambda **kw:self.obs, action=lambda method, args, fn:fn())
        with self.assertRaises(CapabilityUnavailable):
            L2Controller(model).run(context)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][1]["observation"], self.obs)
        self.assertIn("no_execution", json.dumps(model.histories[1]))
        self.assertIn("stable_residual", json.dumps(model.histories[2]))
        act = next(t for t in direct_tools() if t["name"] == "act")
        self.assertEqual(act["parameters"]["properties"]["targets"]["properties"]["right"]["properties"]["hand"], {"type":"null"})

    def test_l2_unknown_gripper_send_is_not_retried(self):
        calls = []
        model = Model([("", [dict(id="g", name="gripper", arguments=json.dumps(dict(
            observation_id=self.obs.id, arm="right", action="close", mode="wrap", reason="Visible rod")))], [])])
        def grip(*args, **kwargs):
            calls.append(args)
            raise ExecutionUncertain("Native write outcome unknown")
        context = SimpleNamespace(task=self.task, episode=self.episode(), robot=SimpleNamespace(hold=lambda:None, gripper=grip),
                                  checkpoint=lambda **kw:self.obs, action=lambda method, args, fn:fn())
        with self.assertRaises(ExecutionUncertain):
            L2Controller(model).run(context)
        self.assertEqual(len(calls), 1)

    def test_generated_policy_cannot_import_sdk_or_use_legacy_microstep_api(self):
        for source in ("import os\ndef policy(robot,memory): pass",
                       "def policy(robot,memory): robot.move_cm(1)"):
            with self.assertRaises(ValueError):
                validate_source(source)
        validate_source("def policy(robot,memory):\n    s=robot.scene(['hole'])\n    return s")

    def test_l3_imports_full_measured_trace_and_three_views(self):
        import h5py
        import numpy as np
        path = self.root / "episode_0.hdf5"
        with h5py.File(path, "w") as data:
            for k, v in dict(hand_stack="manus_revo2", qpos_layout="interleaved_LhRh",
                             action_source="armcmd_handcmd", qpos_arm_source="measured",
                             qpos_hand_source="measured", arm_cmd_matched_frames=3).items():
                data.attrs[k] = v
            data["observations/qpos"] = np.zeros((3, 26))
            data["action"] = np.full((3, 26), .1)
            data["timestamp_ros_nsec"] = [1000000000, 1100000000, 1200000000]
            for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                data["observations/images/" + name] = np.full((3, 4, 4, 3), 125, dtype=np.uint8)
        demo = import_hdf5(path, self.root / "demo", fk=lambda q: self.obs.eef, keyframe_stride=1)
        self.assertEqual(len(demo.observations), 3)
        self.assertEqual(demo.trace["commanded_action26"][0][0], .1)
        self.assertEqual(demo.trace["measured_qpos26"][0][0], 0)
        self.assertEqual(demo.manifest["frames"], 3)
        self.assertFalse(demo.manifest["replay_allowed"])
        with self.assertRaises(ValueError):
            import_hdf5(path, self.root / "stale-demo", fk=lambda q: self.obs.eef, not_before_ns=2000000000)

    def test_l2_persists_context_and_recovers_only_pre_dispatch_input_errors(self):
        # The first action is stale; only the corrected action reaches the port.
        targets = {a: dict(p, hand=None) for a, p in self.obs.eef.items()}
        def reply(oid):
            call = dict(id=oid, name="act", arguments=json.dumps(dict(
                observation_id=oid, targets=targets, steps=1, reason="Align from visible views")))
            return "", [call], [dict(role="assistant", content="direct tool decision")]
        model = Model([reply("stale"), reply("obs-2")])
        c = self.coordinator(ScriptedMonitor([Outcome.ONGOING, Outcome.ONGOING, Outcome.SUCCESS]),
                             {Level.L1: Unavailable(), Level.L2: L2Controller(model), Level.L3: Unavailable()})
        result = c.run()
        self.assertEqual(result["solved_by"], "L2")
        self.assertEqual(len(c.robot.actions), 1)
        self.assertIn("no_execution", json.dumps(model.histories[1]))
        self.assertNotIn("PRIVATE MONITOR", json.dumps(model.histories))

    def test_real_physicalrsi_runtime_and_paired_evolution(self):
        runtime = Path(__import__("os").environ.get("PHYSICALRSI_TEST_RUNTIME", "/tmp/physicalrsi-real-robot-runtime"))
        if not (runtime / "manifest.json").is_file():
            self.skipTest("Prepare a local pure-Python isolation runtime for this integration test")
        from PhysicalRSI_baselines.real_robot.evolution import DevelopmentProposer, PairedEpisodeEvaluator, make_harness
        source = ("def policy(robot, memory):\n"
                  "    scene = robot.scene(['target'])\n"
                  "    target = scene['objects']['target']['pose']\n"
                  "    robot.approach('right', target[:3], target[3:])\n")
        model = Model([(json.dumps(dict(source=source, memory=dict(lesson="derive target from each scene"),
                                              rationale="Use current object geometry instead of replay")), [], [])])
        library = PolicyLibrary(self.root / "library", foundation=model.config.public())
        parent = library.freeze(task=self.task, source="def policy(robot, memory):\n    return None\n", memory={})
        feedback = [dict(task=self.task.id, split="development", outcome="success", solved_by="L3",
                         observations=[self.obs.public()], execution=[dict(kind="human_demonstration", level="L3")])]
        proposer = DevelopmentProposer(library=library, task=self.task, model=model, feedback=feedback)

        base_obs = self.obs
        class SceneRobot(FakeRobot):
            def __init__(self, x):
                super().__init__(base_obs)
                self.x, self.reached = x, False

            def scene(self, names):
                return dict(objects={"target": dict(pose=[self.x, 0, .1, 1, 0, 0, 0])})

            def approach(self, arm, position_m, quaternion_wxyz, *, checkpoint):
                self.reached = position_m == [self.x, 0, .1]
                return dict(executed=True)

        class IndependentMonitor:
            def __init__(self, robot):
                self.robot = robot

            def submit(self, obs):
                f = Future()
                f.set_result(Assessment(obs.id, Outcome.SUCCESS if self.robot.reached else Outcome.ONGOING,
                                        Progress.UNKNOWN, "FAKE DEVICE TEST ONLY"))
                return f

            def close(self):
                pass

        def trial(task, case, candidate, directory):
            robot = SceneRobot(case["x"])
            episode = Episode(directory, task, split="validation", policy_id=candidate["id"])
            controller = L1Controller(library.load(candidate), PolicyRuntime(runtime, timeout_s=10))
            return Coordinator(task=task, robot=robot, monitor=IndependentMonitor(robot), judge=None,
                               episode=episode, controllers={Level.L1: controller, Level.L2: Unavailable(),
                                                            Level.L3: Unavailable()},
                               allowed_levels=(Level.L1,)).run()

        evaluator = PairedEpisodeEvaluator(task=self.task, scene_provider=lambda task, n: [{"x": .1}, {"x": .25}],
                                            trial_runner=trial, identity=dict(scope="fake unit-test devices"), episodes=2)
        loop = make_harness(self.root / "round-1", state_root=self.root / "lineage", parent=parent,
                            proposer=proposer, evaluator=evaluator)
        result = loop.run()
        self.assertNotEqual(result["harness"]["id"], parent["id"])
        self.assertEqual(result["decision"]["decision"], "inherit_child")
        selected = result["decision"]["survivor_id"]
        self.assertEqual(result["decision"]["metrics"][selected]["normalized_score"], 1)
        self.assertEqual(result["decision"]["metrics"][parent["id"]]["normalized_score"], 0)
        self.assertNotIn("validation-scenes", json.dumps(model.histories))
        from PhysicalRSI_baselines.real_robot.learning import selected_policy
        incumbent = selected_policy(self.root / "lineage", task=self.task, qualification="offline")
        self.assertEqual(incumbent["id"], selected)
        with self.assertRaisesRegex(ValueError, "qualification"):
            selected_policy(self.root / "lineage", task=self.task, qualification="physical")
        with self.assertRaisesRegex(ValueError, "task/criteria"):
            selected_policy(self.root / "lineage", task=replace(self.task, id="other"), qualification="offline")


if __name__ == "__main__":
    unittest.main()
