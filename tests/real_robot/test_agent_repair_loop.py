"""Repair regressions: real data boundaries and loop transitions, fake robots only."""
import base64
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PhysicalRSI_demos.real_robot.analysis_tools import AnalysisTools, calculate
from PhysicalRSI_demos.real_robot.contracts import Assessment, Frame, Level, Observation, Outcome, Progress, Task, CapabilityUnavailable
from PhysicalRSI_demos.real_robot.coordinator import Coordinator
from PhysicalRSI_demos.real_robot.evidence import Episode
from PhysicalRSI_demos.real_robot.learning import read_development_episode
from PhysicalRSI_demos.real_robot.learning_projection import proposal_feedback
from PhysicalRSI_demos.real_robot.models import MailboxConfig, MailboxModel, RecordedModel, require_independent_models
from PhysicalRSI_demos.real_robot.monitor import MonitoringSystem
from PhysicalRSI_demos.real_robot.policy_runtime import validate_source
from PhysicalRSI_demos.real_robot.references import RSI
from PhysicalRSI_demos.real_robot.library import dependency_files
from PhysicalRSI_demos.real_robot.reporting import report_episodes
from PhysicalRSI_demos.real_robot.video_reference import VideoReferenceL3
from PhysicalRSI_demos.real_robot.guidance import submit, consume
from test_agent_system import Model, FakeRobot, ScriptedMonitor, Unavailable


class Repairs(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        stamp = time.time_ns()
        frames = []
        for camera in ("front", "left", "right"):
            path = self.root / (camera + ".png")
            Image.new("RGB", (32, 24), (10, 100, 200)).save(path)
            frames.append(Frame(camera, str(path), stamp, hashlib.sha256(path.read_bytes()).hexdigest()))
        pose = dict(position=[0., 0., 0.], quaternion_wxyz=[1., 0., 0., 0.])
        self.obs = Observation("fresh", stamp, tuple(frames), (0.,)*26, dict(left=pose, right=pose))
        self.task = Task("pour", "Pour bottle into cup", "Visible water and stable upright bottle")
        self.episode = Episode(self.root / "episode", self.task)

    def test_both_l2_and_l3_return_to_l1_without_claiming_unassisted_success(self):
        for rescue in (Level.L2, Level.L3):
            with self.subTest(rescue=rescue):
                episode = Episode(self.root / rescue.value, self.task)
                robot = FakeRobot(self.obs)
                order, learned = [], []
                class L1:
                    def run(self, ctx):
                        order.append("L1")
                        if learned:
                            ctx.action("test_grasp", {}, lambda: robot.actions.append("grasp") or {})
                        else:
                            raise CapabilityUnavailable("Need experience")
                class Rescue:
                    def run(self, ctx):
                        order.append(rescue.value)
                        return dict(learning_checkpoint=True)
                class Monitor:
                    def submit(self, obs):
                        f = Future()
                        f.set_result(Assessment(obs.id, Outcome.SUCCESS if robot.actions else Outcome.ONGOING,
                                                Progress.UNKNOWN, "private"))
                        return f
                    def close(self):
                        pass
                def learn(ctx, level):
                    if level == rescue:
                        learned.append(level)
                        return True
                    return False
                controllers = {Level.L1:L1(), Level.L2:Unavailable(), Level.L3:Unavailable()}
                controllers[rescue] = Rescue()
                result = Coordinator(task=self.task, robot=robot, monitor=Monitor(), judge=None,
                    episode=episode, controllers=controllers, rescue_learning=learn).run()
                self.assertEqual(result["outcome"], "success", result)
                self.assertEqual(order, ["L1", rescue.value, "L1"])
                self.assertEqual(learned, [rescue])
                self.assertFalse(result["l1_success"])
                self.assertEqual(result["usage"]["attempts"], 2)
                self.assertEqual(report_episodes([episode.root])["records"][0]["receipt"], "verified")

    def test_l3_reference_never_becomes_current_monitor_evidence(self):
        context = SimpleNamespace(robot=FakeRobot(self.obs), episode=self.episode, task=self.task,
                                  checkpoint=lambda **kwargs: self.obs)
        lesson = dict(sufficient=False, relative_rule="", applicability="", uncertainty="No visible evidence", evidence_frame_ids=[])
        model = Model([("", [dict(id="finish", name="finish_reference", arguments=json.dumps(lesson))], [])])
        with patch("PhysicalRSI_demos.real_robot.video_reference.video_inventory", return_value=[{"front":"episode_0.mp4"}]):
            self.assertTrue(VideoReferenceL3(self.root, model).run(context)["learning_checkpoint"])
        self.assertEqual(context.robot.releases, 0)
        self.assertFalse(self.episode.observations)
        self.assertEqual(self.episode.events[-1]["kind"], "video_reference")
        self.assertFalse(self.episode.events[-1]["reference"]["lesson"]["sufficient"])

    def test_video_learning_follows_bottleneck_and_model_selected_windows(self):
        self.episode.event("controller_unavailable", level="L2", reason="Bottle too close to fingertips")
        def reply(name, args, i):
            return "", [dict(id=str(i), name=name, arguments=json.dumps(args))], []
        model = Model([reply("inspect_video", dict(clip=0,start_s=2.,end_s=6.,samples=3,question="Find enclosure phase"), 1),
            reply("inspect_video", dict(clip=0,start_s=3.1,end_s=3.4,samples=7,question="Check fingertip gap before closure"), 2),
            reply("crop_video_frame", dict(frame_id="f0001",box=[2,2,20,18],scale=3,question="Palm clearance detail"), 3),
            reply("finish_reference", dict(sufficient=True,relative_rule="Center object in the open grasp gap",
                applicability="Before closure, check both opposed finger surfaces",uncertainty="Need current wrist RGB",
                evidence_frame_ids=["f0000","f0002"]), 4)])
        context = SimpleNamespace(robot=FakeRobot(self.obs), episode=self.episode, task=self.task,
                                  checkpoint=lambda **kwargs: self.obs)
        context.task = self.task
        frame = self.obs.frames[0].public()
        windows = []
        def extract(videos, output, **kwargs):
            windows.append((kwargs["start_s"], kwargs["end_s"], kwargs["samples"]))
            return dict(frames=[dict(frame, video_time_s=kwargs["start_s"], source_frame_index=1)])
        with patch("PhysicalRSI_demos.real_robot.video_reference.video_inventory", return_value=[{"front":"episode_0.mp4"}]), \
             patch("PhysicalRSI_demos.real_robot.video_reference.probe_reference", return_value=({}, {c:[0.,10.] for c in ("front","left","right")})), \
             patch("PhysicalRSI_demos.real_robot.video_reference.extract_reference", side_effect=extract):
            result = VideoReferenceL3(self.root, model).run(context)
        self.assertEqual(windows, [(2.,6.,3), (3.1,3.4,7)])
        self.assertTrue(result["sufficient_evidence"])
        reference = self.episode.events[-1]["reference"]
        self.assertEqual(reference["frames"][-1]["captured_ns"], frame["captured_ns"])
        self.assertIn("Bottle too close", json.dumps(model.histories[0]))
        self.assertFalse(reference["current_task_success"])

    def test_learning_projection_strips_controls_including_nested_receipts(self):
        raw = dict(task=self.task.id, split="development", observations=[self.obs.public()],
            execution=[dict(kind="human_demonstration", level="L3", trajectory=dict(measured_qpos26=[[123.]*26])),
                       dict(kind="action_started", level="L2", method="act_eef",
                            arguments=dict(targets={"right":{"position":[999.,0.,0.]}}, reason="Center the bottle in the gap")),
                       dict(kind="action_completed", level="L2", receipt=dict(status="completed", sent_joints=[888.]*14))])
        projected = proposal_feedback([raw])[0]
        text = json.dumps(projected)
        for forbidden in ("qpos26", "sent_joints", "quaternion_wxyz", "trajectory", "targets"):
            self.assertNotIn(forbidden, text)
        def numbers(value):
            if isinstance(value, dict):
                return [n for v in value.values() for n in numbers(v)]
            if isinstance(value, list):
                return [n for v in value for n in numbers(v)]
            return [value] if type(value) in (int, float) else []
        self.assertFalse({123., 999., 888.} & set(numbers(projected)))
        self.assertIn("Center the bottle", text)
        self.assertEqual(len(projected["observations"][0]["frames"]), 3)
        self.assertIn("qpos26", raw["observations"][0])

    def test_memory_waypoints_rejected_but_current_scene_targets_allowed(self):
        with self.assertRaisesRegex(ValueError, "stored memory"):
            validate_source('def policy(robot,memory):\n    for p in memory["waypoints"]:\n        robot.approach("right",p["position"],p["quaternion"])')
        validate_source('def policy(robot,memory):\n    s=robot.scene(["bottle"])\n    robot.approach("right",s["position"],s["quaternion"])')

    def test_analysis_crop_preserves_source_and_forbids_old_ids_or_file_calculations(self):
        tool = AnalysisTools()
        manifest, frames = tool.invoke("inspect_image", dict(observation_id=self.obs.id, camera="right",
            box=[4, 3, 20, 18], scale=3), self.obs, self.episode)
        self.assertEqual(frames[0].captured_ns, self.obs.captured_ns)
        self.assertEqual(manifest["source"]["sha256"], self.obs.frames[2].sha256)
        self.assertFalse(manifest["fresh_observation"])
        with self.assertRaises(ValueError):
            tool.invoke("inspect_image", dict(observation_id="old", camera="right", box=None, scale=1), self.obs, self.episode)
        self.assertAlmostEqual(calculate("sqrt(3**2 + 4**2)"), 5.)
        for code in ("open('/private')", "__import__('os')", "[1]*100000000", "2**100000000"):
            with self.assertRaises((ValueError, TypeError)):
                calculate(code)

    def test_finished_learning_counts_do_not_invalidate_execution_hash(self):
        self.episode.observation(self.obs)
        self.episode.event("controller_enter", level="L2")
        model = RecordedModel(Model([("{}", [], []), ("{}", [], [])]), self.episode, "development")
        model.complete([], [])
        result = self.episode.finish(outcome=Outcome.UNCERTAIN, solved_by=None)
        model.complete([], [])
        self.assertEqual(read_development_episode(self.episode.root)[2]["events_sha256"], result["events_sha256"])
        self.assertEqual(len(list((self.episode.root / "post-episode-model-requests").glob("*.json"))), 1)

    def test_mailbox_not_accepted_as_independent_physical_model(self):
        model = MailboxModel(self.root / "mail", MailboxConfig("current-agent", "test"), role="monitor")
        with self.assertRaisesRegex(ValueError, "independent HTTP"):
            require_independent_models(dict(monitor=model))

    def test_drop_requires_all_views_and_terminates_without_rescue(self):
        payload = dict(observation_id=self.obs.id, outcome="failure", progress="regressed", evidence="Visible drop",
                       termination=dict(event="target_dropped_and_unlocatable", observed_drop=True,
                                        visibility={s:"absent" for s in ("front", "left", "right")}))
        model = Model([(json.dumps(payload), [], [])])
        monitor = MonitoringSystem(model, self.task)
        self.addCleanup(monitor.close)
        assessment = monitor._assess(self.obs, None)
        self.assertTrue(assessment.target_lost)
        payload["termination"]["visibility"]["right"] = "occluded"
        with self.assertRaises(ValueError):
            bad = MonitoringSystem(Model([(json.dumps(payload), [], [])]), self.task)
            try:
                bad._assess(self.obs, None)
            finally:
                bad.close()
        class Lost:
            def submit(self, obs):
                f=Future(); f.set_result(Assessment(obs.id, Outcome.FAILURE, Progress.REGRESSED, "drop", target_lost=True)); return f
            def close(self): pass
        never = SimpleNamespace(run=lambda _: self.fail("No controller after confirmed drop"))
        result = Coordinator(task=self.task, robot=FakeRobot(self.obs), monitor=Lost(), judge=None,
            episode=self.episode, controllers={l:never for l in Level}).run()
        self.assertEqual(result["status"], "target_dropped_and_unlocatable")
        self.assertNotEqual(result["outcome"], "success")

    def test_language_guidance_counted_once_and_distinct_from_judgment(self):
        self.episode.observation(self.obs)
        self.episode.event("controller_enter", level="L1")
        submit(self.episode.root, "把物体放在四指和拇指空隙中心")
        consume(self.episode); consume(self.episode)
        result = self.episode.finish(outcome=Outcome.SUCCESS, solved_by=Level.L1)
        self.assertEqual(result["usage"]["l3_interventions"]["language"], 1)
        self.assertEqual(result["human_judgments"], 0)
        self.assertFalse(result["l1_success"])
        self.assertEqual(report_episodes([self.episode.root])["records"][0]["receipt"], "verified")
        with self.assertRaises(ValueError):
            submit(self.episode.root, "late hint")

    def test_reference_dependencies_cover_imported_proposal_and_isolation(self):
        pins = {p.resolve() for p in dependency_files().values()}
        for relative in ("PhysicalRSI_baselines/robodojo/skill_proposal.py", "PhysicalRSI_core/infra/isolation_bridge.py",
                         "PhysicalRSI_core/infra/language.py", "PhysicalRSI_core/self_harness/artifacts.py"):
            self.assertIn((RSI / relative).resolve(), pins)

    def test_video_extraction_reads_rgb_with_provenance_and_never_opens_hdf5(self):
        import shutil
        import subprocess
        from PhysicalRSI_demos.real_robot.video_reference import video_inventory, extract_reference
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            self.skipTest("ffmpeg/ffprobe unavailable")
        source = self.root / "demonstrations"
        first = source / "video/cam_high/episode_0.mp4"
        first.parent.mkdir(parents=True)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                        "color=c=red:s=32x24:r=4", "-t", "1", "-c:v", "mpeg4", str(first)], check=True)
        for camera in ("cam_left_wrist", "cam_right_wrist"):
            path = source / "video" / camera / first.name
            path.parent.mkdir(parents=True)
            shutil.copyfile(first, path)
        (source / "episode_0.hdf5").write_text("INVALID HDF5: must never be read")
        with patch("h5py.File", side_effect=AssertionError("Do not open demonstration controls")):
            reference = extract_reference(video_inventory(source)[0], self.episode.public / "video-references/0", samples=2)
        self.assertEqual(len(reference["frames"]), 6)
        self.assertFalse(reference["has_motor_or_eef_trajectory"])
        self.assertFalse(reference["current_task_success"])
        self.assertEqual(reference["frames"][0]["captured_ns"], 1)
        self.assertEqual({f["source_frame_index"] for f in reference["frames"]}, {0, 3})
        self.assertEqual({f["video_time_s"] for f in reference["frames"]}, {0., .75})
        self.assertIn("video-relative", reference["time_semantics"])

    def test_online_proposal_failure_preserves_parent_and_allows_fallback(self):
        from PhysicalRSI_demos.real_robot.experiment import DevelopingL1
        parent = {"id": "retained-parent"}
        def fail(**kwargs):
            raise ValueError("malformed model reply")
        l1 = DevelopingL1(library=SimpleNamespace(propose=fail), model=None, runtime=None, parent=parent)
        context = SimpleNamespace(episode=self.episode, task=self.task, robot=FakeRobot(self.obs))
        self.assertFalse(l1.improve(context, Level.L2))
        self.assertIs(l1.parent, parent)
        self.assertEqual(self.episode.events[-1]["kind"], "online_learning_pending")

    def test_l2_analysis_and_return_tool_keep_call_pairs_across_reentry(self):
        from PhysicalRSI_demos.real_robot.controllers import L2Controller
        def reply(name, args, index):
            call = dict(id=str(index), name=name, arguments=json.dumps(args))
            native = dict(role="assistant", content="", tool_calls=[dict(id=str(index), type="function",
                function=dict(name=name, arguments=call["arguments"]))])
            return "", [call], [native]
        model = Model([reply("calculate", {"expression": "sqrt(25)"}, 1),
                       reply("return_to_l1", {"reason": "Use visible enclosure before closure"}, 2),
                       reply("yield_control", {"reason": "Need a demonstration"}, 3)])
        controller = L2Controller(model)
        context = SimpleNamespace(episode=self.episode, task=self.task, robot=FakeRobot(self.obs),
                                  checkpoint=lambda **kwargs: self.obs)
        self.assertTrue(controller.run(context)["learning_checkpoint"])
        with self.assertRaises(CapabilityUnavailable):
            controller.run(context)
        ids = [item["tool_call_id"] for item in controller.history if item["role"] == "tool"]
        self.assertEqual(ids, ["1", "2", "3"])

    def test_repeated_unsolved_l2_revisions_eventually_consult_l3(self):
        order, candidates = [], []
        class Controller:
            def __init__(self, level): self.level = level
            def run(self, ctx):
                order.append(self.level)
                if self.level == "L3":
                    raise KeyboardInterrupt()  # Finish fixture without any real action.
                raise CapabilityUnavailable("Cannot complete")
        def learn(ctx, level):
            candidates.append(level)
            return True
        result = Coordinator(task=self.task, robot=FakeRobot(self.obs), monitor=ScriptedMonitor([]), judge=None,
            episode=self.episode, controllers={l:Controller(l.value) for l in Level}, rescue_learning=learn).run()
        self.assertEqual(order, ["L1", "L2", "L1", "L2", "L3"])
        self.assertEqual(candidates, [Level.L2, Level.L2])
        self.assertEqual(result["status"], "interrupted")

    def test_l2_reentry_receives_public_l1_and_l3_handoff_without_private_or_replay_data(self):
        from PhysicalRSI_demos.real_robot.controllers import L2Controller
        def reply(index):
            args = json.dumps(dict(reason="More current geometry needed"))
            return "", [dict(id=str(index), name="yield_control", arguments=args)], []
        model = Model([reply(1), reply(2), reply(3)])
        controller = L2Controller(model)
        context = SimpleNamespace(episode=self.episode, task=self.task, robot=FakeRobot(self.obs),
                                  checkpoint=lambda **kwargs: self.obs)
        self.episode.event("controller_unavailable", level="L1", reason="OLD bottleneck")
        self.episode.event("controller_unavailable", level="L1", reason="CURRENT wrist depth unresolved")
        (self.episode.private / "secret.json").write_text('"PRIVATE EVALUATOR SECRET"')
        with self.assertRaises(CapabilityUnavailable):
            controller.run(context)
        first = json.dumps(model.histories[0])
        self.assertIn("CURRENT wrist depth unresolved", first)
        self.assertNotIn("OLD bottleneck", first)
        self.episode.event("video_reference", level="L3", reference=dict(
            lesson=dict(relative_rule="Palm faces bottle side before enclosure",
                        applicability="Use fresh wrist RGB", uncertainty="Cap removal not shown",
                        sufficient=False, evidence_frame_ids=["f0001"], hidden="PRIVATE EVALUATOR SECRET"),
            frames=[dict(path="OLD VIDEO PIXELS")], trajectory="REPLAY TRAJECTORY"))
        self.episode.event("controller_unavailable", level="L1", reason="NEW palm view required")
        for _ in range(2):
            with self.assertRaises(CapabilityUnavailable):
                controller.run(context)
        final = json.dumps(model.histories[-1])
        for expected in ("Palm faces bottle side before enclosure", "Use fresh wrist RGB",
                         "Cap removal not shown", "NEW palm view required"):
            self.assertIn(expected, final)
        for forbidden in ("PRIVATE EVALUATOR SECRET", "OLD VIDEO PIXELS", "REPLAY TRAJECTORY"):
            self.assertNotIn(forbidden, final)
        self.assertEqual(final.count("Palm faces bottle side before enclosure"), 1)
        handoffs = [e for e in self.episode.events if e["kind"] == "l2_handoff"]
        self.assertEqual(len(handoffs), 2)
        lesson = handoffs[-1]["public_context"]["historical_video_lessons"][0]["lesson"]
        self.assertIs(lesson["sufficient"], False)
        self.assertFalse(context.robot.actions)

    def test_l2_language_guidance_is_not_repeated_on_reentry(self):
        from PhysicalRSI_demos.real_robot.controllers import L2Controller
        reply = ("", [dict(id="yield", name="yield_control", arguments=json.dumps(dict(reason="Need help")))], [])
        model = Model([reply, reply])
        controller = L2Controller(model)
        self.episode.event("human_guidance", level="L3", guidance="UNIQUE HUMAN SIDE GRASP HINT")
        context = SimpleNamespace(episode=self.episode, task=self.task, robot=FakeRobot(self.obs),
                                  checkpoint=lambda **kwargs: self.obs)
        for _ in range(2):
            with self.assertRaises(CapabilityUnavailable):
                controller.run(context)
        self.assertEqual(json.dumps(model.histories[-1]).count("UNIQUE HUMAN SIDE GRASP HINT"), 1)

    def test_l2_inherits_only_policy_lesson_text_and_known_robot_frames(self):
        from PhysicalRSI_demos.real_robot.controllers import L2Controller
        reply = ("", [dict(id="yield", name="yield_control", arguments=json.dumps(dict(reason="Need view")))], [])
        model = Model([reply, reply])
        inherited = dict(policy_id="frozen-parent", memory=dict(
            lessons=["COORDINATE POSITION AND ORIENTATION", {"trajectory":"RAW JOINT ARRAY"}],
            evaluation="PRIVATE JUDGE", frames="OLD PIXELS", trajectory="RAW JOINT ARRAY"))
        controller = L2Controller(model, policy_context=lambda: inherited,
                                  control_geometry=dict(pose_frame="KNOWN ROBOT TOOL FRAME", scene_information=False))
        context = SimpleNamespace(episode=self.episode, task=self.task, robot=FakeRobot(self.obs),
                                  checkpoint=lambda **kwargs: self.obs)
        for _ in range(2):
            with self.assertRaises(CapabilityUnavailable):
                controller.run(context)
        text = json.dumps(model.histories[-1])
        self.assertEqual(text.count("COORDINATE POSITION AND ORIENTATION"), 1)
        self.assertIn("KNOWN ROBOT TOOL FRAME", text)
        for value in ("PRIVATE JUDGE", "OLD PIXELS", "RAW JOINT ARRAY"):
            self.assertNotIn(value, text)
        events = [e for e in self.episode.events if e["kind"] == "l2_policy_memory"]
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]["public_context"]["current_geometry"])
        self.assertFalse(any(e['kind']=='human_guidance' for e in self.episode.events))
        self.assertFalse(context.robot.actions)

    def test_online_and_final_learning_retain_actual_runtime_capabilities(self):
        from PhysicalRSI_demos.real_robot.experiment import DevelopingL1
        captured = []
        library = SimpleNamespace(propose=lambda **kwargs: captured.append(kwargs['feedback'][0]))
        learner = DevelopingL1(library=library, model=None, runtime=None,
                               parent=dict(id='parent'), perception=object())
        robot = FakeRobot(self.obs)
        robot.capabilities = dict(scene=False, approach=False, direct_eef=True, gripper=True)
        context = SimpleNamespace(episode=self.episode, robot=robot, task=self.task)
        learner.improve(context, Level.L2)
        learner.learn(self.episode, dict(outcome='uncertain', solved_by=None, l2_calls=1, l3_calls=0))
        for feedback in captured:
            caps = feedback['capabilities']
            self.assertFalse(caps['scene'])
            self.assertFalse(caps['approach'])
            for method in ('look', 'align', 'relate'):
                self.assertTrue(caps[method])
        self.assertFalse(robot.actions)

    def test_saved_capability_evidence_reaches_recovered_proposal(self):
        row = dict(task='pour', split='development', observations=[], execution=[
            dict(kind='execution_capabilities', level='L1', capabilities=dict(scene=False,
                 approach=False, look=True, align=True, relate=True))])
        value = proposal_feedback([row])[0]
        self.assertEqual(value['capabilities'], row['execution'][0]['capabilities'])


if __name__ == "__main__":
    unittest.main()
