"""Rendered-image alignment tests. No ROS, SDK, network or robot connection."""
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PhysicalRSI_baselines.real_robot.contracts import ActionRejected, ExecutionUncertain, Frame, Observation, Task
from PhysicalRSI_baselines.real_robot.evidence import Episode
from PhysicalRSI_baselines.real_robot.physical_port import PhysicalRobot
from PhysicalRSI_baselines.real_robot.policy_runtime import PolicyRuntime
from PhysicalRSI_baselines.real_robot.controllers import L1Controller
from PhysicalRSI_baselines.real_robot.visual_perception import project_features
from PhysicalRSI_baselines.real_robot.visual_servo import ServoSettings, VisualServo, fit_response


class PixelDetector:
    """Infer landmarks solely from PNG pixels, never from plant object state."""
    def look(self, observation, features, *, deadline):
        views = {}
        for frame in observation.frames:
            pixels = np.asarray(Image.open(frame.path).convert("RGB"))
            h, w, _ = pixels.shape
            rows = {}
            for name, channel in zip(features, (1, 0)):
                mask = pixels[:, :, channel] > 200
                yy, xx = np.nonzero(mask)
                rows[name] = dict(visibility="visible" if len(xx) else "occluded",
                                  uv=[float(xx.mean())/(w-1), float(yy.mean())/(h-1)] if len(xx) else None,
                                  confidence=.95)
            views[frame.camera] = rows
        return project_features(dict(observation_id=observation.id, views=views), observation, features)


class ImagePlant:
    def __init__(self, root, target, *, flip=False, gain=1, hide=False, rank_deficient=False, fail=False):
        self.root, self.target = root, np.asarray(target)
        self.position = np.zeros(3)
        self.gain, self.hide, self.fail = gain, hide, fail
        self.number, self.actions = 0, []
        self.other = [0, .4, .2]
        self.projections = [np.array([[1.5, 0, 0], [0, 0, 1.5]]),
                            np.array([[0, 1.5, 0], [0, 0, 1.5]]),
                            np.array([[1, .5, 0], [0, 0, 1.5]])]
        if flip:
            self.projections = [p @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]]) for p in self.projections]
        if rank_deficient:
            self.projections = [self.projections[0]] * 3

    def hold(self):
        pass

    def observe(self):
        self.number += 1
        captured = time.time_ns()
        frames = []
        for camera, projection in zip(("front", "left", "right"), self.projections):
            # Distinct rings remain segmentable when the two projections overlap.
            image = Image.new("RGB", (401, 401))
            draw = ImageDraw.Draw(image)
            for xyz, color, radius in [(self.target, "red", 5), (self.position, "lime", 2)]:
                if self.hide and color == "red" and camera == "left":
                    continue
                u, v = (projection @ xyz + .5)*400
                draw.ellipse((u-radius, v-radius, u+radius, v+radius), fill=color)
            path = self.root / f"{self.number}-{camera}.png"
            image.save(path)
            frames.append(Frame(camera, str(path), captured, hashlib.sha256(path.read_bytes()).hexdigest()))
        pose = lambda p: dict(position=[float(v) for v in p], quaternion_wxyz=[1, 0, 0, 0])
        return Observation(f"obs-{self.number}", captured, tuple(frames), (0.,)*26,
                           dict(left=pose(self.other), right=pose(self.position)))

    def act_eef(self, targets, steps, *, checkpoint, observation):
        self.actions.append(targets)
        if self.fail:
            raise ExecutionUncertain("Test unknown-send fault")
        if targets["left"]["position"] != self.other or any(t["hand"] is not None for t in targets.values()):
            raise AssertionError("Visual alignment changed the other arm or a hand")
        if any(t["quaternion_wxyz"] != [1, 0, 0, 0] for t in targets.values()):
            raise AssertionError("Translation-only alignment changed orientation")
        self.position += self.gain*(np.asarray(targets["right"]["position"])-self.position)
        return dict(status="completed", ticks=steps*2)


class ServoTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def context(self, **kw):
        robot = ImagePlant(self.root, kw.pop("target", [.09, -.07, .06]), **kw)
        episode = Episode(self.root / "episode", Task("align", "align visible landmarks", "independent alignment test"))
        context = SimpleNamespace(robot=robot, episode=episode, last_observation=None)
        def checkpoint(**kw):
            context.last_observation = robot.observe()
            episode.observation(context.last_observation)
            return context.last_observation
        context.checkpoint = checkpoint
        context.action = lambda name, args, invoke: invoke()
        return context

    def run_servo(self, context, perception=None):
        return VisualServo(perception or PixelDetector(), settings=ServoSettings(image_tolerance=.01)).align(
            context, "right", "hand marker", "object marker", ["front", "left"], deadline=time.monotonic()+20)

    def test_translates_using_image_response_and_measured_partial_motion(self):
        context = self.context(gain=.5)
        result = self.run_servo(context)
        self.assertEqual(result["status"], "image_aligned")
        self.assertFalse(result["independent_success"])
        self.assertLess(np.linalg.norm(context.robot.position-context.robot.target), .02)
        self.assertGreater(len(context.robot.actions), 3)
        transitions = [e["transition"] for e in context.episode.events if e["kind"] == "visual_servo_transition"]
        first = transitions[0]
        np.testing.assert_allclose(first["measured_translation"], np.array(first["commanded_translation"])*.5)

    def test_same_controller_transfers_to_reversed_camera_axes_and_new_layout(self):
        context = self.context(flip=True, target=[-.08, .06, -.07])
        result = self.run_servo(context)
        self.assertEqual(result["status"], "image_aligned")
        self.assertLess(np.linalg.norm(context.robot.position-context.robot.target), .02)

    def test_occlusion_returns_unknown_without_motion(self):
        context = self.context(hide=True)
        self.assertEqual(self.run_servo(context)["status"], "unobservable")
        self.assertEqual(context.robot.actions, [])

    def test_rank_deficient_views_do_not_guess_third_axis(self):
        context = self.context(rank_deficient=True)
        self.assertEqual(self.run_servo(context)["status"], "insufficient_motion_evidence")
        self.assertEqual(len(context.robot.actions), 3)

    def test_unknown_action_effect_escapes_without_retry(self):
        context = self.context(fail=True)
        with self.assertRaises(ExecutionUncertain):
            self.run_servo(context)
        self.assertEqual(len(context.robot.actions), 1)

    def test_stale_feature_result_cannot_be_rebound_to_new_image(self):
        context = self.context()
        capture = context.checkpoint
        def old(**kw):
            obs = capture()
            obs = replace(obs, captured_ns=obs.captured_ns-31*10**9,
                          frames=tuple(replace(f, captured_ns=f.captured_ns-31*10**9) for f in obs.frames))
            return obs
        context.checkpoint = old
        self.assertEqual(self.run_servo(context)["status"], "stale_observation")
        self.assertEqual(context.robot.actions, [])

    def test_port_rejects_refresh_of_features_before_dispatch(self):
        context = self.context()
        first, second = context.checkpoint(), context.checkpoint()
        port = PhysicalRobot.__new__(PhysicalRobot)
        port.last_observation = second
        port.rpc = lambda *a, **kw: self.fail("Old features reached hardware submission")
        with self.assertRaises(ActionRejected):
            port.act_eef({}, 5, checkpoint=lambda: None, observation=first)

    def test_zero_or_unrelated_measured_response_is_unusable(self):
        transitions = [dict(measured_translation=[0, 0, 0], image_error_change=[1, 1, 1, 1])]*3
        self.assertIsNone(fit_response(transitions, 4))

    def test_original_isolation_driver_can_call_high_level_alignment(self):
        runtime = Path(__import__("os").environ.get("PHYSICALRSI_TEST_RUNTIME", "/tmp/physicalrsi-real-robot-runtime"))
        if not (runtime / "manifest.json").is_file():
            self.skipTest("Prepare isolated Python runtime")
        context = self.context()
        source = '''def policy(robot, memory):
    result = robot.align("right", "hand marker", "object marker", ["front", "left"])
    if result["status"] != "image_aligned":
        robot.yield_control(result["reason"])
'''
        L1Controller(dict(source=source, memory={}), PolicyRuntime(runtime, timeout_s=20),
                     perception=PixelDetector()).run(context)
        self.assertLess(np.linalg.norm(context.robot.position-context.robot.target), .04)
        self.assertFalse(context.episode.events[-1]["result"]["independent_success"])


if __name__ == "__main__":
    unittest.main()
