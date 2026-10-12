"""RGB-only synthetic tests of orientation and object-relative clearance.

Hidden scene coordinates are used to render images and independently check test
results, never to build controller requests or perception outputs.
"""
import hashlib
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PhysicalRSI_baselines.real_robot.contracts import Frame, Observation, Task
from PhysicalRSI_baselines.real_robot.evidence import Episode
from PhysicalRSI_baselines.real_robot.controllers import L1Controller
from PhysicalRSI_baselines.real_robot.policy_runtime import PolicyRuntime
from PhysicalRSI_baselines.real_robot.visual_perception import project_features
from PhysicalRSI_baselines.real_robot.visual_servo import ServoSettings, VisualServo, rotation, validate_relations

PALETTE = {
    "hand A": (0, 255, 0), "hand B": (0, 0, 255), "hand C": (255, 255, 0),
    "object A": (255, 0, 0), "object B": (255, 0, 255), "object C": (0, 255, 255),
    "object bottom": (255, 128, 0), "object top": (128, 0, 255), "support": (255, 255, 255),
}


class Detector:
    def look(self, observation, features, *, deadline):
        views = {}
        for frame in observation.frames:
            rgb = np.asarray(Image.open(frame.path).convert("RGB"))
            rows = {}
            for name in features:
                yy, xx = np.nonzero(np.all(rgb == PALETTE[name], axis=2))
                rows[name] = dict(visibility="visible" if len(xx) else "occluded", confidence=.95,
                    uv=[float(xx.mean())/(rgb.shape[1]-1), float(yy.mean())/(rgb.shape[0]-1)] if len(xx) else None)
            views[frame.camera] = rows
        return project_features(dict(observation_id=observation.id, views=views), observation, features)


class PosePlant:
    def __init__(self, root, *, lift=False, size=.15, flip=False, occlude=False, collinear=False, wrist=False):
        self.root, self.lift, self.size, self.occlude = root, lift, size, occlude
        self.wrist = wrist
        self.position = np.array([.04, -.03, 0.]) if lift else np.zeros(3)
        self.orientation = Rotation.identity() if lift else Rotation.from_rotvec([-.08, .1, -.05])
        self.target = np.array([.045, -.035, .035])
        self.target_orientation = Rotation.from_rotvec([.2, -.18, .15])
        self.landmarks = np.array([[.13, 0, 0], [0, .13, 0], [0, 0, .13]])
        if collinear:
            self.landmarks = np.array([[.06, 0, 0], [.12, 0, 0], [.18, 0, 0]])
        self.projections = [np.array([[1.4, 0, 0], [0, 0, -1.4]]),
                            np.array([[0, 1.4, 0], [0, 0, -1.4]]),
                            np.array([[1.2, 0, 0], [0, 1.2, 0]])]
        if flip:
            self.projections = [p @ np.diag([-1, -1, 1]) for p in self.projections]
        self.number, self.actions = 0, []

    def hold(self):
        pass

    def observe(self):
        self.number += 1
        captured, frames = time.time_ns(), []
        if self.lift:
            objects = [("support", np.zeros(3), 6), ("object top", self.position+[0, 0, self.size], 3),
                       ("object bottom", self.position, 3)]
        else:
            fixed = self.target_orientation.apply(self.landmarks)+self.target
            moving = self.orientation.apply(self.landmarks)+self.position
            objects = [("object "+name, xyz, 6) for name, xyz in zip("ABC", fixed)]
            objects += [("hand "+name, xyz, 2) for name, xyz in zip("ABC", moving)]
        for camera, projection in zip(("front", "left", "right"), self.projections):
            image = Image.new("RGB", (601, 601))
            draw = ImageDraw.Draw(image)
            for name, xyz, radius in objects:
                if self.occlude and camera == "left" and name in {"hand C", "object top"}:
                    continue
                camera_xyz = self.orientation.inv().apply(xyz-self.position) if self.wrist and camera == "right" else xyz
                u, v = (projection @ camera_xyz+.5)*600
                draw.ellipse((u-radius, v-radius, u+radius, v+radius), fill=PALETTE[name])
            path = self.root / f"{self.number}-{camera}.png"
            image.save(path)
            frames.append(Frame(camera, str(path), captured, hashlib.sha256(path.read_bytes()).hexdigest()))
        return Observation(f"obs-{self.number}", captured, tuple(frames), (0.,)*26,
            dict(left=dict(position=[0., .4, .2], quaternion_wxyz=[1., 0., 0., 0.]),
                 right=dict(position=self.position.tolist(), quaternion_wxyz=self.orientation.as_quat()[[3, 0, 1, 2]].tolist())))

    def act_eef(self, targets, steps, *, checkpoint, observation):
        if targets["left"] != dict(observation.eef["left"], hand=None):
            raise AssertionError("Other arm changed")
        if any(t["hand"] is not None for t in targets.values()):
            raise AssertionError("Held hand command changed")
        requested = rotation(targets["right"]["quaternion_wxyz"])
        dq = (requested*self.orientation.inv()).as_rotvec()
        if np.linalg.norm(dq) > .12+1e-9:
            raise AssertionError("Host rotation bound violated")
        self.orientation = Rotation.from_rotvec(dq*.8)*self.orientation
        self.position += .7*(np.asarray(targets["right"]["position"])-self.position)
        self.actions.append(targets)
        return dict(status="completed", ticks=2*steps)


class RelationsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def context(self, folder="case", **kw):
        root = self.root / folder
        root.mkdir()
        plant = PosePlant(root, **kw)
        episode = Episode(root / "episode", Task("relations", "Synthetic visual relations", "Independent test assertions"))
        def checkpoint(**kw):
            obs = plant.observe()
            episode.observation(obs)
            return obs
        return SimpleNamespace(robot=plant, episode=episode, checkpoint=checkpoint,
                               action=lambda name, args, invoke: invoke())

    def servo(self):
        return VisualServo(Detector(), settings=ServoSettings(image_tolerance=.006))

    @staticmethod
    def pose_relations():
        return [dict(camera=camera, moving="hand "+name, target="object "+name)
                for camera in ("front", "left", "right") for name in "ABC"]

    @staticmethod
    def clearance_relations():
        return [dict(camera=c, moving="object bottom", target="support",
                     offset={"from":"object bottom", "to":"object top", "scale":.5})
                for c in ("front", "left")]

    def test_full_pose_converges_from_rgb_without_object_pose_input(self):
        context = self.context(flip=True, wrist=True)
        result = self.servo().relate(context, "right", self.pose_relations(), "pose", deadline=time.monotonic()+30)
        self.assertEqual(result["status"], "image_aligned", result)
        plant = context.robot
        self.assertLess(np.linalg.norm(plant.position-plant.target), .02)
        self.assertLess((plant.orientation*plant.target_orientation.inv()).magnitude(), .1)
        self.assertFalse(result["independent_success"])
        evidence = [e["transition"] for e in context.episode.events if e["kind"] == "visual_servo_transition"]
        rotations = [e for e in evidence if np.linalg.norm(e["commanded_rotation"]) > .01]
        self.assertTrue(rotations)
        np.testing.assert_allclose(np.array(rotations[0]["commanded_rotation"])*.8,
                                   rotations[0]["measured_rotation"], atol=1e-9)

    def test_relative_lift_tracks_object_size_without_centimetre_goal(self):
        heights = []
        for index, size in enumerate((.12, .22)):
            context = self.context(str(index), lift=True, size=size, flip=index == 1)
            result = self.servo().relate(context, "right", self.clearance_relations(), "translation",
                                         deadline=time.monotonic()+30)
            self.assertEqual(result["status"], "image_aligned", result)
            self.assertAlmostEqual(context.robot.position[2], size*.5, delta=.008)
            self.assertLess(np.linalg.norm(context.robot.position[:2]), .01)
            self.assertLess(context.robot.orientation.magnitude(), 1e-9)
            heights.append(context.robot.position[2])
        self.assertGreater(heights[1]-heights[0], .04)

    def test_pose_occlusion_and_unobservable_roll_do_not_guess(self):
        context = self.context("occluded", occlude=True)
        result = self.servo().relate(context, "right", self.pose_relations(), "pose", deadline=time.monotonic()+30)
        self.assertEqual(result["status"], "unobservable")
        self.assertEqual(context.robot.actions, [])
        context = self.context("rank", collinear=True)
        result = self.servo().relate(context, "right", self.pose_relations(), "pose", deadline=time.monotonic()+30)
        self.assertEqual(result["status"], "insufficient_motion_evidence", result)
        self.assertEqual(len(context.robot.actions), 6)

    def test_relation_contract_rejects_fixed_pixel_offsets_and_underspecified_pose(self):
        rows = self.clearance_relations()
        rows[0]["offset"] = [0, -.2]
        with self.assertRaises(ValueError):
            validate_relations("right", rows, "translation")
        with self.assertRaises(ValueError):
            validate_relations("right", self.clearance_relations(), "pose")
        rows = self.clearance_relations()
        rows[0]["offset"]["scale"] = float("nan")
        with self.assertRaises(ValueError):
            validate_relations("right", rows, "translation")

    def test_isolated_policy_can_request_pose_relations(self):
        runtime = Path(__import__("os").environ.get("PHYSICALRSI_TEST_RUNTIME", "/tmp/physicalrsi-real-robot-runtime"))
        if not (runtime / "manifest.json").is_file():
            self.skipTest("Prepare isolated Python runtime")
        context = self.context(wrist=True)
        initial_angle = (context.robot.orientation*context.robot.target_orientation.inv()).magnitude()
        source = '''def policy(robot, memory):
    result = robot.relate("right", memory["relations"], "pose")
    if result["status"] != "image_aligned":
        robot.yield_control(result["reason"])
'''
        L1Controller(dict(source=source, memory=dict(relations=self.pose_relations())),
                     PolicyRuntime(runtime, timeout_s=30), perception=Detector()).run(context)
        final_angle = (context.robot.orientation*context.robot.target_orientation.inv()).magnitude()
        self.assertLess(final_angle, initial_angle*.5)
        self.assertFalse(context.episode.events[-1]["result"]["independent_success"])


if __name__ == "__main__":
    unittest.main()
