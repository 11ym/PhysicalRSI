"""Offline accounting must not discard interrupted or still-writing episodes."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PhysicalRSI_demos.real_robot.contracts import Level, Outcome, Task
from PhysicalRSI_demos.real_robot.evidence import Episode
from PhysicalRSI_demos.real_robot.reporting import report_episodes


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = Task("rod", "Grasp rod", "Lift and hold")

    def episode(self, name, **kwargs):
        return Episode(self.root / name, kwargs.pop("task", self.task), **kwargs)

    def success(self, episode):
        episode.event("controller_enter", level="L1")
        episode.finish(outcome=Outcome.SUCCESS, solved_by=Level.L1)

    def test_missing_result_stays_in_denominator_and_calls_are_lower_bounds(self):
        done, pending = self.episode("done"), self.episode("pending")
        self.success(done)
        pending.event("controller_enter", level="L1")
        pending.event("controller_enter", level="L2")
        report = report_episodes([done.root, pending.root])
        stats = report["groups"][0]["metrics"]
        self.assertEqual(stats["episodes"], 2)
        self.assertEqual(stats["overall_success_rate"], .5)
        self.assertEqual(stats["l1_success_rate"], .5)
        self.assertEqual(stats["l2_calls"], 1)
        self.assertEqual(stats["uncertain"], 1)
        self.assertEqual(stats["receipt_missing"], 1)
        self.assertTrue(stats["call_counts_are_lower_bounds"])
        self.assertEqual(stats["human_judgments_unknown_episodes"], 1)
        self.assertEqual(report["records"][1]["status"], "unknown")
        self.assertFalse((pending.root / "result.json").exists())

    def test_partial_result_and_event_count_only_contiguous_known_prefix(self):
        episode = self.episode("partial")
        episode.event("controller_enter", level="L2")
        (episode.public / "event-000001.json").write_text('{"index":')
        (episode.root / "result.json").write_text('{"outcome":"success",')
        report = report_episodes([episode.root])
        row = report["records"][0]
        self.assertEqual(row["outcome"], "uncertain")
        self.assertEqual(row["l2_calls"], 1)
        self.assertEqual(row["receipt"], "invalid")
        self.assertIn("event_history_incomplete_or_invalid", row["issues"])

    def test_tampered_counts_and_event_history_do_not_count_success(self):
        for name, mutate in (
                ("calls", lambda r: r.update(l2_calls=2)),
                ("bool_count", lambda r: r.update(l3_calls=True)),
                ("hash", lambda r: r.update(events_sha256="wrong")),
                ("policy", lambda r: r.update(policy_id="different")),
                ("credit", lambda r: r.update(l1_success=False))):
            with self.subTest(name=name):
                episode = self.episode(name)
                self.success(episode)
                path = episode.root / "result.json"
                result = json.loads(path.read_text())
                mutate(result)
                path.write_text(json.dumps(result))
                row = report_episodes([episode.root])["records"][0]
                self.assertEqual(row["outcome"], "uncertain")
                self.assertEqual(row["receipt"], "invalid")

    def test_task_criteria_split_qualification_and_policy_groups_are_distinct(self):
        episodes = [self.episode("dev", policy_id="p1"),
                    self.episode("dev2", policy_id="p2"),
                    self.episode("physical", qualification="physical"),
                    self.episode("validation", split="validation"),
                    self.episode("criteria", task=Task("rod", "Grasp rod", "Different criterion"))]
        for episode in episodes:
            self.success(episode)
        report = report_episodes([e.root for e in episodes])
        self.assertEqual(len(report["groups"]), 4)
        group = next(g for g in report["groups"] if g["metrics"]["episodes"] == 2)
        self.assertEqual({p["policy_id"] for p in group["policies"]}, {"p1", "p2"})
        self.assertTrue(all(not g["metrics"]["call_counts_are_lower_bounds"] for g in report["groups"]))

    def test_invalid_metadata_is_visible_in_unknown_group_and_alias_deduplicated(self):
        episode = self.episode("bad")
        (episode.root / "episode.json").write_text("{")
        alias = self.root / "alias"
        alias.symlink_to(episode.root, target_is_directory=True)
        report = report_episodes([episode.root, alias, self.root / "missing"])
        self.assertEqual(report["episodes"], 1)
        self.assertEqual(len(report["duplicate_inputs"]), 1)
        self.assertEqual(len(report["rejected_inputs"]), 1)
        self.assertEqual(report["groups"][0]["qualification"], "unknown")
        self.assertEqual(report["groups"][0]["metrics"]["uncertain"], 1)

    def test_no_private_judgment_or_linked_result_read(self):
        episode = self.episode("private")
        (episode.root / "result.json").symlink_to(episode.private / "secret.json")
        (episode.private / "secret.json").write_text('{"SECRET": true}')
        original = Path.read_text

        def guarded(path, *args, **kwargs):
            self.assertNotIn("evaluation", path.parts)
            self.assertFalse(path.is_symlink())
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", guarded):
            report = report_episodes([episode.root])
        self.assertEqual(report["records"][0]["receipt"], "invalid")
        self.assertNotIn("SECRET", json.dumps(report))

    def test_selected_policy_overrides_initial_manifest_and_l3_success_credit(self):
        episode = self.episode("selected", policy_id="old")
        episode.event("l1_policy_selected", level="L1", policy_id="new")
        episode.event("controller_enter", level="L3")
        episode.finish(outcome=Outcome.SUCCESS, solved_by=Level.L3)
        report = report_episodes([episode.root])
        stats = report["groups"][0]["metrics"]
        self.assertEqual(stats["overall_success_rate"], 1)
        self.assertEqual(stats["l1_success_rate"], 0)
        self.assertEqual(stats["l3_calls"], 1)
        self.assertEqual(report["records"][0]["policy_id"], "new")



if __name__ == "__main__":
    unittest.main()
