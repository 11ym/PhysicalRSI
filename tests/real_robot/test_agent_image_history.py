"""Bound model image payloads without losing tool results or archived history."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PhysicalRSI_demos.real_robot.models import recent_image_history


class ImageHistoryTests(unittest.TestCase):
    def test_recent_groups_and_all_tool_pairs_survive_both_protocols(self):
        for protocol in ("chat", "responses"):
            image_type = "image_url" if protocol == "chat" else "input_image"
            text_type = "text" if protocol == "chat" else "input_text"
            history = [{"role": "system", "content": "policy"}]
            for i in range(20):
                history.extend([
                    {"role": "user", "content": [dict(type=text_type, text=f"obs-{i}")] +
                     [dict(type=image_type, image_url=f"{i}-{camera}") for camera in range(3)]},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": str(i)}]},
                    {"role": "tool", "tool_call_id": str(i), "content": f"result-{i}"}])
            original = copy.deepcopy(history)
            compact = recent_image_history(history)
            self.assertEqual(history, original)
            self.assertEqual(len(compact), len(history))
            images = [b["image_url"] for m in compact if isinstance(m.get("content"), list)
                      for b in m["content"] if b["type"] == image_type]
            self.assertEqual(images, [f"{i}-{c}" for i in (18, 19) for c in range(3)])
            self.assertEqual([m for m in compact if m["role"] in ("tool", "assistant")],
                             [m for m in history if m["role"] in ("tool", "assistant")])
            self.assertEqual(recent_image_history(compact), compact)
            self.assertEqual(compact[-3], history[-3])

    def test_current_images_cannot_all_be_removed(self):
        for keep in (0, -1, True):
            with self.assertRaises(ValueError):
                recent_image_history([], keep=keep)


if __name__ == "__main__":
    unittest.main()
