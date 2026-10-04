"""PhysicalRSI task adapter for the Robopianist piano demos.

The adapter keeps PhysicalRSI's task-package and memory/skill contracts while
delegating the optional simulator run to a separately installed robopianist
checkout.  It never silently reports a simulator run as a physical qualification.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import uuid
import html
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json, file_digest
from PhysicalRSI.Embodied_Harness.memory.store import MemoryStore

PIECES = {
    "revolutionary": {
        "title": "Chopin — Étude Op. 10 No. 12 Revolutionary",
        "midi": "artifacts/full-song-baseline/revolutionary.mid",
        "skill": "piano.revolutionary.long_horizon",
    },
    "october": {
        "title": "Tchaikovsky — The Seasons Op. 37a No. 10 October",
        "midi": "assets/midi/tchaikovsky_october.mid",
        "skill": "piano.october.expression",
    },
}


class PianoDemo:
    protocol_version = 1

    def __init__(self, configuration, workspace, *, rsi_plan=None, harness=None, **_):
        self.configuration = dict(configuration)
        self.workspace = Path(workspace).resolve()
        self.rsi_plan = rsi_plan or {"scheme": "hybrid"}
        self.harness = harness

    @property
    def root(self) -> Path:
        return Path(self.configuration.get(
            "robopianist_root",
            os.environ.get("ROBOPIANIST_ROOT", "robopianist-rp1m"),
        )).expanduser().resolve()

    @property
    def python(self):
        return self.configuration.get("python", os.environ.get("ROBOPIANIST_PYTHON", sys.executable))

    def check(self):
        missing = []
        for name, piece in PIECES.items():
            if not (self.root / piece["midi"]).exists():
                missing.append(name)
        dependencies = {"ready": False}
        try:
            probe = subprocess.run([self.python, "-c",
                "import mujoco, dm_control, robopianist; from mujoco_utils import composer_utils"],
                cwd=self.root, capture_output=True, text=True, timeout=30)
            dependencies = {"ready": probe.returncode == 0, "error": probe.stderr[-2000:]}
        except (OSError, subprocess.TimeoutExpired) as error:
            dependencies["error"] = str(error)
        renderer = self.root / "examples/revolutionary_hybrid_demo.py"
        return {
            "ready": not missing and dependencies["ready"] and renderer.is_file(),
            "python": self.python,
            "dependencies": dependencies,
            "renderer_available": renderer.is_file(),
            "task": "piano",
            "pieces": list(PIECES),
            "missing_assets": missing,
            "backend": str(self.root),
            "scope": "Robopianist simulator adapter; no physical qualification",
            "render": self.configuration.get("render", {}),
        }

    def run(self, value=None):
        request = dict(value or {})
        if request.get("interactive", False):
            raise NotImplementedError(
                "Live keyboard/MIDI control is not implemented. "
                "Provide input_midi to render your own recorded performance, or run without "
                "interactive=true for the reference score.")
        render = {"width": 1920, "height": 1080, **self.configuration.get("render", {})}
        for dimension in ("width", "height"):
            size = render[dimension]
            if type(size) is not int or not 64 <= size <= 4096 or size % 2:
                raise ValueError(f"{dimension} must be an even integer between 64 and 4096")
        seconds = request.get("seconds")
        if seconds is not None and (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                                    or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("seconds must be a positive finite preview duration")
        fps = request.get("fps")
        if fps is not None:
            if (isinstance(fps, bool) or not isinstance(fps, (int, float))
                    or not math.isfinite(fps) or fps <= 0):
                raise ValueError("fps must be positive and finite")
            if not request.get("audit"):
                raise ValueError("Explicit fps requires an audited replay")
            render['fps'] = fps
        piece_name = request.get("piece", self.configuration.get("piece", "revolutionary"))
        if piece_name not in PIECES:
            raise ValueError(f"piece must be one of {sorted(PIECES)}")
        piece = PIECES[piece_name]
        midi = self.root / piece["midi"]
        input_midi = request.get("input_midi")
        if input_midi is not None:
            if not isinstance(input_midi, str) or not input_midi.strip():
                raise ValueError("input_midi must be a non-empty path")
            input_midi_path = Path(input_midi).expanduser()
            if not input_midi_path.is_absolute():
                input_midi_path = self.root / input_midi_path
            input_midi_path = input_midi_path.resolve()
            if not input_midi_path.is_file():
                raise FileNotFoundError(f"User MIDI performance not found: {input_midi_path}")
            midi = input_midi_path
        elif not midi.exists():
            raise FileNotFoundError(f"Missing MIDI asset for {piece_name}: {midi}")
        audit_data = None
        replay_source = None
        if request.get("audit"):
            audit_path = Path(request["audit"]).expanduser()
            if not audit_path.is_absolute():
                audit_path = self.root / audit_path
            audit_data = json.loads(audit_path.read_text())
            if audit_data.get("midi_sha256") != file_digest(midi):
                raise ValueError("Replay audit belongs to a different score")
            if audit_data.get("direct_key_actuation") is not False:
                raise ValueError("Replay audit must use passive keys")
            if not audit_data.get("controller_sha256"):
                raise ValueError("Replay audit must record the controller revision")
            if seconds is None and not audit_data.get("full_song"):
                raise ValueError("Full-song replay requires a full-song audit")
            if fps is not None:
                control_dt = audit_data['control_timestep']
                if not math.isclose(fps, 1/control_dt):
                    if not audit_data.get('recorded_physics_substeps'):
                        raise ValueError("Higher fps requires recorded physics substeps")
                    stride = 1/(fps*audit_data['physics_timestep'])
                    if (stride < 1 or not math.isclose(stride, round(stride))
                            or not math.isclose(control_dt*fps, round(control_dt*fps))):
                        raise ValueError("fps must divide the physics rate and preserve control boundaries")
            replay_source = {str(path): file_digest(path) for path in (
                audit_path, audit_path.with_suffix('.npz'),
                audit_path.with_suffix('.midi-events.json'))}
        episode = uuid.uuid4().hex
        output = self.workspace / "renders" / f"{piece_name}-{episode}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.python,
            str(self.root / "examples" / "revolutionary_hybrid_demo.py"),
            "--midi", str(midi), "--output", str(output),
            "--width", str(render["width"]), "--height", str(render["height"]),
        ]
        if audit_data is not None:
            command = [self.python, str(self.root / "examples/render_piano_contact_replay.py"),
                "--audit", str(audit_path), "--output", str(output),
                "--width", str(render["width"]), "--height", str(render["height"])]
        if seconds is not None:
            command.extend(["--seconds", str(seconds)])
        if fps is not None:
            command.extend(["--fps", str(fps)])
        config_path = self.root / self.configuration.get(
            "controller_config", "robopianist/rsi/configs/passive_ik.json")
        controller_config = None
        controller_evidence = None
        if audit_data is not None:
            controller_config = audit_data['controller_config']
        elif config_path.is_file():
            controller_config = json.loads(config_path.read_text())
            command.extend(["--config", str(config_path)])
            evidence_path = config_path.with_suffix('.evidence.json')
            if evidence_path.is_file():
                candidate_evidence = json.loads(evidence_path.read_text())
                if (candidate_evidence.get('controller_config_sha256') == file_digest(config_path)
                        and candidate_evidence.get('controller_sha256') == file_digest(
                            self.root / 'robopianist/rsi/hand_controller.py')):
                    controller_evidence = candidate_evidence
        elif "controller_config" in self.configuration:
            raise FileNotFoundError(config_path)
        skill_revision = file_digest(self.root / "robopianist/rsi/hand_controller.py")
        if audit_data is not None:
            skill_revision = audit_data['controller_sha256']
        render_sources = {name: file_digest(self.root / name) for name in (
            "examples/revolutionary_hybrid_demo.py",
            "examples/render_piano_contact_replay.py",
            "robopianist/models/arenas/humanoid_visual.py",
            "robopianist/models/arenas/piano_room_visual.py",
        ) if (self.root / name).is_file()}

        metadata = {
            "schema": "physicalrsi.piano-demo/v1",
            "episode_id": episode,
            "piece": piece_name,
            "title": piece["title"],
            "midi": str(midi),
            "midi_role": "user_performance" if input_midi is not None else "reference_score",
            "skill": piece["skill"],
            "memory": {
                "mode": "persistent_skill_memory",
                "skill_choice": piece["skill"],
                "lessons": [
                    "short-horizon contact windows compose into phrase checkpoints",
                    "preserve melody and release timing across long horizons",
                ],
            },
            "render": {
                "scene": "practice_room",
                "camera": "wide_piano_body",
                "quality": "high",
                **render,
                "text_overlay": False,
            },
            "preview_seconds": seconds,
            "full_song_requested": seconds is None,
            "interactive": bool(request.get("interactive", False)),
            "command": command,
            "scope": "simulator demo; visual/audio evidence only",
            "qualification": None,
            "controller_config": controller_config,
            "controller_evidence": controller_evidence,
            "direct_key_actuation": False,
            "render_sources": render_sources,
            "replay_source": replay_source,
        }
        snapshot = MemoryStore(self.workspace / "memory").snapshot({
            "piece": piece_name, "skill": piece["skill"],
            "midi_sha256": file_digest(midi),
            "midi_role": "user_performance" if input_midi is not None else "reference_score",
            "skill_revision": skill_revision,
            "controller_config": controller_config,
            "controller_evidence": controller_evidence,
            "render_sources": render_sources,
            "replay_source": replay_source,
            "lessons": metadata["memory"]["lessons"],
            "render": metadata["render"],
            "preview_seconds": seconds,
            "contact_consistency_verified": False,
        })
        metadata["memory"].update(revision=snapshot.revision, path=str(snapshot.root))
        metadata["midi_sha256"] = file_digest(midi)
        metadata["skill_revision"] = skill_revision
        metadata["state"] = "prepared"
        path = self.workspace / "episodes" / f"{episode}.json"
        atomic_json(path, metadata)
        if request.get("execute", False):
            process_env = os.environ.copy()
            process_env.setdefault("MUJOCO_GL", "egl")
            process_env.setdefault("PYOPENGL_PLATFORM", "egl")
            metadata["state"] = "running"
            atomic_json(path, metadata)
            try:
                completed = subprocess.run(command, cwd=self.root, check=True,
                    capture_output=True, text=True, env=process_env)
                if not output.is_file() or not output.with_suffix(".json").is_file():
                    raise RuntimeError("Renderer exited without video and metrics")
                metrics = json.loads(output.with_suffix(".json").read_text())
                if metrics.get("direct_key_actuation") is not False:
                    raise ValueError("Piano demo requires passive keys, not score actuation")
                if metrics.get("render", {}).get("text_overlay") is not False:
                    raise ValueError("Piano demo requires a video without text overlays")
            except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as error:
                metadata["state"] = "failed"
                metadata["error"] = str(error)
                metadata["renderer_stderr"] = (getattr(error, "stderr", "") or "")[-4000:]
                atomic_json(path, metadata)
                raise
            metadata["state"] = "completed"
            metadata["metrics"] = metrics
            metadata["video_sha256"] = file_digest(output)
            metadata["renderer_stdout"] = completed.stdout[-4000:]
            metadata["video"] = str(output) if output.exists() else None
            # A completed render is evidence, not a successful playing skill.
            result_memory = MemoryStore(self.workspace / "memory").snapshot({
                **snapshot.read(), "parent": snapshot.revision,
                "episode_id": episode, "metrics": metrics,
                "video_sha256": metadata["video_sha256"],
                "qualified": False,
            })
            metadata["memory"].update(parent=snapshot.revision,
                revision=result_memory.revision, path=str(result_memory.root))
            player = output.with_suffix(".html")
            player.write_text('<!doctype html><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1">'
                '<title>' + html.escape(piece["title"]) + '</title>'
                '<style>body{margin:0;background:#111;color:#eee;font:16px sans-serif}'
                'main{max-width:1400px;margin:auto;padding:20px}video{width:100%}</style>'
                '<main><h1>' + html.escape(piece["title"]) + '</h1>'
                '<video controls playsinline preload="metadata" src="' +
                html.escape(output.name, quote=True) + '"></video>'
                '<p>Simulator diagnostic. Audio follows measured keys. '
                'Contact consistency and performance qualification remain unverified.</p></main>',
                encoding="utf-8")
            metadata["playback"] = str(player)
        path = self.workspace / "episodes" / f"{episode}.json"
        atomic_json(path, metadata)
        return dict(metadata, evidence=str(path), video=str(output) if output.exists() else None)

    def evolve(self):
        return {
            "state": "available",
            "task": "piano",
            "skills": [piece["skill"] for piece in PIECES.values()],
            "memory": "persistent skill/memory revisions are recorded in each episode",
            "scope": "Self-Harness boundary exposed; simulator qualification remains separate",
            "qualification": None,
        }

    def status(self):
        return {
            "version": 1,
            "task": "piano",
            "pieces": {name: dict(piece) for name, piece in PIECES.items()},
            "backend": str(self.root),
            "rsi_plan": self.rsi_plan,
            "scope": "Robopianist simulator adapter; no physical qualification",
        }


def create(*, configuration, workspace, source_directory, language_model,
           rsi_plan=None, harness=None):
    return PianoDemo(configuration, workspace, rsi_plan=rsi_plan, harness=harness)


create.configuration_schema = {
    "type": "object",
    "properties": {
        "python": {"type": "string", "minLength": 1},
        "piece": {"enum": ["revolutionary", "october"]},
        "robopianist_root": {"type": "string", "minLength": 1},
        "controller_config": {"type": "string", "minLength": 1},
        "render": {"type": "object"},
    },
    "additionalProperties": False,
}
