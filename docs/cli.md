# CLI and conversation

Run `physicalrsi` after installation, or `./physicalrsi` in the repository. The terminal retains the original wordmark, teal prompt, history and slash completion. `--plain` disables colors; repeated `--command` arguments execute commands in order and exit nonzero on failure.

## Command map

| Workflow | Commands |
| --- | --- |
| Preview and installation | `/doctor`, `/demos`, `/demo piano`, `/demo dexjoco`, `/baselines` |
| Core | `/experiment [id]`, `/demo`, `/run`, `/evolve`, `/status` |
| Task execution | `/task path.json`, `/draft`, `/check`, `/baseline [JSON]`, `/play [JSON]` |
| Self-Harness configuration | `/scheme hybrid`, `/rsi-plan`, `/harness`, `/rsi` |
| Memory and skills | `/skill-memory`, `/skills`, `/tools`, `/memory`, `/compose`, `/save`, `/load` |
| RoboDojo | `/robodojo`, `/robodojo TASK` |
| Dexjoco | `/layouts COUNT [SEED]`, `/collect COUNT`, `/train STEPS`, `/cycle ROUNDS` |
| Jobs | `/jobs`, `/logs [ID]`, `/cycle status`, `/cycle pause` |
| Conversation | `/model config.json`, natural-language text, `/chat TEXT`, `/reset-chat` |
| Exit | `/exit` or Ctrl-D |

`/status` reports application state. `/jobs` and `/logs` report the separate persistent workbench processes. `/cycle status` reports the configured pi05 cycle's recorded progress; inspect job/process evidence when determining whether a run is still alive. A process exit of zero alone does not qualify a policy.

## Preview

```bash
physicalrsi preview piano --workspace .physicalrsi --port 8765
physicalrsi preview dexjoco --workspace .physicalrsi --port 8765
physicalrsi --plain --command '/demos'
```

Open the returned URL. On a remote machine, use SSH port forwarding. The preview server persists after the command exits. A workspace records its port/PID under `showcase/server.json`; another workspace never silently attaches to it. Set `PHYSICALRSI_SHOWCASE_OUTPUT` before launch to store generated job output elsewhere.

For the explicit port above, run this on your own computer, replacing `user@remote-host` with your SSH destination, then open `http://127.0.0.1:8765/#piano`:

```bash
ssh -N -L 8765:127.0.0.1:8765 user@remote-host
```

## Conversation

Copy `configs/model.example.json`, set its API root/model, and set the named API-key environment variable outside the repository. Then:

```text
/model /path/to/model.json
Open the piano demo.
Show me the Dexjoco demo.
Generate two mouse layouts.
Show the latest job output.
Collect demonstrations on those two layouts.
Fine-tune the model for 100 steps.
```

Natural-language requests use bounded tools over this exact slash-command registry. Wait for dependent jobs to finish; the model cannot treat a job-start receipt as success. Slash commands and supplied recordings require no model credentials. `hello world` prints setup instructions when no model is configured.

## Live Dexjoco sequence

Configure `DEXJOCO_ROOT`, `DEXJOCO_PYTHON`, and `PHYSICALRSI_VLA_PYTHON` before starting the workbench. The installed checkout must include its mouse teacher at `rsi/click_mouse_datagen.py`.

```text
/layouts 2 3000
/jobs
/logs
/collect 2
/jobs
/train 100
/jobs
```

For continual pi05 training, configure the fleet/cycle first as described in [the workbench guide](demo-workbench.md):

```text
/cycle 2
/cycle status
/logs
/cycle pause
```

`/cycle 0` explicitly requests indefinite rounds until a pause is requested. Pausing takes effect between rounds. Inspect the recorded outcome of interrupted jobs before rerunning them; the workbench does not fabricate an exit code for an orphaned process.
