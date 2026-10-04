# Demo workbench

Start from the physicalRSI CLI:

```bash
./physicalrsi --command '/baselines'
./physicalrsi --command '/demo piano'
./physicalrsi --command '/demo dexjoco'
```

Each command returns a local browser URL. On a remote machine, forward port 8765 to your browser. The server listens on localhost. `PHYSICALRSI_SHOWCASE_PORT` changes the port.

The initial workbench presents recorded RoboDojo rollouts, a piano performance, and Dexjoco demonstrations. Select a card to inspect its skill flow and play its video. The command panel supports the same navigation and `/play`. The baseline Run evaluation button starts the installed XPolicyLab entry and streams its output. The command panel also starts Dexjoco collection and SmolVLA fine-tuning.

Four supplied recordings are bundled under `PhysicalRSI_demos/media/` and streamed with byte-range support. Set `PHYSICALRSI_DATA_ROOT` to discover additional archived recordings and `ROBOPIANIST_ROOT` for optional piano iteration evidence. Model weights remain external.

The selected piano clip is the requested score-locked October performance. Its audio and hand poses follow the score; the accompanying metric is a score-actuator diagnostic. The copied piano skill library is linked from the demo README. Optional Skill evolution displays a separate archived contact experiment when the original piano checkout is configured.

To run the server directly:

```bash
python -m PhysicalRSI_demos.showcase.server --port 8765
```

## Live baseline evaluation

Before starting the workbench, set `PHYSICALRSI_XPOLICYLAB_ROOT` to an installed XPolicyLab checkout containing `policy/physicalRSI`, `PHYSICALRSI_SKILL_CONFIG` to its prepared skill configuration, `PHYSICALRSI_POLICY_PYTHON` to its Python executable, and `ROBODOJO_CONDA_ENV` to its simulator environment. Set the agent API key in the server environment using `PHYSICALRSI_AGENT_API_KEY`, `OPENAI_API_KEY`, or `ARK_API_KEY`.

Select a baseline and click **Run evaluation**. One evaluation runs at a time; the panel shows subprocess output and the actual exit status. `PHYSICALRSI_POLICY_GPU` and `PHYSICALRSI_ENV_GPU` select GPUs (both default to `0`). Results and logs go to `PHYSICALRSI_SHOWCASE_OUTPUT`, defaulting to `<workspace>/showcase/runs`. Completed rollout videos are added to the library.

In the piano tab, **Skill evolution** displays the installed October contact-skill iteration: candidate contact F1, retained candidates, and the resulting memory revision. This recorded iteration is separate from the higher-scoring performance clip. When a completed evaluation writes MP4 files under its run directory, the library discovers them after the run finishes.

For Dexjoco collection, set `DEXJOCO_ROOT` to the installed environment and teacher source checkout and `DEXJOCO_PYTHON` to its interpreter before starting the workbench. In the terminal or browser command panel, `/layouts 2` first generates and saves two initial states. Wait for it to finish, then `/collect 2` requests demonstrations on those exact layouts. The collector uses simulator-assisted teaching, records the rollout, and saves RGB/proprioception/action chunks only for successful episodes. `TMPDIR` must point to writable local storage for video encoding. The collector needs Zarr v2 and compatible numcodecs; from this checkout run `"$DEXJOCO_PYTHON" -m pip install ".[dexjoco]"` to install the declared collection extras alongside the base CLI. The Dexjoco simulator/teacher remains separately installed. After collection succeeds, `/train 100` starts 100 SmolVLA fine-tuning steps on that dataset. Set `PHYSICALRSI_VLA_PYTHON` to a Python environment containing LeRobot/SmolVLA; it defaults to `DEXJOCO_PYTHON`. `PHYSICALRSI_VLA_MODEL` selects a pretrained model or local checkpoint (default `lerobot/smolvla_base`). Training saves the policy, normalization statistics, and loss history. Training loss alone does not establish rollout success.

For natural-language requests, configure the CLI model with `/model` before opening the workbench. The launcher passes that workspace's model configuration to the server. A directly launched server uses `PHYSICALRSI_SHOWCASE_MODEL_CONFIG`. The conversation can request collection and training through bounded tools; credentials stay in environment variables. For example: “Generate two mouse layouts”, then “Collect two mouse demonstrations”, and finally “Fine-tune the VLA for 100 steps”. Wait for each prerequisite job to complete.

For an already authenticated local CLI model, set `PHYSICALRSI_SHOWCASE_AGENT_CLI` to its executable and optionally `PHYSICALRSI_SHOWCASE_AGENT_MODEL`. This invokes `exec` with structured output in a temporary read-only workspace. The model chooses a bounded collection or training request; the workbench executes it and streams the actual job output. This mode takes precedence over the API model configuration.

For a simulator installed in a Python virtual environment, set `PHYSICALRSI_SIM_PYTHON` instead of `ROBODOJO_CONDA_ENV`. This path starts the installed policy server and runs one headless RoboDojo episode with cameras enabled. It records under the run's `rollouts` directory; the standard conda path retains the installed adapter's evaluation settings.


## Repeated pi05 training

The pi05 cycle uses the installed Dexjoco source and a configured DSW fleet. Set these before starting the workbench:

```bash
export DEXJOCO_ROOT=/path/to/dexjoco
export DEXJOCO_PYTHON=/path/to/collection/python
export PHYSICALRSI_PI05_PYTHON=/path/to/pi05/python
export PHYSICALRSI_DSW_SSH_KEY=/path/to/ssh/key
export PHYSICALRSI_PI05_CYCLE="$DEXJOCO_ROOT/runs/pi05-cycle"
export PHYSICALRSI_SHOWCASE_AGENT_CLI=/path/to/authenticated/cli
```

The cycle directory contains `cycle.json` with the DSW `hosts`, `first_seed`, and recorded `rounds`. All hosts use the same shared source, data and checkpoint paths. The training environment needs the installed OpenPI dependencies and a working multi-host NCCL configuration. Choose a compatible NCCL installation and network interface for your fleet; no host addresses or SSH keys are bundled.

In the workbench, enter `/cycle 2`, or ask: “Continue pi05 for two rounds, generating 100 new layouts per round.” An active training round is followed without restarting its workers. Each round keeps its sampled layouts, generates successful demonstrations for all of them, trains pi05 with previous demonstrations included, and compares the candidate and incumbent on the same independent evaluation layouts. Normalization stays fixed across rounds. Failed generation attempts and agent-proposed repairs are retained. The agent proposes bounded teacher parameters; simulation determines whether they work.

The panel shows the current round, demonstration count, training step and loss. Training logs and rollout videos remain available in the workbench. A candidate is retained only when its paired evaluation success rate increases. Loss alone does not promote a model.

The per-round `evaluation-layouts.json` files are validation sets: their results
guide checkpoint selection. They are separate from the demonstration layouts
used for training and normalization. The fleet evaluator rejects overlapping
seeds, initial joint states or mouse poses. This measures new layouts of the
same task and sampling distribution, not generalization to different tasks.

Reserve `final-test/layouts.json` for a final evaluation after development and
checkpoint selection finish. Freeze the selected checkpoint before running it;
do not use final-test outcomes to update training data, teacher memory or select
another checkpoint. Report validation and final-test results separately.

`/cycle 0` continues across rounds. To finish the current round and stop before the next:

```bash
touch "$PHYSICALRSI_PI05_CYCLE/pause-requested"
```

Remove that request before starting another cycle. Each round saves layouts, teacher attempts, successful RGB/action demonstrations, normalization, training logs, checkpoints, evaluation videos and the checkpoint comparison. The completed round is also copied to `/mnt/data/physicalrsi-demo-workbench/pi05-cycle`.
