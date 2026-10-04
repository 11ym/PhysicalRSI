# Executable manipulation skills

- `pi05.py`: observation-conditioned manipulation with a supplied checkpoint.
- `sparse.py`: manipulation with per-environment causal visual history.
- `code_policy.py`: isolated execution of a verified primitive program assembly.

`execution_skills.load_skill` adapts these implementations to the common
observe/execute/reset/close lifecycle. `compose_skill` builds a PhysicalRSI
`Operation` sequence with typed inputs and outputs. The memory stage uses the
standard immutable `MemoryStore` snapshot reader, not a separate memory format.

The pi05 and sparse backend sources live here. OpenPI supplies the model
architecture; trained weights must be provided explicitly. The code-policy loader
requires a pinned source, harness, primitive assembly, and resource configuration.
A capability description alone is never treated as executable code.

## Unified inference package

The runtime exposes one skill library containing pi05, pi05-sparse-memory and
code-policy implementations, with separate weight identities recorded explicitly.
At episode start an API-backed agent reads instructions, observations and immutable
task-aware memory, then chooses a registered, configured skill composition. That
choice determines the actual execution skill. The composition remains stable until
reset; observations, policy instructions and actions are not rewritten. Missing
credentials or an invalid choice stop execution without a fallback.

`configs/skill_package.json` records all implementations and asset identities.
This is a unified inference package, not a merged checkpoint.

## Frozen code programs

Each code program can be registered with its own skill name and
`implementation: code-policy`. Its composition uses `memory.snapshot` followed
by that skill name. `composition_definitions` extends the frozen catalog without
replacing built-in definitions. Descriptions tell the agent what each skill does.

The `frozen-program-service` runtime accepts a pinned command, dependency file
hashes, working directory, environment, and output root. It starts an owned
WebSocket service, forwards observations and action requests, normalizes gripper
aliases, and closes the service on reset or failure. It does not read a benchmark
task-to-program lookup. The primitive-assembly runtime remains available.
