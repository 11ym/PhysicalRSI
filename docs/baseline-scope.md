# Baseline scope

This preview contains a local software demo and baseline adapters. The public product paths are `/demo` and `/baseline`.

The clone-ready fold-clothes task is `configs/fold_clothes.task.json`. It runs an abstract, deterministic phase model and records the full phase trace. The task mirrors the execution contract of a physical fold task; it does not simulate cloth dynamics or certify a robot.

The curated RoboDojo profile names two task examples: `stack_bowls` and `put_bottles_into_dustbin`. Their route entries are API and integration examples. They do not ship official skills, model weights, private simulator assets, historical layouts, or physical qualification evidence.

The code-policy boundary accepts paired development evidence and the task primitive API. Static checks reject imports, file access, network access, dynamic execution, and dunder escape paths. A proposal must pass admission and validation on fresh layouts before it becomes a committed route.

The software baseline validates package loading, preflight, execution wiring, and evidence persistence. Native perception, planning, simulator behavior, and robot qualification must be established independently in the target environment.
