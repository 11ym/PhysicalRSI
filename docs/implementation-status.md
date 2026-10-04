# Implementation status

This preview now includes a complete local software Self-Harness interaction in `PhysicalRSI_demos/`. The interaction is intentionally deterministic so it can verify the control flow without external model weights or a simulator.

| Area | Preview status | Evidence |
| --- | --- | --- |
| Memory snapshot and typed read | Implemented | `PhysicalRSI/Embodied_Harness/memory/` |
| System 1 execution contract | Implemented in the local service demo | `PhysicalRSI_demos/runtime.py` |
| Trajectory with proposal and transition records | Implemented | `PhysicalRSI_core/infra/trajectory.py` and the runtime demo |
| System 2 proposal boundary | Implemented as a deterministic proposer port | `PhysicalRSI_demos/runtime_evolution.py` |
| Parent and child admission | Implemented | `PhysicalRSI_core/self_harness/` |
| Fresh paired evaluation and survivor selection | Implemented for the local counter contract | `PhysicalRSI_demos/runtime_evolution.py` |
| Persistent lineage and restart-safe receipts | Implemented | `PhysicalRSI_core/lineage/` and the demo workspace |
| External model-generated code or skill repair | Adapter boundary only | Requires an explicitly configured provider |
| pi05 / pi05-sparse-memory adapter metadata | Implemented | `PhysicalRSI_baselines/robodojo/skills/` |
| pi05 / pi05-sparse-memory training | Provider required | RoboDojo is evaluation-only; the separate Dexjoco demo has SmolVLA and pi05 training entry points, with external dependencies and weights |
| External provider and checkpoint manifest | Implemented | `PhysicalRSI_baselines/robodojo/provider.py` |
| Reviewed primitive-only code-policy demo | Implemented | `PhysicalRSI_baselines/robodojo/code_policy_demo.py` |
| Core skill and exploration-memory catalog demo | Implemented | `PhysicalRSI_demos/skill_memory.py`, `/skill-memory` |
| Native RoboDojo or robot qualification | Not included | Requires external simulator, policies, assets, and environment evidence |

Run the local evidence path with:

```bash
python -m PhysicalRSI_demos.runtime --workspace /tmp/physicalrsi-runtime
python -m PhysicalRSI_demos.runtime_evolution --workspace /tmp/physicalrsi-evolution
```

The `/evolve` CLI command invokes the same Self-Harness demo. A successful local run proves software contracts, evidence binding, and lineage behavior; it does not prove physical-task success or benchmark performance.

The public release also includes the versioned experiment runtime, a workspace-bound media server, four bundled recordings, a copied piano skill library, and terminal/conversation commands for Dexjoco layouts, collection, training and cycle control. Simulator and training execution require configured external installations.
