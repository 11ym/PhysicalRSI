# Auto-engineering integration TODO

## Objective

Demonstrate the complete physicalRSI loop on the existing Isaac Sim lab task:

**Native failure → System 2 proposal → Self-Harness validation → immutable skill
and memory revision → System 1 retrieval and reuse on unseen variations.**

Keep the current scene as the experimental task. This backlog describes future
work; it does not claim that skill condensation or transfer is already implemented.

## Current foundation

- The `auto-engineering` launcher route calls the core `SelfHarness`, survivor
  selector and `HarnessState`, with source hashes and native episode evidence.
- System 1 reuses reviewed pick/lift/carry/release code and reads parameter memory.
- System 2 currently proposes only a predefined transit-height repair.
- Simulation runs use a subprocess; this is not an attested security sandbox.
- A successful presentation video is separate from evidence of learning or reuse.

## 1. Integrate with the standard task workflow

- [ ] Register the lab task through the existing task/application interfaces so
  it is discoverable from `/demo`, while keeping Isaac dependencies optional.
- [ ] Map observations, actions, trajectories and results to existing core
  contracts; reuse core experiment/evidence storage instead of creating a second
  experiment framework. Preserve `scope` and `qualification` fields.
- [ ] Route native execution through the existing isolation facilities where
  supported, with time/resource limits and a declared policy input boundary.
  Document any remaining gaps rather than claiming sandbox enforcement.
- [ ] Make selected revisions loadable for subsequent runs without overwriting
  the original experiment workspace.

Acceptance: one documented launcher workflow can run the task, inspect its
evidence and selected revision, and start a fresh run from that revision.

## 2. Add evidence-backed skill condensation

- [ ] Define a reusable obstacle-clearance skill with inputs, preconditions,
  supported ranges, failure conditions, implementation version and evidence links.
- [ ] Let System 2 use development observations and failure traces to propose a
  conditional rule, rather than merely storing a fixed height such as `0.36 m`.
  A candidate rule is: observed obstacle top + the sample's downward extent from
  the grasp point + a validated safety margin, subject to reachability limits.
- [ ] Condense repeated development experience into a compact skill and memory
  entry. Retain raw episodes, counterexamples and provenance; do not replace
  evidence with an unverified narrative summary.
- [ ] Record the actual proposal mechanism and its inputs. Distinguish an
  agent-generated candidate from a manually authored engineering rule.
- [ ] Admit and evaluate proposals through the existing Self-Harness; publish
  accepted skills and memory as immutable revisions with parent and selection
  evidence. Preserve rollback to the previous revision.

Acceptance: a proposed skill can be traced to its source failures, validation
episodes and selection decision. Failed candidates remain inspectable.

## 3. Demonstrate retrieval and reuse

- [ ] Have System 1 retrieve the selected skill using current observations and
  declared sample specifications, and log the retrieved skill/memory revision.
- [ ] Reuse it on unseen obstacle heights, sample dimensions and layouts without
  rerunning System 2 or manually tuning the policy for each evaluation case.
- [ ] Check applicability before execution; reject or request further engineering
  when an observation lies outside the skill's validated range.
- [ ] Keep simulator object poses and private evaluator state out of proposal,
  condensation and policy inputs. Ground-truth evaluation and presentation
  recording remain separate, explicitly labelled consumers.

Acceptance: fresh runs show which skill was retrieved, why it applied, and the
native outcome, including an out-of-range case handled explicitly.

## 4. Run controlled comparisons

- [ ] Compare three frozen variants: no learned memory (original policy), fixed
  transit-height memory, and the retrieved conditional skill.
- [ ] Predeclare development/validation/held-out splits, variation ranges, seed
  counts, compute budgets and selection criteria before examining test outcomes.
  Use identical held-out cases across variants.
- [ ] Keep perception, controller and evaluator fixed across variants. Include
  cases that distinguish the conditional skill from simply raising the fixed
  height, such as declared reach or overhead-clearance limits.
- [ ] Report native successes and failures, obstacle/distractor disturbance,
  execution cost and applicability coverage. Count refusals separately; do not
  silently remove them from the evaluation denominator.
- [ ] Report uncertainty and preserve non-improving outcomes. Keep prototype
  results separate from results for the upgraded scene and changed task variants.

Acceptance: a reproducible report determines whether conditional skill reuse
helps relative to the simpler alternatives, without assuming a positive result.

## 5. Present the complete loop

- [ ] Build a report and short demo sequence showing failure, diagnosis, candidate
  skill, validation, selected revision and reuse on an unseen case.
- [ ] Link each claimed improvement to its execution conditions, observations,
  native receipts and exact skill/memory version.
- [ ] Keep path-traced presentation output labelled as replay of recorded states;
  it is not another rollout or additional success evidence.
- [ ] Add focused contract, revision-integrity and retrieval-boundary checks;
  run the full smoke suite when shared CLI or contract code changes. Require
  native Isaac runs for simulation claims.

Completion means the workflow and comparison are reproducible. It does not
establish physical qualification, general autonomous engineering, or transfer
to other benchmarks or robots.
