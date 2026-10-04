# Prepared XPolicyLab integration

The runtime exposes one skill library containing pi05, pi05-sparse-memory and
code-policy implementations, with separate weight identities recorded explicitly.
At episode start an API-backed agent reads instructions, observations and immutable
task-aware memory, then chooses a registered, configured skill composition. That
choice determines the actual execution skill. The composition remains stable until
reset; observations, policy instructions and actions are not rewritten. Missing
credentials or an invalid choice stop execution without a fallback.

Reference architecture: https://yanming03.github.io/PhysicalRSI/v2/.
The author reports that this architecture has been reviewed. The package lists
internal checkpoints explicitly and records agent decisions, memory revisions and
execution revisions for comparison with the reviewed artifact.

Evaluation only: installation, runtime source, checkpoint downloads, and standard
server/client scripts are included. Training and data processing are unsupported.
No pull request has been created by this preparation step. Record fresh debug and
simulator results and artifact availability before claiming release readiness.
