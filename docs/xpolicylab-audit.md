# Skill library integration audit

The runtime exposes one skill library containing pi05, pi05-sparse-memory and
code-policy implementations, with separate weight identities recorded explicitly.
At episode start an API-backed agent reads instructions, observations and immutable
task-aware memory, then chooses a registered, configured skill composition. That
choice determines the actual execution skill. The composition remains stable until
reset; observations, policy instructions and actions are not rewritten. Missing
credentials or an invalid choice stop execution without a fallback.

The project author reports official review of the architecture described at
https://yanming03.github.io/PhysicalRSI/v2/. That page describes skill selection
within a harness containing code-policy, pi05 and sparse-memory capabilities.
This report does not independently certify that the current artifact is identical
to the reviewed or previously evaluated version.

Local unit tests exercise memory-conditioned skill choice, execution, lifecycle
and failure handling. Earlier debug transport and checkpoint inference checks are
not simulator success-rate evidence. Revalidate the final package through the
standard evaluation entrypoint before reporting performance.

Public asset availability, official evaluation results and the exact artifact's
publication status must be recorded separately. No performance equivalence or
verified leaderboard status is inferred from the common skill interface alone.
