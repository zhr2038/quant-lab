# Migration and Compatibility

Legacy `auto.single.*`, old `PAPER_READY`, and historical factor outputs remain in
the Lake for audit and old-report compatibility. They no longer seed the new
authoritative lifecycle. The retirement registry disables known invalid factors
without deleting history.

Existing Research Plane task types remain strict and backward compatible.
Factor Research fields are accepted only under the `factor_research`
discriminator. No second queue, SSH account, worker, or cache was introduced.

Legacy AI proposal tables remain readable as historical data. New Stage 2 tasks
use the v4 hypothesis-draft contract and write new Gold datasets. New workers
refuse old implementation-shaped tasks instead of silently mixing semantics.

The old cloud Factor Factory command remains available for explicit maintenance
and historical reproduction, but it is removed from the hourly production
refresh. The new NAS timer is separately gated and defaults off.
