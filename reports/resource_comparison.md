# Resource Comparison

## Before

The legacy path ran `build-factor-factory` inside the cloud V5 research refresh.
It shared a daily chain capped at 60% CPU and 4 GB memory and could rescan and
reselect the broad factor space. Historical runtime telemetry did not isolate
Factor Factory RSS from the rest of that chain, so no fabricated legacy peak is
reported.

## After

| Stage | CPU limit | Memory limit | Observed wall/compute | Observed peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Cloud Factor snapshot request | 30% | 900 MB | about 9 s including systemd/status overhead | host did not retain inactive cgroup peak |
| NAS Factor compute | 3 CPUs | 8 GiB | 24.88 s | 354,426,880 B |
| Cloud Factor import | 40% | 1.2 GB | 4.06 s | hard-capped; no OOM |
| NAS bound Alpha compute | 3 CPUs | 8 GiB | 20.65 s | 1,439,264,768 B |
| Cloud Alpha import | 40% | 1.2 GB | 120.68 s | hard-capped; no OOM |
| Cloud V5 light refresh | 40% | 1.2 GB | production timer retained | 995,586,048 B observed peak |

The host's systemd version exposes live memory peak while a service is active
but does not retain `MemoryPeak` after these oneshots exit. The exact cloud peak
therefore remains unobserved; acceptance is based on enforced cgroup limits and
successful completion without OOM. This limitation is recorded rather than
replacing it with an estimate.

The restored V5 refresh was observed live at 995,586,048 bytes memory peak and
1,993,785,344 bytes swap peak. It remained below `MemoryMax`, and the heavy
Factor command was absent, but the swap volume is a separate performance risk.

## Data movement

| Task | Snapshot input | Downloaded | Blob cache hit | Result bytes |
| --- | ---: | ---: | ---: | ---: |
| Factor Research | 388,332 B | 311,072 B | 77,260 B (19.90%) | 554,306 B |
| Bound Alpha Factory | 16,295,243 B | 16,220,714 B | 74,529 B (0.46%) | 2,562,375 B |

Factor Research no longer copies or scans the full Lake on NAS. The cloud seals
projected immutable inputs; the Worker uses the shared content-addressed cache.
The low Alpha cache ratio is expected for the first exact final-commit snapshot.

## Current capacity

- qyun2 root: 79 GiB total, 41 GiB used, 36 GiB free, 54% used.
- qyun2 Lake: about 12.6 GiB; Research Queue: about 97 MiB.
- NAS `/volume1`: about 916 GiB total, 796 GiB available, 13% used.
- NAS memory: 15.4 GiB; both tasks remained below the 6 GiB acceptance peak.

This change reduces cloud heavy-compute exposure and bounds the research search
space. It does not prove that the tested hypotheses are profitable.
