# Node incidents during the benchmark (2026-10-25, UTC)

Source: `journalctl` (persistent journal) of the GPU node, read after recovery with
`kubectl debug node/<node> --profile=sysadmin` + `chroot /host`.

## Boot history

| Boot | Start | End | How it ended |
|---|---|---|---|
| -4 | 10-19 13:10 | **10-25 20:46:56** | **Abrupt stop.** No shutdown sequence, no kernel error |
| -3 | 10-25 21:16:04 | 21:16:49 | ACPI power button at 21:16:28 → clean power-off (recovery cycle) |
| -2 | 10-25 21:18:30 | **22:30:55** | **ACPI power button at 22:30:24** (repeated 22:30:34, 22:30:44) → clean power-off |
| -1 | 10-25 22:32:50 | 22:41:40 | ACPI power button at 22:41:27 → clean power-off (manual reboot by the account owner) |

## Incident 1 — 20:46:56: abrupt stop (host-side failure)

- The journal simply stops: the last lines are routine kubelet/sshd messages. There is **no** shutdown sequence, kernel panic, `Xid`/`NVRM` GPU error, PCIe AER, machine check, OOM or watchdog message.
- The only anomaly is 4 minutes earlier:
  ```
  20:42:44 kernel: clocksource: Long readout interval, skipping watchdog check: cs_nsec: 3174222416 wd_nsec: 3174222197
  ```
  The guest's clock read took **3.17 s**, meaning the VM's vCPUs were frozen for about 3 seconds by the hypervisor.
- **Conclusion**: the VM was stalled and then stopped from outside. This is a host/hypervisor failure, consistent with the Akamai host issue reported by the account owner. The workload was running a single 8k-token request at the time.

## Incident 2 — 22:30:24: ordered power-off from the platform

```
22:30:24 systemd-logind: Power key pressed short.
22:30:24 systemd-logind: Powering off...
22:30:24 kubelet: "Shutdown manager detected new shutdown event" event="shutdown"
22:30:34 systemd-logind: Power key pressed short.
22:30:44 systemd-logind: Power key pressed short.
22:30:55 systemd: System Power Off.
```

- "Power key pressed" is a **virtual ACPI power button** pressed by the hypervisor. Repeating it every 10 s is how the Linode platform performs a **Shut Down / Reboot** job (Cloud Manager, API, or an Akamai-side job), not something triggered inside the VM.
- No SSH login, `shutdown` command or error preceded it. Memory/disk/PID pressure were all `False`.
- The **same signature** appears at 21:16:28 (recovery after incident 1) and at 22:41:27 (manual reboot by the account owner).
- **Conclusion**: the node was powered off by a platform-level action. The workload (TP=1, 16 concurrent 10k requests) cannot press the ACPI power button. The account's Activity Feed shows who or what issued it.

## What was ruled out for both incidents

| Hypothesis | Evidence against |
|---|---|
| GPU fault | No `Xid` / `NVRM` errors in the kernel log |
| PCIe / IOMMU fault (TP=4 peer-to-peer) | No AER / DMAR / IOMMU messages. Incident 2 happened with **TP=1** (no inter-GPU traffic) |
| Power / thermal | Incident 1 (TP=4): GPUs at ≤381 W of 600 W and ≤62 °C. Incident 2 (TP=1): one GPU at its 600 W limit (~78 °C) is within spec, and a GPU power peak would not press the VM's ACPI power button |
| Out of memory | No OOM killer, kubelet reported `MemoryPressure False` until shutdown |
| Kernel crash | No panic / oops / lockup messages |
