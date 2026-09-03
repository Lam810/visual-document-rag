# Ascend Verification Log

Real-hardware notes from testing this project on a Huawei Ascend NPU cluster. Kept
as a dated log rather than folded into the README, because some of it is an open
problem, not a solved one — future readers (including future us) need the
timeline and the ruled-out causes, not just a conclusion.

Cluster identifying details (hostnames, accounts, IPs) are intentionally omitted;
what's kept is everything relevant to reproducing or diagnosing the same class of
issue on any Slurm-managed Ascend cluster.

## Environment

| Component | Version |
| --- | --- |
| Cluster scheduler | Slurm, GPU/NPU partition via `--gres=npu:N` |
| NPU | Ascend 910B (reported as `Ascend910` / `Ascend910_9382` depending on the query) |
| CANN | 8.5.0 |
| Driver | 25.5.0 |
| OS / arch | openEuler 22.03 LTS-SP4, aarch64 |
| Python | 3.10.19 (via `uv`) |
| torch / torch-npu | 2.7.1 / 2.7.1 (PyPI `manylinux_2_28_aarch64` wheels) |

Install (via `uv`, aarch64 wheels resolve cleanly from PyPI, no special index needed):

```bash
uv venv --python 3.10 vdr-env
source vdr-env/bin/activate
uv pip install torch==2.7.1 torch-npu==2.7.1
```

## Issue 1 — `torch-npu` has undeclared runtime dependencies, and the failure mode is worse than a missing import

**Symptom:** a fresh `torch==2.7.1` + `torch-npu==2.7.1` install (nothing else) raises on
plain `import torch`:

```
File ".../torch_npu/npu/_memory_viz.py", line 10, in <module>
    import yaml
ModuleNotFoundError: No module named 'yaml'
...
RuntimeError: Failed to load the backend extension: torch_npu. You can disable
extension auto-loading with TORCH_DEVICE_BACKEND_AUTOLOAD=0.
```

**Why this is worse than it looks:** since PyTorch 2.6, `import torch` auto-loads
registered accelerator backends through Python's entry-point mechanism. `torch_npu`
registers itself this way, so a broken `torch_npu` doesn't fail where you'd expect
(`import torch_npu`) — it fails inside `import torch` itself, as a `RuntimeError`,
not an `ImportError`. Code that only catches `ImportError` around a `torch_npu`
probe (a reasonable thing to write before knowing this) will crash instead of
falling back to CPU.

**Fix applied:** `numpy<2.0` and `pyyaml` added to
[`requirements-npu.txt`](requirements-npu.txt); [`device.py`](device.py)'s
detection now catches `Exception`, not just `ImportError`, around every
`import torch`.

## Issue 2 — PaddleOCR's `use_gpu` silently outranks `use_npu`

Found by reading PaddleOCR's `tools/infer/utility.py` upstream, not by hardware
reproduction: device selection is an `if use_gpu: ... elif use_npu: ...` chain, and
`check_gpu()` quietns `use_gpu` back to `False` when Paddle isn't CUDA-built —
so leaving `use_gpu: true` in the OCR stage's config on an NPU box doesn't error,
it just loses the `elif` branch and runs on CPU with no warning.

**Fix applied:** [`pipeline.py`](pipeline.py)'s `_ocr_device_flags` sets the two
flags as a mutually exclusive pair derived from the resolved device, never both.

## Issue 3 — every upload reprocessed the entire document history

Not Ascend-specific, but sharpened by it: PDF-Extract-Kit takes a directory, not a
file, so pointing it at the upload folder reprocesses every previously uploaded
PDF through the full vision pipeline on every new upload. On a rented/shared NPU
allocation that's wasted accelerator time, not just wasted wall-clock.

**Fix applied:** [`pipeline.py`](pipeline.py)'s `_scoped_inputs` stages a
hardlink (or copy) of just the new file into a temporary directory per run.

## Issue 4 — NPU device open hangs indefinitely (UNRESOLVED, cluster-side)

This is the one real-hardware run couldn't get past, and it blocks verifying
everything downstream: the CUDA→NPU shim under real load, the ModelScope
embedding relocation, PaddleOCR on-device, and the PDF-Extract-Kit pipeline
end-to-end.

**Symptom:** driver and metadata calls return correct answers instantly -
`torch.npu.is_available()` → `True`, `torch.npu.device_count()` → `2`,
`torch.npu.get_device_name(0)` → a real device name, `npu-smi info` reports
`Health: OK` with no stale processes. The first call that actually opens the
device for compute - `torch.npu.set_device(0)`, or any tensor's `.npu()` -
hangs with no exception, no output, indefinitely.

**Reproduced 3 times, identically:**

| # | Node | Launch method | Outcome |
| --- | --- | --- | --- |
| 1 | node A | interactive `srun` | hung ~13 min until the Slurm `--time` limit killed it |
| 2 | node B | interactive `srun`, explicit `ASCEND_RT_VISIBLE_DEVICES=0,1` | hung, killed by a 45s internal `timeout` |
| 3 | node B | `sbatch` (a plain batch script, to rule out anything about an interactive session) | hung, killed by a 60s internal `timeout` (exit 124) |

**Ruled out:**
- **Contention** - the partition had no other running jobs at the time of testing.
- **Permissions** - all `/dev/davinci*` nodes are world read-write
  (`crw-rw-rw-`); the CANN install directory being owned by `root` produces a
  cosmetic `torch_npu` warning but is normal for a shared toolkit install and
  unrelated to the hang.
- **How the job was launched** - interactive `srun` and batch `sbatch` fail the
  same way, so it isn't about session/pty inheritance through automation.
- **Device targeting** - explicit `ASCEND_RT_VISIBLE_DEVICES` made no difference.

**Observation that may be the actual cause:** a job granted `--gres=npu:2` can
still see all of a node's device files (16 in this case), and nothing in the job
environment (`ASCEND_RT_VISIBLE_DEVICES`, `HCCL_*`, or otherwise) indicates which
2 are actually reserved for it. That smells like the Slurm GRES plugin for this
partition isn't scoping devices per job the way the NVIDIA/CUDA plugin does for
`CUDA_VISIBLE_DEVICES` - and if two jobs land on the same node, both defaulting
to opening device 0, that's a plausible way to get exactly this kind of silent
hang at the HDC (host-device communication) layer. This is a hypothesis, not a
confirmed root cause; confirming it needs access this account doesn't have
(host-level HDC daemon logs, other users' job outcomes).

**Root cause (confirmed 2026-09-03): a node-level hardware fault, not a
configuration or code problem.** The original reproduction pinned the job to a
specific node with `-w`, which is exactly what hid the answer. Re-running the
same `sbatch` script three days later *without* naming a node let the scheduler
place it elsewhere - and it worked on the first try, in 16 seconds:
`set_device(0)` returned immediately and a real matmul produced a value. By
then the cluster's own monitoring had drained both originally-tested nodes with
the reason `Not responding`.

One of the two bad nodes did eventually surface a real CANN error instead of
hanging, and it points at the same layer:

```
NPU function error: error code is 507033
Inner_Error_Device_Subprocess_Startup_Timeout(E39007)
        Possible Cause: 1.Failed to start the subprocess. 2. The HDC link is faulty.
        tsd client wait response fail ... check hdc service
        TsdOpen failed. devId=0, tdt error=31
        open device 0 failed, runtime result = 507033
```

**Two hypotheses this log carried earlier were wrong, and are corrected here
rather than deleted, because both are tempting and both cost time:**

1. *"The Slurm GRES plugin isn't scoping devices per job."* No. The working node
   presents an identically bare environment - no `ASCEND_RT_VISIBLE_DEVICES`, no
   `SLURM_JOB_GRES`, all 16 device files visible to a 2-device request - and
   serves compute fine. Device isolation does happen, just not through the
   environment: a job allocated card 3 gets `Invalid card id` from
   `npu-smi info -t board -i 0`, so it genuinely cannot address cards it wasn't
   given.
2. *"Bare `srun`/`sbatch --gres=npu:N` is an unsupported path; accelerator work
   must go through a container/portal service."* No. Plain `sbatch --gres=npu:2`
   is sufficient. This one came from reading platform documentation and pattern-
   matching it onto the symptom, without testing the claim - the same failure
   mode the rest of this log exists to avoid.

**Practical takeaway for anyone hitting a device-open hang on a Slurm-managed
Ascend cluster:** do not pin a node, and do not start by auditing your own
configuration. Resubmit and let the scheduler choose; if it then works, you had
a bad node. Check the partition for `drained` nodes to see which hardware the
cluster already knows about. Diagnosing an unhealthy accelerator from inside a
user-level job is close to impossible - the decisive evidence (host HDC/TSD
daemon logs) is not readable from there.

**Status: resolved.** `scripts/check_ascend.py` keeps its 60-second `SIGALRM`
guard around the device-touching calls: the hang itself was real, and turning a
silently-consumed job allocation into a one-minute readable `FAIL` is worth
keeping whether the cause is a bad node or anything else.

## What this does and doesn't verify

**Verified on real Ascend hardware (Ascend 910C, 2 dies per card, 64 GB HBM per
die, CANN 8.5.0, aarch64):** driver detection, `torch`/`torch_npu` installation
and import (including the crash in Issue 1), device metadata queries, `npu-smi`
health reporting, and - since 2026-09-03 - **NPU compute itself**:
`torch.npu.set_device()` followed by a real matmul returns a correct result.

**Still unverified:** the CUDA→NPU shim's effect under real model load,
ModelScope embedding inference, PaddleOCR inference, and the PDF-Extract-Kit
pipeline end-to-end. Issue 4 no longer blocks these - they simply have not been
run yet. The device-selection
logic for all of these was verified separately with stubbed `torch_npu` /
`paddle` modules (correct device strings for every backend/shim-state
combination), but a stub proves the *logic* is right, not that inference
*runs* - that step is still open.
