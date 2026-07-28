"""
resource_monitor.py
-------------------
psutil-based CPU / memory sampler for heavy phases.

``ResourceMonitor`` samples the whole process tree (parent + recursive children)
on a background thread and reports aggregate CPU% and RSS, plus a system-wide
memory peak. It is generic — any long-running or subprocess-spawning phase can
wrap itself in it to get before/after resource numbers:

    with ResourceMonitor() as mon:
        do_expensive_work()        # may spawn subprocesses
    stats = mon.result()           # aggregates dict

Two callers today: the asset-classifier training phases (which shell out to
mlx-lm / ollama) and the vector-search embedding pipeline (which loads a local
embedding model and streams batches to Qdrant). Both spawn or drive heavy work
whose real cost lives partly in child processes, which is why the monitor walks
the full tree rather than just ``psutil.Process()``.

Notes / caveats:
* CPU% is summed across the process tree, so it can exceed 100% on a multi-core
  machine (e.g. 730% ≈ ~7.3 cores busy).
* psutil cannot read Metal GPU utilisation. On Apple Silicon's unified memory,
  though, MLX / ONNX GPU buffers count toward process RSS, so ``mem_peak_mb`` is
  a meaningful proxy for the GPU memory footprint.
"""

from __future__ import annotations

import logging
import threading
import time

import psutil

logger = logging.getLogger(__name__)

_MB = 1024 * 1024

# Default sampling cadence. Callers with a strong opinion (e.g. the classifier's
# multi-minute training phases) pass their own ``interval``; this default keeps
# the class usable standalone without a config dependency.
DEFAULT_SAMPLE_INTERVAL_SEC = 2.0


class ResourceMonitor:
    """
    Sample CPU% and RSS for the current process tree on a background thread.

    Usage
    -----
        with ResourceMonitor() as mon:
            do_expensive_work()        # may spawn subprocesses
        stats = mon.result()           # aggregates dict

    The sampling thread re-discovers children every tick because heavy
    subprocesses are typically spawned *after* the monitor starts. psutil's
    per-process ``cpu_percent(interval=None)`` measures CPU since its previous
    call, and that "previous call" state lives on the ``Process`` *instance* — so
    the cached instances in ``self._procs`` are reused across ticks (rebuilding
    them every tick would reset every child's CPU reading to 0.0). The first
    sample for a freshly-seen process still reads 0% CPU (no prior reference
    yet), which is negligible across a multi-second-or-longer phase.
    """

    def __init__(self, interval: float = DEFAULT_SAMPLE_INTERVAL_SEC):
        self.interval = interval
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._cpu_sum = 0.0
        self._cpu_peak = 0.0
        self._mem_sum = 0.0
        self._mem_peak = 0.0
        self._sys_mem_peak = 0.0
        self._samples = 0
        # pid -> Process, reused across ticks to preserve cpu_percent() state.
        self._procs: dict[int, psutil.Process] = {}
        self._t0: float | None = None
        self._elapsed = 0.0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ResourceMonitor":
        self._t0 = time.monotonic()
        self._procs = {self._proc.pid: self._proc}
        self._prime(self._proc)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="resource-monitor"
        )
        self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2 + 5)
        if self._t0 is not None:
            self._elapsed = time.monotonic() - self._t0
        return False  # never suppress exceptions

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _prime(self, proc: psutil.Process) -> None:
        """First cpu_percent(None) call sets the reference point; returns 0.0."""
        try:
            proc.cpu_percent(None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def _refresh_tree(self) -> list[psutil.Process]:
        """
        Return the live process tree, reusing cached Process instances so each
        child's cpu_percent() state is preserved tick-to-tick. New children are
        added (and primed); dead ones drop out.
        """
        current: dict[int, psutil.Process] = {self._proc.pid: self._proc}
        try:
            for child in self._proc.children(recursive=True):
                # Reuse the cached instance when we've seen this pid before.
                current[child.pid] = self._procs.get(child.pid, child)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        for pid, proc in current.items():
            if pid not in self._procs:
                self._prime(proc)  # first reading would be 0 anyway; set its ref

        self._procs = current
        return list(current.values())

    def _sample(self) -> None:
        cpu = 0.0
        mem = 0.0
        for p in self._refresh_tree():
            try:
                cpu += p.cpu_percent(None)
                mem += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        mem_mb = mem / _MB
        try:
            sys_mem_mb = psutil.virtual_memory().used / _MB
        except Exception:
            sys_mem_mb = 0.0

        self._cpu_sum += cpu
        self._cpu_peak = max(self._cpu_peak, cpu)
        self._mem_sum += mem_mb
        self._mem_peak = max(self._mem_peak, mem_mb)
        self._sys_mem_peak = max(self._sys_mem_peak, sys_mem_mb)
        self._samples += 1

    def _run(self) -> None:
        # Sample immediately so even a short phase yields at least one reading,
        # then on each interval until stopped. A single bad tick (transient
        # psutil error) must never kill the thread and silently stop monitoring.
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:
                logger.exception("Resource sampling tick failed; continuing")
            self._stop.wait(self.interval)

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def result(self) -> dict:
        n = max(self._samples, 1)
        elapsed = self._elapsed
        if elapsed == 0.0 and self._t0 is not None:
            elapsed = time.monotonic() - self._t0
        return {
            "elapsed_seconds": elapsed,
            "cpu_avg_pct": self._cpu_sum / n,
            "cpu_peak_pct": self._cpu_peak,
            "mem_avg_mb": self._mem_sum / n,
            "mem_peak_mb": self._mem_peak,
            "sys_mem_peak_mb": self._sys_mem_peak,
            "sample_count": self._samples,
        }
