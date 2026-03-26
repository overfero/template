"""
run_tracking_parallel.py
========================
Launcher untuk 2 kamera paralel menggunakan subprocess + taskset.
Taskset di-apply di level OS SEBELUM Python dimulai → persis seperti 2 terminal manual.
Setelah kedua worker selesai, hasilnya di-fuse di post-processing.

Usage:
    python3 run_tracking_parallel.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.abspath(__file__))
WORKER_SCRIPT = os.path.join(ROOT, "tracking_worker.py")

# ── Konfigurasi ────────────────────────────────────────────────────────────
PYTHON      = sys.executable   # pakai interpreter yang sama dengan main process
CORES_TOP   = "0-3"
CORES_BOT   = "4-7"
OMP_THREADS = "4"

# ── Data structures ────────────────────────────────────────────────────────
@dataclass
class FrameState:
    frame: int
    taken: list[dict] = field(default_factory=list)
    # taken entry: {"product": str, "trail_len": int}


@dataclass
class FusedState:
    frame: int
    taken_product: str | None = None
    trail_len: int = 0
    source: str = ""
    top_taken: list[dict] = field(default_factory=list)
    bot_taken: list[dict] = field(default_factory=list)


# ── Fusion ─────────────────────────────────────────────────────────────────
def _best(taken_list: list[dict]) -> dict | None:
    return max(taken_list, key=lambda x: x["trail_len"]) if taken_list else None


def fuse_all(
    top_frames: list[FrameState],
    bot_frames: list[FrameState],
) -> list[FusedState]:
    """
    Fusion strategy per frame:
    ┌────────────┬────────────┬───────────────────────────────────────────┐
    │ top taken  │ bot taken  │ Keputusan                                 │
    ├────────────┼────────────┼───────────────────────────────────────────┤
    │ ✅         │ ❌         │ Pakai top                                 │
    │ ❌         │ ✅         │ Pakai bottom                              │
    │ ✅ (A)     │ ✅ (A)     │ Agree → pakai yang trail lebih panjang    │
    │ ✅ (A)     │ ✅ (B)     │ Conflict → pakai yang trail lebih panjang │
    │ ❌         │ ❌         │ Tidak ada yang taken                      │
    └────────────┴────────────┴───────────────────────────────────────────┘
    """
    n = max(len(top_frames), len(bot_frames))
    fused: list[FusedState] = []

    for i in range(n):
        top_f = top_frames[i] if i < len(top_frames) else FrameState(frame=i + 1)
        bot_f = bot_frames[i] if i < len(bot_frames) else FrameState(frame=i + 1)

        best_top = _best(top_f.taken)
        best_bot = _best(bot_f.taken)

        state = FusedState(
            frame=i + 1,
            top_taken=top_f.taken,
            bot_taken=bot_f.taken,
        )

        if best_top and not best_bot:
            state.taken_product = best_top["product"]
            state.trail_len     = best_top["trail_len"]
            state.source        = "top"

        elif best_bot and not best_top:
            state.taken_product = best_bot["product"]
            state.trail_len     = best_bot["trail_len"]
            state.source        = "bottom"

        elif best_top and best_bot:
            if best_top["trail_len"] >= best_bot["trail_len"]:
                winner, src = best_top, "top"
            else:
                winner, src = best_bot, "bottom"
            label = "agree" if best_top["product"] == best_bot["product"] else f"conflict→{src}"
            state.taken_product = winner["product"]
            state.trail_len     = winner["trail_len"]
            state.source        = label

        fused.append(state)

    return fused


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> list[FusedState]:
    # Buat file temporer untuk output masing-masing worker
    fd_top, path_top = tempfile.mkstemp(suffix="_top.json", prefix="sfridge_")
    fd_bot, path_bot = tempfile.mkstemp(suffix="_bot.json", prefix="sfridge_")
    os.close(fd_top)
    os.close(fd_bot)

    # Env untuk setiap worker (thread dibatasi sesuai core yang di-pin)
    def worker_env(omp_n: str) -> dict:
        e = os.environ.copy()
        e.update({
            "OMP_NUM_THREADS":      omp_n,
            "GOMP_SPINCOUNT":       "0",
            "MKL_NUM_THREADS":      omp_n,
            "OPENBLAS_NUM_THREADS": omp_n,
        })
        return e

    # Launch via taskset — pinning terjadi di kernel level SEBELUM Python start
    cmd_top = ["taskset", "-c", CORES_TOP, PYTHON, WORKER_SCRIPT, "--camera", "top",    "--out", path_top]
    cmd_bot = ["taskset", "-c", CORES_BOT, PYTHON, WORKER_SCRIPT, "--camera", "bottom", "--out", path_bot]

    print(f"[MAIN] Launching CamTop  : {' '.join(cmd_top)}")
    print(f"[MAIN] Launching CamBot  : {' '.join(cmd_bot)}")
    print("[MAIN] Menunggu kedua worker selesai...\n")

    t_wall = time.perf_counter()

    proc_top = subprocess.Popen(cmd_top, env=worker_env(OMP_THREADS))
    proc_bot = subprocess.Popen(cmd_bot, env=worker_env(OMP_THREADS))

    # Tunggu keduanya selesai
    ret_top = proc_top.wait()
    ret_bot = proc_bot.wait()

    elapsed_wall = time.perf_counter() - t_wall
    print(f"\n[MAIN] Kedua worker selesai dalam {elapsed_wall:.1f}s")

    if ret_top != 0 or ret_bot != 0:
        print(f"[MAIN] ⚠️  Worker exit codes: top={ret_top} bot={ret_bot}")

    # Baca hasil JSON
    with open(path_top) as f:
        top_raw = json.load(f)
    with open(path_bot) as f:
        bot_raw = json.load(f)

    os.unlink(path_top)
    os.unlink(path_bot)

    top_frames = [FrameState(frame=r["frame"], taken=r["taken"]) for r in top_raw]
    bot_frames = [FrameState(frame=r["frame"], taken=r["taken"]) for r in bot_raw]

    # ── Post-processing fusion ──────────────────────────────────────────
    fused = fuse_all(top_frames, bot_frames)

    # ── Tulis semua per-frame state ke CSV ─────────────────────────────
    import csv
    csv_path = os.path.join(ROOT, "fusion_result.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "taken_product", "trail_len", "source",
                          "top_taken", "bot_taken"])
        for s in fused:
            top_str = "; ".join(f"{x['product']}(t={x['trail_len']})" for x in s.top_taken)
            bot_str = "; ".join(f"{x['product']}(t={x['trail_len']})" for x in s.bot_taken)
            writer.writerow([
                s.frame,
                s.taken_product or "",
                s.trail_len,
                s.source,
                top_str,
                bot_str,
            ])

    taken_count = sum(1 for f in fused if f.taken_product)
    print(f"[MAIN] CSV → {csv_path}")
    print(f"[MAIN] {len(fused)} frames fused  |  {taken_count} taken events  |  {elapsed_wall:.1f}s")

    return fused


if __name__ == "__main__":
    main()
