"""Watchdog that monitors the active GPU experiment (PID 21568).
As soon as the active experiment finishes and releases GPU memory,
it triggers the massive scaled-up experiment automatically.
"""

import os
import subprocess
import sys
import time

TARGET_PID = 21568
POLL_SEC = 10

print("================================================================================", flush=True)
print(" [GPU Watchdog Service Active]", flush=True)
print(f" Monitoring PID: {TARGET_PID} | Polling every {POLL_SEC}s", flush=True)
print(" As soon as the GPU is released, 'scale_up_massive_model.py' will run.", flush=True)
print("================================================================================", flush=True)

def is_pid_alive(pid):
    try:
        res = subprocess.check_output(
            ["powershell", "-Command", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id"],
            encoding='utf-8'
        ).strip()
        return len(res) > 0
    except Exception:
        return False

def get_gpu_memory():
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,nounits,noheader"],
            encoding='utf-8'
        ).strip().split(',')
        return int(res[0].strip()), int(res[1].strip())
    except Exception:
        return 0, 0

start_t = time.time()
while True:
    alive = is_pid_alive(TARGET_PID)
    mem, util = get_gpu_memory()
    elapsed_min = (time.time() - start_t) / 60.0

    print(f"[{time.strftime('%H:%M:%S')} | {elapsed_min:.1f}m] PID {TARGET_PID} running: {alive} | GPU VRAM: {mem}MiB, Util: {util}%", flush=True)

    if not alive and mem < 1500:
        print(f"\n>>> [GPU Released!] Target experiment finished. Cooling down for 10s...", flush=True)
        time.sleep(10)
        print(">>> [Launching Massive Experiment Now!]\n", flush=True)
        cmd = [sys.executable, "-u", "scale_up_massive_model.py", "--epochs", "20"]
        subprocess.run(cmd, check=True)
        print("\n>>> [Massive Experiment Completed Successfully!]", flush=True)
        break

    time.sleep(POLL_SEC)
