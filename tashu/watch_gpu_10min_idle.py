"""GPU 10-Minute Continuous Idle Watchdog:
Monitors NVIDIA GPU every 15 seconds.
When GPU VRAM usage drops below 500 MiB and remains continuously idle for 10 full minutes (600 seconds),
it automatically launches the 50.2M parameter model training script (train_50m_mse_protocol.py).
"""

import os
import subprocess
import sys
import time

TARGET_PID = 3096
IDLE_THRESHOLD_MIB = 500
REQUIRED_IDLE_SECONDS = 600  # Strictly 10 minutes
POLL_INTERVAL_SEC = 15

print("=" * 88, flush=True)
print(" [GPU 10분 연속 유휴 감시(Watchdog) 서비스 시작]", flush=True)
print(f" 감시 대상: PID {TARGET_PID} 종료 및 GPU VRAM < {IDLE_THRESHOLD_MIB} MiB", flush=True)
print(f" 기동 조건: GPU 유휴 상태가 '연속 {REQUIRED_IDLE_SECONDS // 60}분' 유지될 때", flush=True)
print(" 기동 작업: .venv\\Scripts\\python.exe -u train_50m_mse_protocol.py", flush=True)
print("=" * 88, flush=True)


def is_pid_alive(pid):
    try:
        res = subprocess.check_output(
            ["powershell", "-Command", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id"],
            encoding='utf-8'
        ).strip()
        return len(res) > 0
    except Exception:
        return False


def get_gpu_status():
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,nounits,noheader"],
            encoding='utf-8'
        ).strip().split(',')
        return int(res[0].strip()), int(res[1].strip())
    except Exception:
        return 9999, 100


idle_seconds = 0
start_time = time.time()

while True:
    pid_running = is_pid_alive(TARGET_PID)
    mem_used, gpu_util = get_gpu_status()
    now_str = time.strftime('%H:%M:%S')

    # Condition: PID is dead and VRAM is below threshold
    if not pid_running and mem_used < IDLE_THRESHOLD_MIB:
        idle_seconds += POLL_INTERVAL_SEC
        idle_min = idle_seconds / 60.0
        req_min = REQUIRED_IDLE_SECONDS / 60.0
        print(
            f"[{now_str}] [GPU 유휴 감지] VRAM: {mem_used}MiB, Util: {gpu_util}% | "
            f"연속 유휴 시간: {idle_min:.1f}분 / {req_min:.0f}분 필요 ({idle_seconds}/{REQUIRED_IDLE_SECONDS}s)",
            flush=True
        )

        if idle_seconds >= REQUIRED_IDLE_SECONDS:
            print("\n" + "=" * 88, flush=True)
            print(f">>> [기동 조건 충족] GPU가 10분 이상 연속 유휴 상태를 유지했습니다!", flush=True)
            print(">>> [초대형 50.2M STGNN & MSE 표준 프로토콜 학습을 시작합니다!]", flush=True)
            print("=" * 88 + "\n", flush=True)

            cmd = [sys.executable, "-u", "train_50m_mse_protocol.py"]
            ret = subprocess.run(cmd)
            print(f"\n>>> [학습 및 평가 완료] 반환 코드: {ret.returncode}", flush=True)
            break
    else:
        # GPU is busy
        if idle_seconds > 0:
            print(f"[{now_str}] [주의] GPU 사용 감지됨! 유휴 타이머 초기화 (이전 유휴: {idle_seconds / 60.0:.1f}분)", flush=True)
            idle_seconds = 0

        status_desc = f"PID {TARGET_PID} 실행 중" if pid_running else "외부 프로세스 사용 중"
        print(f"[{now_str}] [GPU 사용 중] 상태: {status_desc} | VRAM: {mem_used}MiB, Util: {gpu_util}%", flush=True)

    time.sleep(POLL_INTERVAL_SEC)
