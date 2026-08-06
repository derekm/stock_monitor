import time, json, psutil, subprocess
LOG = "/tmp/util_monitor.csv"
with open(LOG, "w") as f:
    f.write("t, gpu_util, gpu_mem_mb, cpu_total, " + ",".join(f"cpu{i}" for i in range(psutil.cpu_count())) + "\n")
print(f"monitoring {psutil.cpu_count()} cores -> {LOG}", flush=True)
t0 = time.time()
while time.time() - t0 < 36000:  # 10h cap to cover the full training run
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    ).stdout.strip().split("\n")[0].split(",")
    gpu_util, gpu_mem = gpu[0].strip(), gpu[1].strip()
    cpu = psutil.cpu_percent(interval=1, percpu=True)
    cpu_total = sum(cpu) / len(cpu)
    row = f"{time.time()-t0:.0f}, {gpu_util}, {gpu_mem}, {cpu_total:.1f}, " + ",".join(f"{c:.1f}" for c in cpu)
    with open(LOG, "a") as f:
        f.write(row + "\n")
