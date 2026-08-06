import time, subprocess, psutil

PID = 12324
p = psutil.Process(PID)
N = psutil.cpu_count()

def gpu():
    try:
        g = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu',
                            '--format=csv,noheader'], capture_output=True, text=True)
        return int(g.stdout.strip().split('\n')[0].replace(' %', ''))
    except Exception:
        return -1

def per_core():
    # per-core CPU% snapshot
    return list(psutil.cpu_percent(interval=None, percpu=True))

print("t,elapsed,gpu,threads,maxcore,cores_str", flush=True)
t0 = time.time()
p.cpu_percent()  # reset
rows = []
for i in range(480):  # 8 min @ 1s
    g = gpu()
    th = p.num_threads()
    # need a tiny interval to get per-core; use 0.0 then sample
    cores = psutil.cpu_percent(interval=0.0, percpu=True)
    maxc = max(cores)
    rows.append((i, round(time.time()-t0,1), g, th, maxc))
    if i % 3 == 0:  # print every 3s to keep log readable
        print(f"{i},{round(time.time()-t0,1)},{g},{th},{maxc:.0f},{'/'.join(f'{c:.0f}' for c in cores)}", flush=True)
    time.sleep(1)

# analysis
low_gpu = [r for r in rows if r[2] < 20]
single_core = [r for r in rows if r[4] > 50 and (sum(1 for c in psutil.cpu_percent(interval=0,percpu=True) if c>30) <= 1)]
print(f"\n=== ANALYSIS (8 min, {len(rows)} samples) ===", flush=True)
print(f"GPU<20%: {len(low_gpu)}/{len(rows)} ({100*len(low_gpu)/len(rows):.0f}%)", flush=True)
print(f"GPU mean: {sum(r[2] for r in rows)/len(rows):.0f}%  max: {max(r[2] for r in rows)}%", flush=True)
print(f"max single-core spikes (>50% on one core): {len(single_core)} samples", flush=True)
print(f"thread count: started {rows[0][3]}, ended {rows[-1][3]}", flush=True)
