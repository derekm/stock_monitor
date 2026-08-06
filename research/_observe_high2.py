import time, subprocess, psutil

PID = 29284
p = psutil.Process(PID)
N = psutil.cpu_count()

def gpu():
    try:
        g = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu',
                            '--format=csv,noheader'], capture_output=True, text=True)
        return int(g.stdout.strip().split('\n')[0].replace(' %', ''))
    except Exception:
        return -1

print("t,elapsed,gpu,threads,maxcore,cores_str", flush=True)
t0 = time.time()
p.cpu_percent()
rows = []
for i in range(480):
    g = gpu()
    th = p.num_threads()
    cores = psutil.cpu_percent(interval=0.0, percpu=True)
    maxc = max(cores)
    rows.append((g, th, maxc, cores))
    if i % 3 == 0:
        print(f"{i},{round(time.time()-t0,1)},{g},{th},{maxc:.0f},{'/'.join(f'{c:.0f}' for c in cores)}", flush=True)
    time.sleep(1)

low_gpu = [r for r in rows if r[0] < 20]
print(f"\n=== ANALYSIS (8 min, {len(rows)} samples) ===", flush=True)
print(f"GPU<20%: {len(low_gpu)}/{len(rows)} ({100*len(low_gpu)/len(rows):.0f}%)", flush=True)
print(f"GPU mean: {sum(r[0] for r in rows)/len(rows):.0f}%  max: {max(r[0] for r in rows)}%  min:{min(r[0] for r in rows)}%", flush=True)
# single-core boundedness: maxcore>40 while rest <15
sc = [r for r in rows if r[2] > 40 and sum(1 for c in r[3] if c > 20) <= 1]
print(f"single-core-bound samples (>40% one core, others idle): {len(sc)}/{len(rows)}", flush=True)
print(f"thread count: {rows[0][1]} -> {rows[-1][1]}", flush=True)
