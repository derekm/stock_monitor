import time, glob, subprocess, csv, psutil, os

PID = 45236
p = psutil.Process(PID)
N = psutil.cpu_count()

def gpu():
    try:
        g = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu',
                            '--format=csv,noheader'], capture_output=True, text=True)
        return int(g.stdout.strip().split('\n')[0].replace(' %', ''))
    except Exception:
        return -1

def tickers():
    return len(glob.glob('granite_ckpts/per_ticker/*/'))

rows = []
t0 = time.time()
print("t,elapsed,tickers,gpu,threads,maxcore,core1,sys", flush=True)
p.cpu_percent()  # baseline
for i in range(150):  # 150s
    cores = psutil.cpu_percent(percpu=True)
    mx = max(cores)
    g = gpu()
    tk = tickers()
    th = p.num_threads()
    sys = psutil.cpu_percent()
    rows.append((i, round(time.time()-t0, 1), tk, g, th, round(mx, 1),
                 round(cores[0], 1), round(sys, 1)))
    print(f"{i},{round(time.time()-t0,1)},{tk},{g},{th},{round(mx,1)},{round(cores[0],1)},{round(sys,1)}", flush=True)
    time.sleep(1)

# analysis: find long single-core gaps (gpu<10 for >=2 consecutive samples while proc has 1 thread)
print("\n--- ANALYSIS ---", flush=True)
low = [r for r in rows if r[3] < 10]
print(f"GPU<10% samples: {len(low)}/{len(rows)}", flush=True)
# group consecutive low-gpu runs
runs = []
cur = []
for r in rows:
    if r[3] < 10:
        cur.append(r)
    else:
        if cur:
            runs.append(cur); cur = []
if cur:
    runs.append(cur)
print(f"Low-GPU gap events (>=2s): {sum(1 for r in runs if len(r)>=2)}", flush=True)
for r in runs:
    if len(r) >= 2:
        tk0, tk1 = r[0][2], r[-1][2]
        print(f"  gap {len(r)}s: t={r[0][1]}->{r[-1][1]}s  tickers {tk0}->{tk1}  maxcore={max(x[5] for x in r)}", flush=True)
