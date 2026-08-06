import time, glob, subprocess, psutil

PID = 45880
p = psutil.Process(PID)
N = psutil.cpu_count()

def gpu():
    try:
        g = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu',
                            '--format=csv,noheader'], capture_output=True, text=True)
        return int(g.stdout.strip().split('\n')[0].replace(' %', ''))
    except Exception:
        return -1

def gpu_procs():
    try:
        r = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                            '--format=csv,noheader'], capture_output=True, text=True)
        return r.stdout.strip().replace('\r', '')
    except Exception:
        return ''

def tickers():
    return len(glob.glob('granite_ckpts/per_ticker/*/'))

def ollama_cpu():
    try:
        return round(psutil.Process(4128).cpu_percent(interval=0), 1)
    except Exception:
        return -1.0

rows = []
t0 = time.time()
print("t,elapsed,tickers,gpu,threads,ollama_cpu,gpu_proc_pids", flush=True)
p.cpu_percent()
for i in range(180):
    g = gpu()
    tk = tickers()
    th = p.num_threads()
    oc = ollama_cpu()
    procs = gpu_procs().replace('\n', ' | ')
    rows.append((i, round(time.time()-t0,1), tk, g, th))
    print(f"{i},{round(time.time()-t0,1)},{tk},{g},{th},{oc},{procs}", flush=True)
    time.sleep(1)

# gaps: gpu < 15 for >=2 consecutive
low = [r for r in rows if r[3] < 15]
print(f"\nGPU<15% samples: {len(low)}/{len(rows)}", flush=True)
runs = []; cur = []
for r in rows:
    if r[3] < 15:
        cur.append(r)
    else:
        if cur: runs.append(cur); cur=[]
if cur: runs.append(cur)
print(f"Gap events (>=2s): {sum(1 for r in runs if len(r)>=2)}", flush=True)
for r in runs:
    if len(r) >= 2:
        print(f"  gap {len(r)}s, tickers {r[0][2]}->{r[-1][2]}, ollama_then={ollama_cpu()}", flush=True)
