import psutil, time, subprocess

MARKERS = ['_stride_test', '_cuda_graph', '_cuda_bench', '_ptsteps', '_observe']
procs = list(psutil.process_iter(['pid', 'ppid', 'name', 'cmdline']))

def is_mine(p):
    return p.info['name'] == 'python.exe' and any(
        m in ' '.join(p.info['cmdline'] or []) for m in MARKERS)

to_kill = set()
for p in procs:
    if is_mine(p):
        to_kill.add(p.info['pid'])
        try:
            for ch in psutil.Process(p.info['pid']).children(recursive=True):
                to_kill.add(ch.pid)
        except Exception:
            pass
# also catch orphans whose parent is already marked
for p in procs:
    if p.info['pid'] not in to_kill and p.info['ppid'] in to_kill:
        to_kill.add(p.info['pid'])

print("killing tree pids:", sorted(to_kill), flush=True)
for pid in to_kill:
    try:
        psutil.Process(pid).kill()  # SIGKILL -> frees CUDA context
    except Exception as e:
        print("  err", pid, e, flush=True)

time.sleep(4)
apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                      capture_output=True, text=True).stdout.strip()
util = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader'],
                      capture_output=True, text=True).stdout.strip()
left = [(p.info['pid'], ' '.join(p.info['cmdline'] or [])[:50])
        for p in psutil.process_iter(['pid', 'name', 'cmdline']) if is_mine(p)]
print(f"GPU compute apps after kill: {apps!r}", flush=True)
print(f"GPU util: {util}", flush=True)
print(f"my scripts still alive: {left}", flush=True)
