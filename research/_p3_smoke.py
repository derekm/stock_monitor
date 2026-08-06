import pass3_sweep as P, time, json
P.STEPS = 200
results = []
for tk in ["AEP", "NVR"]:
    print(f"\n=== {tk} ===", flush=True)
    for name, ctx, hor, patch, dec, lr, pre, obj, multi, seeds in [
        ("baseline", 512, 96, 64, True, 1e-4, True, "price", False, 1),
        ("returns", 512, 96, 64, True, 1e-4, True, "returns", False, 1),
        ("multi", 512, 96, 64, True, 1e-4, True, "price", True, 1),
        ("scratch", 512, 96, 64, True, 1e-4, False, "price", False, 1),
        ("lr3e-4", 512, 96, 64, True, 3e-4, True, "price", False, 1),
    ]:
        t = time.time()
        wins = P.build_windows(tk, ctx, hor, obj, multi)
        r = P.train_score(wins, ctx, hor, patch, dec, lr, pre, obj, seeds)
        r.update(ticker=tk, exp=name, build_s=round(time.time() - t, 1))
        results.append(r)
        print(f"  {name}: nw={len(wins)} {r}", flush=True)
json.dump(results, open("/tmp/p3_smoke_results.json", "w"), indent=2)
print("\nSMOKE DONE", flush=True)
