# black_litterman.py

Black-Litterman posterior expected returns and long-only weights.

1. Equilibrium \(\pi = \delta \Sigma w_{mkt}\) (reverse optimization)
2. Views \(P\mu = Q + \varepsilon\), \(\Omega\) from He–Litterman
3. Posterior \(\mu_{BL}\) blends prior and views
4. Mean-variance weights with \(\mu_{BL}\) (long-only)

Uses **Ledoit-Wolf** covariance by default.

```bash
python black_litterman.py --universe portfolio --save
python black_litterman.py --universe portfolio \
  --view SMCI:-0.05 --view PFE:0.08 --view KHC:0.07 --save
```

Output: `black_litterman_weights.csv`
