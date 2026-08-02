# Live forecast dashboard

## Start

```bash
./start_dashboard.sh
# Dashboard http://127.0.0.1:8765/index.html
# API       http://127.0.0.1:5055/health
```

## Forecast studio options

- Target: portfolio / index(es) / custom tickers / sectors
- Horizon, context, days-ago, from-first-trade
- Multivariate on/off
- Peer mode: none | index | correlated | uncorrelated | custom
- Peer index / peer tickers / top N
- Run labels + in-memory multi-shot comparison (overlay prior runs)

## API

`POST /forecast` with JSON body — always **live** computation.

## Tests

```bash
npm i
npx playwright install chromium
./start_dashboard.sh &   # separate terminal
npx playwright test
```
