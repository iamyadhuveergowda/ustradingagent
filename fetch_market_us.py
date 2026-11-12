#!/usr/bin/env python3
"""
fetch_market_us.py — writes market_us.json: US index levels for the dashboard strip.

Run this on a schedule (e.g. a GitHub Action every 15 min during US hours), NOT
from build_dashboard.py. The generator stays offline; this script is the only
thing in the repo that touches the network.

    python3 fetch_market_us.py                 # writes market_us.json in the repo root
    python3 fetch_market_us.py --out other.json

Source: Yahoo Finance's public chart endpoint (unofficial, no API key). It is
rate-limited and can change without notice, so every failure path keeps the
last good value and marks it stale instead of dropping the tile.

Change is computed against the previous SESSION's close: the last daily bar
dated before the quote's own Eastern date. (Yahoo's `chartPreviousClose` is the
close before the 5-day window, not yesterday's close, so it is not used.)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

INDICES = [
    # (display name, Yahoo symbol)
    ("S&P 500", "^GSPC"),
    ("Dow Jones", "^DJI"),
    ("Nasdaq Composite", "^IXIC"),
    ("Russell 2000", "^RUT"),
    ("CBOE VIX", "^VIX"),
]
ET = ZoneInfo("America/New_York")
UA = "Mozilla/5.0 (compatible; ustradingagent-dashboard/1.0)"
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=5d&interval=1d"


def fetch_chart(symbol: str, retries: int = 3, timeout: int = 20) -> dict:
    """Return the `result[0]` block for one symbol, retrying on 429/5xx."""
    from urllib.parse import quote

    url = CHART_URL.format(sym=quote(symbol, safe=""))
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.load(resp)
            result = (payload.get("chart") or {}).get("result") or []
            if not result:
                raise ValueError("empty result")
            return result[0]
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError, TimeoutError) as e:
            last_err = e
            retriable = isinstance(e, urllib.error.HTTPError) and e.code in (429, 500, 502, 503)
            if not retriable and not isinstance(e, (urllib.error.URLError, TimeoutError)):
                break
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{symbol}: {last_err}")


def build_entry(name: str, symbol: str, block: dict) -> dict:
    """Turn one Yahoo chart block into a dashboard entry. Pure function, unit-testable."""
    meta = block["meta"]
    quote_price = meta.get("regularMarketPrice")
    quote_ts = meta.get("regularMarketTime")
    if quote_price is None or quote_ts is None:
        raise ValueError(f"{symbol}: no regularMarketPrice/Time")

    quote_dt = datetime.fromtimestamp(quote_ts, timezone.utc)
    quote_date_et = quote_dt.astimezone(ET).date()

    quote = block["indicators"]["quote"][0]
    bars = []
    for ts, close in zip(block.get("timestamp") or [], quote.get("close") or []):
        if close is None:
            continue
        bars.append((datetime.fromtimestamp(ts, timezone.utc).astimezone(ET).date(), float(close)))

    prior = [c for d, c in bars if d < quote_date_et]
    prev_close = prior[-1] if prior else None
    change = (quote_price - prev_close) if prev_close else None
    change_pct = (100.0 * change / prev_close) if prev_close else None

    regular = (meta.get("currentTradingPeriod") or {}).get("regular") or {}
    now = time.time()
    is_open = bool(regular.get("start") and regular.get("end") and regular["start"] <= now <= regular["end"])

    return {
        "name": name,
        "symbol": symbol,
        "price": round(float(quote_price), 2),
        "prev_close": round(prev_close, 2) if prev_close is not None else None,
        "change": round(change, 2) if change is not None else None,
        "change_pct": round(change_pct, 3) if change_pct is not None else None,
        "quote_time_utc": quote_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "quote_time_et": quote_dt.astimezone(ET).strftime("%H:%M"),
        "quote_date_et": quote_date_et.isoformat(),
        "session_open": is_open,
        "spark": [round(c, 2) for _, c in bars[-5:]],
        "stale": False,
    }


def load_previous(path: Path) -> dict:
    try:
        return {i["symbol"]: i for i in json.loads(path.read_text()).get("indices", [])}
    except (OSError, ValueError, AttributeError):
        return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "market_us.json"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    previous = load_previous(out)

    entries, failures = [], []
    for name, symbol in INDICES:
        try:
            entries.append(build_entry(name, symbol, fetch_chart(symbol)))
        except Exception as e:  # noqa: BLE001 — keep going, carry the last good value
            failures.append(str(e))
            if symbol in previous:
                kept = dict(previous[symbol], stale=True)
                entries.append(kept)
                print(f"  ! {symbol}: {e} — kept last value ({kept.get('quote_time_utc')})", file=sys.stderr)
            else:
                print(f"  ! {symbol}: {e} — no previous value, tile will be omitted", file=sys.stderr)

    if not entries:
        print("no index data fetched; leaving market_us.json untouched", file=sys.stderr)
        return 1

    doc = {
        "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "Yahoo Finance chart API (unofficial)",
        "indices": entries,
    }
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {out} with {len(entries)} indices ({len(failures)} failed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
