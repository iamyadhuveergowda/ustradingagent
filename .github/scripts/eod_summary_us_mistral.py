#!/usr/bin/env python3
"""
Mistral EOD scoring (US) — SCORING ONLY, no Telegram, no LLM.

Scores every unscored verdict in desk_history_mistral.json against
post-signal Alpaca 1-minute IEX bars, using the EXACT same deterministic
first-hit logic as the Gemini desk's eod_summary_us.py (first hit of
target vs stop wins; a bar touching both is AMBIGUOUS, never an
automatic loss). The scoring functions below are copied verbatim from
eod_summary_us.py so both desks are graded by identical rules — only
the input file and the delivery differ.

Deliberately NOT included: the agent book (the Gemini EOD already
reports it), LLM narration, and any Telegram message. This job's only
output is the scored history file, which the daily desk comparison reads.

Env: ALPACA_API_KEY / ALPACA_SECRET_KEY (paper keys, bars for scoring).
Usage: python3 eod_summary_us_mistral.py [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import urllib.parse
import urllib.request
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def repo_root() -> Path:
    override = os.getenv("DESK_REPO_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2]


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch_bars(ticker: str, start: datetime, end: datetime) -> list[dict]:
    """Alpaca 1-minute bars (IEX feed). Returns [] on any failure."""
    key = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        return []
    bars: list[dict] = []
    token = None
    try:
        for _ in range(4):  # a session is 390 minutes; pages of 10k — 1 is plenty
            params = {
                "timeframe": "1Min", "feed": "iex", "adjustment": "split", "limit": 10000,
                "start": start.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": end.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            if token:
                params["page_token"] = token
            url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers={
                "APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            bars.extend(data.get("bars", []) or [])
            token = data.get("next_page_token")
            if not token:
                break
    except Exception as exc:
        print(f"Bars fetch failed for {ticker}: {exc}", file=sys.stderr)
        return []
    return bars


def score_record(rec: dict, bars: list[dict]) -> tuple[str, float | None]:
    """First-hit scoring from the signal time to the 16:00 ET close.
    Returns (outcome, outcome_r). Theoretical R (no slippage model)."""
    entry, stop, tgt = float(rec["entry"]), float(rec["stop"]), float(rec["target"])
    risk = abs(entry - stop)
    if risk <= 0 or not bars:
        return "PENDING", None
    buy = rec["action"] == "BUY"
    reward_r = abs(tgt - entry) / risk
    for b in bars:
        hi, lo = float(b["h"]), float(b["l"])
        hit_t = (hi >= tgt) if buy else (lo <= tgt)
        hit_s = (lo <= stop) if buy else (hi >= stop)
        if hit_t and hit_s:
            return "AMBIGUOUS", None
        if hit_t:
            return "TARGET", round(reward_r, 2)
        if hit_s:
            return "STOP", -1.0
    last_close = float(bars[-1]["c"])
    r = (last_close - entry) / risk if buy else (entry - last_close) / risk
    return "CLOSE", round(r, 2)


def desk_stats(records: list[dict]) -> dict:
    scored = [r for r in records if r.get("outcome") in ("TARGET", "STOP", "CLOSE")]
    amb = sum(1 for r in records if r.get("outcome") == "AMBIGUOUS")
    out = {"judged": len(records), "scored": len(scored), "ambiguous": amb}
    for verdict in ("GO", "SKIP"):
        s = [r for r in scored if r.get("verdict") == verdict]
        if not s:
            out[verdict] = None
            continue
        rs = [float(r["outcome_r"]) for r in s if r.get("outcome_r") is not None]
        if verdict == "GO":
            good = sum(1 for r in rs if r > 0)
            out[verdict] = {"n": len(s), "positive_r_pct": round(100.0 * good / len(s), 1),
                            "mean_r_if_taken": round(statistics.mean(rs), 2) if rs else None}
        else:
            avoided = sum(1 for r in s if r.get("outcome") == "STOP" or (r.get("outcome_r") is not None and float(r["outcome_r"]) < 0))
            out[verdict] = {"n": len(s), "would_have_lost_pct": round(100.0 * avoided / len(s), 1),
                            "mean_r_if_taken": round(statistics.mean(rs), 2) if rs else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="ET date to score (default: today ET)")
    args = ap.parse_args()

    root = repo_root()
    history_path = root / "desk_history_mistral.json"
    history = load_json(history_path, [])
    if not isinstance(history, list):
        history = []

    now_et = datetime.now(ET)
    report_date = args.date or now_et.date().isoformat()

    changed = False
    session_over = now_et.time() >= dtime(16, 5) or now_et.date().isoformat() > report_date
    for rec in history:
        if rec.get("outcome") is not None:
            continue
        try:
            sig = datetime.fromisoformat(str(rec["signal_time"]))
        except Exception:
            continue
        sig_et = sig.astimezone(ET)
        if sig_et.date().isoformat() > report_date:
            continue
        if sig_et.date().isoformat() == report_date and not session_over:
            continue
        close_et = datetime.combine(sig_et.date(), dtime(16, 0), tzinfo=ET)
        bars = fetch_bars(rec["ticker"], sig_et, close_et)
        outcome, outcome_r = score_record(rec, bars)
        if outcome != "PENDING":
            rec["outcome"], rec["outcome_r"] = outcome, outcome_r
            rec["outcome_at"] = now_et.isoformat(timespec="seconds")
            changed = True
    if changed:
        save_json(history_path, history)

    today = [r for r in history if str(r.get("signal_time", ""))[:10] == report_date]
    for label, recs in (("Mistral desk today", today), ("Mistral desk cumulative", history)):
        d = desk_stats(recs)
        print(f"{label}: judged {d['judged']}, scored {d['scored']}, ambiguous {d['ambiguous']}")
        if d.get("GO"):
            print(f"  GO: {d['GO']['n']} verdicts, {d['GO']['positive_r_pct']}% ended positive-R, "
                  f"avg {d['GO']['mean_r_if_taken']}R if taken")
        if d.get("SKIP"):
            print(f"  SKIP: {d['SKIP']['n']} verdicts, {d['SKIP']['would_have_lost_pct']}% would have lost, "
                  f"avg {d['SKIP']['mean_r_if_taken']}R if taken")
    print(f"Scored history written: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
