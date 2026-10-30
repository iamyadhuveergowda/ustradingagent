#!/usr/bin/env python3
"""
End-of-day summary for the US intraday agent — deterministic first.

Two books are reported:
  1. AGENT BOOK — statistics computed IN CODE from
     trades/daily_trade_ledger_us.json (never summarised from a truncated
     snippet by an LLM; that was the NIFTY EOD's core defect).
  2. DESK BOOK — every advisory verdict in desk_history.json is scored
     against what price actually did AFTER the signal time, using Alpaca
     1-minute IEX bars: first hit of target vs stop wins; a bar touching
     both is scored AMBIGUOUS (excluded from accuracy, reported as a
     count) instead of being auto-scored a loss. Scores are written back
     into desk_history.json so verdicts accumulate a track record.

The LLM (Gemini) only narrates the computed numbers. The prompt forbids
inventing figures and forbids rule-change prescriptions — strategy
changes are a monthly human decision, not a daily LLM output.

Env (GitHub Actions secrets):
  GEMINI_API_KEY / GEMINI_MODEL   optional narration (stats-only if unset)
  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID   optional delivery
  ALPACA_API_KEY / ALPACA_SECRET_KEY      paper keys, bars for scoring

Usage: python3 eod_summary_us.py [--date YYYY-MM-DD] [--no-send]
Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

SCORE_WINDOW_DAYS = 10  # older unscored records are closed as NO_DATA (mirrors the NIFTY EODs)
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-3.8-flash"

def gemini_post_with_retry(req, timeout=90, attempts=3):
    """POST to Gemini with retries on timeouts / 5xx / 429 only (429 uses 30s/60s backoff, mirroring the desk scripts; other 4xx fail fast).

    A 4xx (e.g. 404 bad model) fails fast -- retrying cannot fix it.
    A timeout, like the Oct 24 mock run, is retried: 90s per attempt,
    10s then 20s backoff, max ~5 min total. Raises the last error.
    """
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                last_exc = exc
            elif 400 <= exc.code < 500:
                raise
            else:
                last_exc = exc
        except Exception as exc:
            last_exc = exc
        if attempt < attempts:
            print(f"Gemini attempt {attempt} failed ({last_exc}); retrying...",
                  file=sys.stderr)
            if isinstance(last_exc, urllib.error.HTTPError) and last_exc.code == 429:
                time.sleep(30 * attempt)
            else:
                time.sleep(10 * attempt)
    raise last_exc

WIN, LOSS, EXPIRED = "WIN 🟢", "LOSS 🔴", "EXPIRED ⚪"


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


def load_json_strict(path: Path, default):
    """Like load_json, but a present-but-unparseable file is an error.

    A missing file still returns the default. A present file that is not
    valid JSON must never be silently replaced by an empty list — that
    would destroy the permanent desk history. Raises RuntimeError; the
    caller lets it abort the run before anything is written or sent.
    """
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"{path} exists but is not valid JSON ({exc}); refusing to overwrite it") from exc


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Agent book statistics (pure functions of the ledger)
# ---------------------------------------------------------------------------
def trade_r(t: dict) -> float | None:
    try:
        risk = abs(float(t["entry"]) - float(t["initial_sl"])) * int(t["qty"])
        if risk <= 0:
            return None
        return float(t["net_pnl"]) / risk
    except Exception:
        return None


def exit_reason(t: dict) -> str:
    exits = [e.get("text", "") for e in t.get("events", []) if e.get("type") == "EXIT"]
    if not exits:
        return "unknown"
    return exits[-1].split(":")[0].strip() or "unknown"


def book_stats(trades: list[dict]) -> dict:
    settled = [t for t in trades if t.get("status") in (WIN, LOSS, EXPIRED)]
    wins = [t for t in settled if float(t.get("net_pnl", 0)) > 0]
    losses = [t for t in settled if float(t.get("net_pnl", 0)) <= 0]
    rs = [r for r in (trade_r(t) for t in settled) if r is not None]
    by_side, reasons = {}, {}
    for side in ("BUY", "SELL"):
        s = [t for t in settled if t.get("action") == side]
        if s:
            by_side[side] = {
                "n": len(s),
                "wins": sum(1 for t in s if float(t.get("net_pnl", 0)) > 0),
                "net": round(sum(float(t.get("net_pnl", 0)) for t in s), 2),
            }
    for t in settled:
        reasons[exit_reason(t)] = reasons.get(exit_reason(t), 0) + 1
    return {
        "settled": len(settled),
        "wins": len(wins), "losses": len(losses),
        "win_rate_pct": round(100.0 * len(wins) / len(settled), 1) if settled else None,
        "net_usd": round(sum(float(t.get("net_pnl", 0)) for t in settled), 2),
        "avg_win_usd": round(statistics.mean(float(t["net_pnl"]) for t in wins), 2) if wins else None,
        "avg_loss_usd": round(statistics.mean(float(t["net_pnl"]) for t in losses), 2) if losses else None,
        "mean_r": round(statistics.mean(rs), 2) if rs else None,
        "median_r": round(statistics.median(rs), 2) if rs else None,
        "by_side": by_side,
        "exit_reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Desk outcome scoring against post-signal bars
# ---------------------------------------------------------------------------
def fetch_bars(ticker: str, start: datetime, end: datetime) -> list[dict]:
    """Alpaca 1-minute bars (IEX feed). Returns [] on any failure."""
    key = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        print("ERROR: ALPACA_API_KEY/ALPACA_SECRET_KEY not set; cannot score bars", file=sys.stderr)
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



def _bars_from_signal(bars: list[dict], sig_et) -> list[dict]:
    """Keep only bars at/after the signal time (client-side slice,
    mirroring the NIFTY slice_from_signal semantics). Bars with a
    missing/unparseable timestamp are kept, never silently dropped."""
    out = []
    for b in bars:
        t = b.get("t")
        if t is None:
            out.append(b)
            continue
        try:
            bt = datetime.fromisoformat(str(t).replace("Z", "+00:00")).astimezone(ET)
        except Exception:
            out.append(b)
            continue
        if bt >= sig_et:
            out.append(b)
    return out


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


# ---------------------------------------------------------------------------
# Narration + delivery
# ---------------------------------------------------------------------------
def narrate(stats: dict) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return ""
    prompt = (
        "You are writing the end-of-day note for a US intraday paper-trading desk. "
        "Below are statistics computed in code — they are the ONLY facts you may use. "
        "Write 3-5 short bullet observations in plain language. Rules: use only the "
        "numbers given; never invent a figure; never prescribe a rule change or a "
        "parameter value (observations only — strategy changes are a monthly human "
        "decision); if the sample is tiny, say so plainly.\n\n"
        f"STATS: {json.dumps(stats)}"
    )
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3}}
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST")
    try:
        data = json.loads(gemini_post_with_retry(req))
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as exc:
        print(f"Narration failed: {exc}", file=sys.stderr)
        return ""



def _split_chunks(text, limit=3900):
    chunks = []
    cur = ''
    for line in text.splitlines():
        if cur and len(cur) + len(line) + 1 > limit:
            chunks.append(cur)
            cur = ''
        while len(line) > limit:
            chunks.append(line[:limit])
            line = line[limit:]
        cur = cur + line + chr(10) if cur else line + chr(10)
    if cur:
        chunks.append(cur)
    return chunks


def send_telegram(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        print("Telegram not configured; message was:\n" + text)
        return
    for i, chunk in enumerate(_split_chunks(text), start=1):
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=json.dumps({"chat_id": chat, "text": chunk}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                if not (200 <= resp.status < 300):
                    print(f"ERROR: Telegram send failed: HTTP {resp.status} (chunk {i}) — message NOT delivered via Telegram", file=sys.stderr)
        except Exception as exc:
            print(f"ERROR: Telegram send failed: {exc} (chunk {i}) — message NOT delivered via Telegram", file=sys.stderr)


def fmt_book(label: str, s: dict) -> str:
    if not s["settled"]:
        return f"{label}: no settled trades"
    sides = " · ".join(
        f"{k} {v['n']} trades, {v['wins']}W, ${v['net']:+,.2f}" for k, v in s["by_side"].items())
    reasons = ", ".join(f"{k} ×{v}" for k, v in sorted(s["exit_reasons"].items()))
    aw = s['avg_win_usd'] if s['avg_win_usd'] is not None else "—"
    al = s['avg_loss_usd'] if s['avg_loss_usd'] is not None else "—"
    return (f"{label}: {s['settled']} settled · {s['wins']}W/{s['losses']}L "
            f"({s['win_rate_pct']}%) · net ${s['net_usd']:+,.2f}\n"
            f"  avg win ${aw} / avg loss ${al} · "
            f"mean R {s['mean_r']} · median R {s['median_r']}\n"
            f"  {sides}\n  exits: {reasons}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="ET date to report (default: today ET)")
    ap.add_argument("--no-send", action="store_true")
    args = ap.parse_args()

    root = repo_root()
    ledger = load_json(root / "trades" / "daily_trade_ledger_us.json", [])
    history_path = root / "desk_history.json"
    history = load_json_strict(history_path, [])
    if not isinstance(ledger, list):
        ledger = []
    if not isinstance(history, list):
        raise RuntimeError(f"{history_path} is not a JSON list; refusing to overwrite it")

    now_et = datetime.now(ET)
    report_date = args.date or now_et.date().isoformat()

    # ---- Score any unscored desk verdicts whose session has ended ---------
    changed = False
    # R2: session_over is computed from the current time only, so a
    # manual --date backfill for a past date is not gated on the
    # report date; scheduled runs (report_date == today) behave
    # exactly as before.
    session_over = now_et.time() >= dtime(16, 5)
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
        if sig_et.date().isoformat() == now_et.date().isoformat() and not session_over:
            continue
        close_et = datetime.combine(sig_et.date(), dtime(16, 0), tzinfo=ET)
        bars = _bars_from_signal(fetch_bars(rec["ticker"], sig_et, close_et), sig_et)
        try:
            outcome, outcome_r = score_record(rec, bars)
        except Exception as exc:
            print(f"Scoring failed for {rec.get('ticker')} {rec.get('signal_time')}: {exc}; marking NO_DATA", file=sys.stderr)
            rec["outcome"], rec["outcome_r"] = "NO_DATA", None
            rec["outcome_at"] = now_et.isoformat(timespec="seconds")
            changed = True
            continue
        age_days = (now_et.date() - sig_et.date()).days
        if outcome == "PENDING" and age_days > SCORE_WINDOW_DAYS:
            outcome, outcome_r = "NO_DATA", None
        if outcome != "PENDING":
            rec["outcome"], rec["outcome_r"] = outcome, outcome_r
            rec["outcome_at"] = now_et.isoformat(timespec="seconds")
            changed = True
    if changed:
        save_json(history_path, history)

    # ---- Deterministic statistics -----------------------------------------
    today_trades = [t for t in ledger if t.get("date") == report_date]
    stats = {
        "date": report_date,
        "agent_today": book_stats(today_trades),
        "agent_cumulative": book_stats(ledger),
        "desk_today": desk_stats([r for r in history
                                  if str(r.get("signal_time", ""))[:10] == report_date]),
        "desk_cumulative": desk_stats(history),
    }

    lines = [f"US EOD — {report_date} (ET)"]
    lines.append(fmt_book("Agent today", stats["agent_today"]))
    lines.append(fmt_book("Agent cumulative", stats["agent_cumulative"]))
    for label, key in (("Desk today", "desk_today"), ("Desk cumulative", "desk_cumulative")):
        d = stats[key]
        if not d["judged"]:
            lines.append(f"{label}: no verdicts")
            continue
        parts = [f"judged {d['judged']}, scored {d['scored']}, ambiguous {d['ambiguous']}"]
        if d.get("GO"):
            parts.append(f"GO: {d['GO']['n']} verdicts, {d['GO']['positive_r_pct']}% ended positive-R, "
                         f"avg {d['GO']['mean_r_if_taken']}R if taken")
        if d.get("SKIP"):
            parts.append(f"SKIP: {d['SKIP']['n']} verdicts, {d['SKIP']['would_have_lost_pct']}% would have lost, "
                         f"avg {d['SKIP']['mean_r_if_taken']}R if taken")
        lines.append(f"{label}: " + " · ".join(parts))
    lines.append("Desk is advisory only; scores use theoretical R from signal time.")

    note = narrate(stats)
    if note:
        lines.append("\n— Desk notes —")
        lines.append(note)

    message = "\n".join(lines)
    print(message)
    if not args.no_send:
        send_telegram(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
