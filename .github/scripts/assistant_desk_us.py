#!/usr/bin/env python3
"""
Assistant desk for the US intraday agent — ADVISORY ONLY.

Reads signals/latest.json (published by trading_agent_us_v2 on the Oracle
server and pushed to this repo), judges EVERY candidate with Gemini, and:
  * writes desk_verdict.json          (latest judged batch)
  * appends to desk_history.json      (permanent record the EOD scorer uses)
  * sends one Telegram message per new signal batch

The desk has NO connection to the trading agent: nothing here can place,
block, or modify a trade. It is an advice log whose accuracy the EOD
script measures. (Designed 2026-10-23; the NIFTY tradetracker desk's known
defects — judging only candidates[0], never committing the history file,
and verdict records without prices — are deliberately avoided here.)

Env (GitHub Actions secrets):
  GEMINI_API_KEY     required for verdicts
  GEMINI_MODEL       optional, default "gemini-2.5-flash"
  TELEGRAM_BOT_TOKEN optional (no Telegram if unset)
  TELEGRAM_CHAT_ID   optional

Stdlib only. Exit code 0 in all expected situations (stale signal, market
closed, already judged) so the workflow stays green; the commit step only
pushes when a file actually changed.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
FRESH_MINUTES = float(os.getenv("DESK_FRESH_MINUTES", "30"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def repo_root() -> Path:
    override = os.getenv("DESK_REPO_ROOT")
    if override:
        return Path(override)
    # .github/scripts/<this file> -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def gemini_judge(candidates: list[dict], spy_pct, signal_time_et: str) -> list[dict]:
    """One Gemini call judging all candidates. Returns one verdict dict per
    candidate (ticker-matched). Raises on transport/API failure."""
    api_key = os.environ["GEMINI_API_KEY"]
    features = []
    for c in candidates:
        entry, stop, tgt = float(c["ltp"]), float(c["stop"]), float(c["target"])
        atr = float(c.get("indicators", {}).get("atr") or 0) or 0.0
        risk = abs(entry - stop)
        features.append({
            "ticker": c["ticker"], "action": c["signal"],
            "entry": entry, "stop": stop, "target": tgt,
            "qty": c.get("qty", 0),
            "rsi": c.get("indicators", {}).get("rsi"),
            "atr_15m": atr,
            "vol_ratio": c.get("indicators", {}).get("vol_ratio"),
            "stop_distance_atr": round(risk / atr, 2) if atr > 0 else None,
            "target_distance_atr": round(abs(tgt - entry) / atr, 2) if atr > 0 else None,
            "reward_to_risk": round(abs(tgt - entry) / risk, 2) if risk > 0 else None,
            "scout_reason": c.get("scout_reason", ""),
            "candles_5m_last12": (c.get("candles_5m") or [])[-12:],
        })
    prompt = (
        "You are the senior analyst desk for a US intraday paper-trading system. "
        "This is an ADVISORY review: your verdict is logged and your accuracy is "
        "scored at end of day; you do not place trades.\n\n"
        f"Session context: signal time {signal_time_et} ET (regular session 09:30-16:00 ET). "
        f"SPY day move: {spy_pct if spy_pct is not None else 'unknown'}%. "
        "Candles are regular-session 5-minute bars [time, open, high, low, close, volume]. "
        "Volume comes from the IEX feed only (~2-4% of consolidated tape): treat vol_ratio "
        "as directional evidence, never as exact. There is no news feed and no VWAP in this "
        "payload — judge structure and geometry only, and say so if that limits you.\n\n"
        "For EACH candidate decide GO or SKIP. Weigh: (1) candle structure — is the pullback "
        "orderly with a directional close, or is price extended/exhausted into the entry; "
        "(2) geometry — stop distance in ATR, reward:risk, and whether the target is realistic "
        "for the remaining session; (3) regime — does the trade direction agree with the SPY "
        "day move; (4) timing — entries after 14:30 ET have little room, midday entries need "
        "stronger structure. Be selective: a SKIP costs nothing.\n\n"
        "Return ONLY this JSON: {\"verdicts\": [{\"ticker\": str, \"verdict\": \"GO\"|\"SKIP\", "
        "\"confidence\": int 0-100, \"notes\": {\"technical\": str, \"macro\": str, \"risk\": str}}]} "
        "— one entry per candidate, citing concrete numbers from the payload in the notes.\n\n"
        f"Candidates: {json.dumps(features)}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    parsed = json.loads(text)
    verdicts = parsed.get("verdicts", [])
    by_ticker = {str(v.get("ticker", "")).upper(): v for v in verdicts if isinstance(v, dict)}
    out = []
    for c in candidates:
        v = by_ticker.get(c["ticker"].upper())
        if v is None:
            out.append({"ticker": c["ticker"], "verdict": "SKIP", "confidence": 0,
                        "notes": {"technical": "desk returned no verdict for this ticker",
                                  "macro": "", "risk": "fail-closed default"}})
            continue
        verdict = str(v.get("verdict", "SKIP")).upper()
        if verdict not in ("GO", "SKIP"):
            verdict = "SKIP"
        try:
            conf = max(0, min(100, int(v.get("confidence", 0))))
        except (TypeError, ValueError):
            conf = 0
        notes = v.get("notes") if isinstance(v.get("notes"), dict) else {"technical": str(v.get("notes", ""))}
        out.append({"ticker": c["ticker"], "verdict": verdict, "confidence": conf, "notes": notes})
    return out


def send_telegram(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        print("Telegram not configured; message was:\n" + text)
        return
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=json.dumps({"chat_id": chat, "text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            if resp.status != 200:
                print(f"Telegram send failed: HTTP {resp.status}", file=sys.stderr)
    except Exception as exc:  # fail-soft: never fail the run over an alert
        print(f"Telegram send failed: {exc}", file=sys.stderr)


def main() -> int:
    root = repo_root()
    latest_path = root / "signals" / "latest.json"
    verdict_path = root / "desk_verdict.json"
    history_path = root / "desk_history.json"

    latest = load_json(latest_path, None)
    if not latest or not latest.get("candidates"):
        print("No candidates in signals/latest.json; nothing to judge.")
        return 0

    signal_id = str(latest.get("signal_id", ""))
    try:
        gen = datetime.fromisoformat(str(latest["generated_at"]))
    except Exception:
        print("latest.json has no parseable generated_at; skipping.")
        return 0
    gen_et = gen.astimezone(ET)
    now_et = datetime.now(ET)
    age_min = (now_et - gen_et).total_seconds() / 60.0
    if gen_et.date() != now_et.date() or age_min > FRESH_MINUTES or age_min < -5:
        print(f"Signal {signal_id} is stale (age {age_min:.0f} min); skipping.")
        return 0
    if now_et.weekday() >= 5 or not (dtime(9, 30) <= now_et.time() <= dtime(16, 5)):
        print(f"Outside ET session ({now_et:%H:%M}); skipping.")
        return 0

    history = load_json(history_path, [])
    if not isinstance(history, list):
        history = []
    if any(isinstance(r, dict) and r.get("signal_id") == signal_id for r in history):
        print(f"Signal {signal_id} already judged; skipping.")
        return 0

    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set; cannot judge.", file=sys.stderr)
        return 0

    spy_pct = latest.get("spy_day_pct")
    try:
        verdicts = gemini_judge(latest["candidates"], spy_pct, gen_et.strftime("%H:%M"))
    except Exception as exc:
        print(f"Gemini judging failed: {exc}", file=sys.stderr)
        return 0

    judged_at = now_et.isoformat(timespec="seconds")
    cand_by_ticker = {c["ticker"]: c for c in latest["candidates"]}
    records = []
    for v in verdicts:
        c = cand_by_ticker[v["ticker"]]
        records.append({
            "signal_id": signal_id,
            "signal_time": gen_et.isoformat(timespec="seconds"),
            "judged_at": judged_at,
            "ticker": v["ticker"], "action": c["signal"],
            "entry": float(c["ltp"]), "stop": float(c["stop"]), "target": float(c["target"]),
            "qty": c.get("qty", 0),
            "verdict": v["verdict"], "confidence": v["confidence"], "notes": v["notes"],
            "outcome": None, "outcome_r": None, "outcome_at": None,
        })
    history.extend(records)
    save_json(history_path, history)
    save_json(verdict_path, {
        "signal_id": signal_id, "judged_at": judged_at,
        "spy_day_pct": spy_pct, "advisory_only": True, "verdicts": records,
    })

    spy_txt = f"{spy_pct:+.2f}%" if isinstance(spy_pct, (int, float)) else "n/a"
    lines = [f"US DESK (advisory) — {gen_et:%H:%M} ET · SPY {spy_txt}"]
    for r in records:
        icon = "GO ✅" if r["verdict"] == "GO" else "SKIP ⛔"
        tech = (r["notes"] or {}).get("technical", "")
        lines.append(f"{r['ticker']} {r['action']} @ {r['entry']:.2f} → {icon} ({r['confidence']})")
        if tech:
            lines.append(f"  {tech[:220]}")
    lines.append("Advisory only — the agent does not see this verdict.")
    send_telegram("\n".join(lines))
    print(f"Judged {len(records)} candidate(s) from {signal_id}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
