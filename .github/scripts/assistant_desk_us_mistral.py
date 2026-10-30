#!/usr/bin/env python3
"""
Assistant desk for the US intraday agent — ADVISORY ONLY.

Reads signals/latest.json (published by trading_agent_us_v2 on the Oracle
server and pushed to this repo), judges EVERY candidate with Mistral, and:
  * writes desk_verdict_mistral.json  (latest judged batch)
  * appends to desk_history_mistral.json (permanent record for comparison)

Same script as the Gemini desk, with ONLY the LLM call swapped to Mistral
and separate output files, so both desks judge every batch side by side
without touching each other's records.
  * sends one Telegram message per new signal batch

FEASIBILITY (draft patch, Oct 2026, step 2 alone; 1 and 4 deferred):
a deterministic flag, not a hard reject — INFEASIBLE if target >=5.5 ATR
with <=60 min to the 16:00 ET close, or >=4.5 ATR with <=30 min; TIGHT
for >=5.5 ATR with <=120 min or >=4.5 ATR with <=60 min. Stored, shown
on Telegram, and injected into the prompt. Mistral may still GO, but
must justify it explicitly.

The desk has NO connection to the trading agent: nothing here can place,
block, or modify a trade. It is an advice log whose accuracy the EOD
script measures. (Designed 2026-10-23; the NIFTY tradetracker desk's known
defects — judging only candidates[0], never committing the history file,
and verdict records without prices — are deliberately avoided here.)

Env (GitHub Actions secrets):
  MISTRAL_API_KEY    required for verdicts
  MISTRAL_MODEL      optional, default "ministral-14b-2512"
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
import time
import urllib.request
import urllib.error
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
try:
    FRESH_MINUTES = float(os.getenv("DESK_FRESH_MINUTES", "30"))
except (TypeError, ValueError):
    FRESH_MINUTES = 30.0
DEDUP_WINDOW_MINUTES = 120  # old Muse rule: same ticker+direction within 2h = same setup

def _parse_dt(s):
    try:
        dt = datetime.fromisoformat(str(s))
        return dt
    except Exception:
        return None

def filter_deduped(candidates, history, gen_dt):
    """Return (kept, skipped_tickers) applying the 2h ticker+direction rule.

    A candidate is skipped if history already holds the same ticker and
    action with a signal_time 0-120 min before this batch's signal_time.
    signal_id dedupe (in main) still runs first; this is in addition to it.
    """
    kept, skipped = [], []
    for c in candidates:
        tick = str(c.get("ticker", "")).upper()
        act = str(c.get("signal", ""))
        dup = False
        for r in history:
            if not isinstance(r, dict):
                continue
            if str(r.get("ticker", "")).upper() != tick or str(r.get("action", "")) != act:
                continue
            sig_dt = _parse_dt(r.get("signal_time") or r.get("judged_at"))
            if sig_dt is None:
                continue
            # normalise naive datetimes to the batch timezone
            if sig_dt.tzinfo is None and gen_dt.tzinfo is not None:
                sig_dt = sig_dt.replace(tzinfo=gen_dt.tzinfo)
            delta_min = (gen_dt - sig_dt).total_seconds() / 60.0
            if 0 <= delta_min <= DEDUP_WINDOW_MINUTES:
                dup = True
                break
        if dup:
            skipped.append(tick)
        else:
            kept.append(c)
    return kept, skipped
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL") or "ministral-14b-2512"

def mistral_post_with_retry(req, timeout=90, attempts=3):
    """POST to Mistral with retries on timeouts / 5xx / 429 only.

    A 4xx other than 429 (e.g. 404 bad model) fails fast -- retrying
    cannot fix it. 429 is retryable: rate-limit windows are usually
    per-minute, so 429 uses longer backoff (30s then 60s) than
    timeouts/5xx (10s then 20s). A timeout, like the Oct 24 mock run,
    is retried: 90s per attempt. Raises the last error.
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
            print(f"Mistral attempt {attempt} failed ({last_exc}); retrying...",
                  file=sys.stderr)
            if isinstance(last_exc, urllib.error.HTTPError) and last_exc.code == 429:
                time.sleep(30 * attempt)
            else:
                time.sleep(10 * attempt)
    raise last_exc



def feasibility_flag(target_atr, minutes_left) -> str:
    if target_atr is None or minutes_left is None:
        return ""
    try:
        atr = float(target_atr); mins = int(minutes_left)
    except Exception:
        return ""
    if (mins <= 60 and atr >= 5.5) or (mins <= 30 and atr >= 4.5):
        return f"INFEASIBLE: target {atr:.2f} ATR with {mins} min to close"
    if (mins <= 120 and atr >= 5.5) or (mins <= 60 and atr >= 4.5):
        return f"TIGHT: target {atr:.2f} ATR with {mins} min to close"
    return ""


def minutes_to_close_us(gen_dt) -> int:
    return (16*60) - (gen_dt.hour*60 + gen_dt.minute)

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



def _norm_ticker(tick) -> str:
    return str(tick or "").upper().replace("-", "").replace(" ", "")


def load_earnings_calendar_us(root: Path) -> dict:
    """{normalised ticker: {ISO dates}} from earnings_calendar_us.json.

    Flag only, never blocks a verdict. Fail-open: a missing or
    malformed file disables the results-risk flag with one log line.
    STALENESS WARNING: this calendar is a snapshot (Oct 2026 US
    earnings dates). It goes stale and needs a refresh route before
    it is relied on long-term; a stale calendar cannot invent dates,
    but it can miss new ones.
    """
    path = root / "earnings_calendar_us.json"
    out = {}
    if not path.exists():
        print("No earnings_calendar_us.json; results-risk flag inactive")
        return out
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"earnings_calendar_us.json unreadable ({exc}); results-risk flag OFF")
        return {}
    if not isinstance(rows, list):
        print("earnings_calendar_us.json is not a list; results-risk flag OFF")
        return {}
    for row in rows:
        try:
            d = datetime.fromisoformat(str(row["date"])).date().isoformat()
            out.setdefault(_norm_ticker(row["ticker"]), set()).add(d)
        except Exception:
            print(f"Skipping bad earnings calendar row: {row!r}")
            continue
    return out


def results_flag(calendar: dict, ticker, signal_date) -> str:
    """One-line flag, or '' if no results date is near the signal date.

    Window: reports today, tomorrow, yesterday, 2 days ago, or 3 days
    ago (Friday results flagged on Monday). Anything else is not
    flagged, so a stale calendar cannot spam every verdict.
    """
    dates = calendar.get(_norm_ticker(ticker)) or set()
    if not dates or signal_date is None:
        return ""
    for d in sorted(dates):
        try:
            ed = datetime.fromisoformat(d).date()
        except Exception:
            continue
        delta = (ed - signal_date).days
        if delta == 0:
            return f"reports results TODAY ({d})"
        if delta == 1:
            return f"reports results TOMORROW ({d})"
        if delta == -1:
            return f"reported results YESTERDAY ({d})"
        if delta == -2:
            return f"reported results 2 DAYS AGO ({d})"
        if delta == -3:
            return f"reported results 3 DAYS AGO ({d})"
    return ""


def mistral_judge(candidates: list[dict], spy_pct, signal_time_et: str, minutes_left=None, calendar=None, signal_date=None) -> list[dict]:
    """One Mistral call judging all candidates. Returns one verdict dict per
    candidate (ticker-matched). Raises on transport/API failure."""
    api_key = os.environ["MISTRAL_API_KEY"]
    # N5: validate per candidate — one malformed candidate must not
    # kill the whole batch. Skip it with a log line, judge the rest.
    _valid = []
    for _c in candidates:
        try:
            if not _c.get("ticker") or not _c.get("signal"):
                raise ValueError("missing ticker/signal")
            float(_c["ltp"]); float(_c["stop"]); float(_c["target"])
            if _c.get("indicators") is not None and not isinstance(_c.get("indicators"), dict):
                raise ValueError("indicators not a dict")
        except Exception as exc:
            print(f"Skipping malformed candidate {_c.get('ticker', '?')}: {exc}", file=sys.stderr)
            continue
        _valid.append(_c)
    candidates = _valid
    if not candidates:
        raise RuntimeError("All candidates malformed; nothing to judge")
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
            "results_risk": results_flag(calendar or {}, c["ticker"], signal_date),
            "feasibility": feasibility_flag(round(abs(tgt - entry) / atr, 2) if atr > 0 else None, minutes_left),
            "candles_5m_last12": (c.get("candles_5m") or [])[-12:],
            "minutes_left": minutes_left,
        })
    prompt = (
        "You are the senior analyst desk for a US intraday paper-trading system. "
        "This is an ADVISORY review: your verdict is logged and your accuracy is "
        "scored at end of day; you do not place trades.\n\n"
        f"Session context: signal time {signal_time_et} ET (regular session 09:30-16:00 ET). "
        f"Minutes remaining to the 16:00 ET close: {minutes_left} — use THIS number for all "
        "time-to-close reasoning; do not recompute it from the signal time. "
        f"SPY day move: {spy_pct if spy_pct is not None else 'unknown'}%. "
        "Candles are regular-session 5-minute bars [time, open, high, low, close, volume]. "
        "Volume comes from the IEX feed only (~2-4% of consolidated tape): treat vol_ratio "
        "as directional evidence, never as exact. There is no news feed and no VWAP in this "
        "payload — judge structure and geometry only, and say so if that limits you.\n\n"
        "FEASIBILITY: each candidate carries a feasibility field computed from "
        "target_distance_atr and minutes left to the 16:00 ET close, NOT from Mistral. "
        "If it starts INFEASIBLE, mention it and default to SKIP unless the structure "
        "is exceptional and you say why; if TIGHT, weigh it in risk. If empty, do not "
        "invent a feasibility problem.\n\n"
        "For EACH candidate decide GO or SKIP. Weigh: (1) candle structure — is the pullback "
        "orderly with a directional close, or is price extended/exhausted into the entry; "
        "(2) geometry — stop distance in ATR, reward:risk, and whether the target is realistic "
        "for the remaining session; (3) regime — does the trade direction agree with the SPY "
        "day move; (4) timing — entries after 14:30 ET have little room, midday entries need "
        "stronger structure. Be selective: a SKIP costs nothing. "
        "RESULTS RISK: each candidate carries a results_risk field taken ONLY from earnings_calendar_us.json supplied with this prompt. If it is non-empty, you MUST mention it in risk/notes and weigh pre-/post-results volatility: a GO on a results-today/tomorrow name needs exceptional structure. If it is empty, do NOT invent a results date or claim a company is reporting.\n\n"
        "CONFIDENCE: confidence is your conviction in the verdict you just gave for that candidate (GO or SKIP), as an integer. A verdict you stand behind is 50-95; a SKIP you believe in is 60-90. Never output 0 and never omit the confidence field — 0 means you have no opinion, not low conviction, and is not a valid confidence for a verdict. "
        "Return ONLY this JSON: {\"verdicts\": [{\"ticker\": str, \"verdict\": \"GO\"|\"SKIP\", "
        "\"confidence\": int 50-95, \"trigger\": str, \"notes\": {\"technical\": str, \"macro\": str, \"risk\": str}}]} "
        "trigger is the price condition that makes the verdict actionable: for GO, "
        "the entry condition; for SKIP, the condition that would make it valid "
        "(e.g. 'Valid only on a break below 1183.60'), or '' if none exists in-session. "
        "— one entry per candidate, citing concrete numbers from the payload in the notes.\n\n"
        f"Candidates: {json.dumps(features)}"
    )
    body = {
        "model": MISTRAL_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        "https://api.mistral.ai/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    # Mistral free tier allows ~1 request/second; a short pause before the
    # call keeps a queued push-run + cron-run pair from bursting it.
    time.sleep(1.2)
    data = json.loads(mistral_post_with_retry(req))
    text = data["choices"][0]["message"]["content"]
    parsed = json.loads(text)
    verdicts = parsed.get("verdicts", [])
    by_ticker = {str(v.get("ticker", "")).upper(): v for v in verdicts if isinstance(v, dict)}
    out = []
    for c in candidates:
        v = by_ticker.get(c["ticker"].upper())
        if v is None:
            raise RuntimeError(f"Mistral returned no verdict for {c['ticker']}")
        verdict = str(v.get("verdict", "SKIP")).upper()
        if verdict not in ("GO", "SKIP"):
            verdict = "SKIP"
        try:
            conf = int(v.get("confidence", 0))
        except (TypeError, ValueError):
            raise RuntimeError(f"Invalid confidence for {c['ticker']}: {v.get('confidence')!r}")
        if conf <= 0 or conf > 100:
            raise RuntimeError(f"Invalid confidence {conf} for {c['ticker']}: 0 is not a valid confidence")
        notes = v.get("notes") if isinstance(v.get("notes"), dict) else {"technical": str(v.get("notes", ""))}
        out.append({"ticker": c["ticker"], "verdict": verdict, "confidence": conf, "trigger": str(v.get("trigger", "")).strip(), "notes": notes})
    return out


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


def send_telegram(text):
    token = os.getenv('TELEGRAM_BOT_TOKEN', '')
    chat = os.getenv('TELEGRAM_CHAT_ID', '')
    if not token or not chat:
        print('Telegram not configured; message was:' + chr(10) + text)
        return
    for i, chunk in enumerate(_split_chunks(text), start=1):
        req = urllib.request.Request(
            'https://api.telegram.org/bot' + token + '/sendMessage',
            data=json.dumps({'chat_id': chat, 'text': chunk}).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                if not (200 <= resp.status < 300):
                    print('ERROR: Telegram send failed: HTTP %s (chunk %d) — verdict saved but NOT delivered via Telegram' % (resp.status, i), file=sys.stderr)
        except Exception as exc:
            print('ERROR: Telegram send failed: %s (chunk %d) — verdict saved but NOT delivered via Telegram' % (exc, i), file=sys.stderr)


def main() -> int:
    root = repo_root()
    latest_path = root / "signals" / "latest.json"
    verdict_path = root / "desk_verdict_mistral.json"
    history_path = root / "desk_history_mistral.json"

    latest = load_json(latest_path, None)
    if not latest or not latest.get("candidates"):
        print("No candidates in signals/latest.json; nothing to judge.")
        return 0

    signal_id = str(latest.get("signal_id", ""))
    if not signal_id:
        print("latest.json has no signal_id; skipping (cannot dedupe).")
        return 0
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
    if not (dtime(9, 30) <= gen_et.time() <= dtime(16, 0)):
        print(f"Signal generated outside ET session ({gen_et:%H:%M}); skipping.")
        return 0

    history = load_json_strict(history_path, [])
    if not isinstance(history, list):
        raise RuntimeError(f"{history_path} is not a JSON list; refusing to overwrite it")
    if any(isinstance(r, dict) and r.get("signal_id") == signal_id for r in history):
        print(f"Signal {signal_id} already judged; skipping.")
        return 0

    kept, skipped = filter_deduped(latest["candidates"], history, gen_et)
    if skipped:
        print(f"Deduped (same ticker+direction within 2h, skipping): {', '.join(skipped)}")
    if not kept:
        print("All candidates deduped; skipping (no Mistral call, no commit).")
        return 0

    calendar = load_earnings_calendar_us(root)
    if not os.getenv("MISTRAL_API_KEY"):
        print("MISTRAL_API_KEY not set; cannot judge.", file=sys.stderr)
        return 0

    spy_pct = latest.get("spy_day_pct")
    try:
        verdicts = mistral_judge(kept, spy_pct, gen_et.strftime("%H:%M"), minutes_left=minutes_to_close_us(gen_et), calendar=calendar, signal_date=gen_et.date())
    except Exception as exc:
        print(f"Mistral judging failed: {exc}", file=sys.stderr)
        # C7: judging failures are otherwise silent (exit 0, no verdict):
        # alert loudly, but still write nothing.
        try:
            send_telegram(f"DESK judging failed: {exc} — batch {signal_id} NOT judged")
        except Exception as alert_exc:
            # C7 nit: a failing alert must not itself be silent.
            print(f"DESK judging-failure alert failed: {alert_exc}", file=sys.stderr)
        return 0

    judged_at = now_et.isoformat(timespec="seconds")
    cand_by_ticker = {c["ticker"]: c for c in latest["candidates"]}
    records = []
    for v in verdicts:
        c = cand_by_ticker[v["ticker"]]
        records.append({
            "signal_id": signal_id,
            "signal_time": gen_et.isoformat(timespec="seconds"),
            "judged_at": judged_at, "age_min": round(age_min, 1),
            "ticker": v["ticker"], "action": c["signal"],
            "entry": float(c["ltp"]), "stop": float(c["stop"]), "target": float(c["target"]),
            "qty": c.get("qty", 0),
            "verdict": v["verdict"], "confidence": v["confidence"], "trigger": v.get("trigger", ""), "notes": v["notes"], "results_risk": results_flag(calendar, v["ticker"], gen_et.date()),
            "feasibility": feasibility_flag(round(abs(float(c["target"]) - float(c["ltp"])) / (float(c.get("indicators", {}).get("atr") or 0) or 0.0), 2) if float(c.get("indicators", {}).get("atr") or 0) else None, minutes_to_close_us(gen_et)),
            "outcome": None, "outcome_r": None, "outcome_at": None,
        })
    history.extend(records)
    save_json(history_path, history)
    save_json(verdict_path, {
        "signal_id": signal_id, "judged_at": judged_at, "age_min": round(age_min, 1),
        "spy_day_pct": spy_pct, "advisory_only": True, "verdicts": records,
    })

    spy_txt = f"{spy_pct:+.2f}%" if isinstance(spy_pct, (int, float)) else "n/a"
    lines = [f"US MISTRAL DESK (advisory) — {gen_et:%H:%M} ET · SPY {spy_txt}"]
    for r in records:
        icon = "GO ✅" if r["verdict"] == "GO" else "SKIP ⛔"
        tech = (r["notes"] or {}).get("technical", "")
        risk = (r["notes"] or {}).get("risk", "")
        lines.append(f"{r['ticker']} {r['action']} @ {r['entry']:.2f} → {icon} ({r['confidence']}%)")
        lines.append(f"  SL {r['stop']:.2f} / Target {r['target']:.2f}")
        if r.get("trigger"):
            lines.append(f"  Trigger: {r['trigger']}")
        if r.get("results_risk"):
            lines.append(f"  Results risk: {r['results_risk']}")
        if r.get("feasibility"):
            lines.append(f"  Feasibility: {r['feasibility']}")
        if tech:
            lines.append(f"  {tech}")
        if risk:
            lines.append(f"  Key risk: {risk}")
    send_telegram("\n".join(lines))
    print(f"Judged {len(records)} candidate(s) from {signal_id}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
