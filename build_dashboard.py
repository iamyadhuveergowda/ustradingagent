#!/usr/bin/env python3
"""
build_dashboard.py — regenerates a richer index.html for the US Intraday Agent.

Reads ONLY files that already exist in this repo (plus the current index.html,
which it parses for the broker-side numbers it cannot recompute), and writes
index.html. This script makes no network requests and modifies nothing else.

Index levels (S&P 500, Dow, Nasdaq, Russell 2000, VIX) come from market_us.json,
which is written by fetch_market_us.py — a separate script that does the
network fetch. If market_us.json is absent the strip shows a notice instead.

Usage:
    python3 build_dashboard.py                 # read from repo root, write index.html
    python3 build_dashboard.py --out out.html  # write somewhere else
    python3 build_dashboard.py --repo /path    # different checkout

The agent's own sync step should call this AFTER it writes the JSON files and
BEFORE it commits. Example (from the sync script):

    from build_dashboard import render_html
    html = render_html(
        equity=equity, available=available, realized=realized,
        today_pnl=today_pnl, updated_label="2026-11-11 21:07 IST",
        mode="PAPER", version="v2.9",
    )
    open("index.html", "w").write(html)

If you'd rather not touch the sync script, just run this file after the sync
and let the existing commit step pick up index.html.
"""

from __future__ import annotations

import argparse
import html as _html
import json
import math
import re
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

_IST = ZoneInfo("Asia/Kolkata")
_EASTERN = ZoneInfo("America/New_York")
_STAMP_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})[ T](\d{2}):(\d{2})(?::\d{2})?\s*(ET|UTC|IST)?", re.I
)


def _to_ist_label(raw: str) -> str:
    """Normalize an 'Updated' stamp to IST for the page header.

    Old pages carry ET or UTC stamps (the v1 page wrote ET, the first
    generator wrote UTC). Parse the stamp in its own zone and render it
    in IST; anything unparseable passes through unchanged so a bad stamp
    can never break the page.
    """
    if not raw:
        return raw
    m = _STAMP_RE.search(raw)
    if not m:
        return raw
    date_part, hh, mm = m.group(1), int(m.group(2)), int(m.group(3))
    tz = (m.group(4) or "IST").upper()
    try:
        naive = datetime.strptime(f"{date_part} {hh:02d}:{mm:02d}", "%Y-%m-%d %H:%M")
    except ValueError:
        return raw
    if tz == "ET":
        aware = naive.replace(tzinfo=_EASTERN)
    elif tz == "UTC":
        aware = naive.replace(tzinfo=timezone.utc)
    else:  # already IST (or no zone given — treat as IST, the display zone)
        aware = naive.replace(tzinfo=_IST)
    return aware.astimezone(_IST).strftime("%Y-%m-%d %H:%M IST")

# ---------------------------------------------------------------------------
# load helpers
# ---------------------------------------------------------------------------


def _load(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # never let a half-written file kill the page
        print(f"  ! could not parse {path.name}: {exc}")
        return default


def _load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def esc(value) -> str:
    return _html.escape("" if value is None else str(value), quote=True)


def money(value, digits=2) -> str:
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.{digits}f}"


def signed_money(value, digits=2) -> str:
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    sign = "-" if value < 0 else "+"
    return f"{sign}${abs(value):,.{digits}f}"


def pct(value, digits=1) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}%"


# ---------------------------------------------------------------------------
# broker-side numbers come from the existing index.html
# ---------------------------------------------------------------------------

_KPI_RE = re.compile(
    r'<div class="lbl">(?P<label>[^<]+)</div>\s*<div class="big[^"]*">(?P<value>.*?)</div>',
    re.S,
)
_TAG_RE = re.compile(r"<[^>]+>")
_NUM_RE = re.compile(r"-?\$?-?[\d,]+(?:\.\d+)?")


def parse_existing_index(path: Path) -> dict:
    """Pull the equity/available/realized/today numbers out of the current page.

    These are the only values that live in the broker account rather than in a
    repo file, so we reuse whatever the agent last wrote instead of inventing
    them. Falls back to computing equity from the ledger when absent.
    """
    out = {
        "equity": None,
        "available": None,
        "realized": None,
        "today_pnl": None,
        "updated": "",
        "version": "",
        "mode": "",
    }
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8")

    m = re.search(r'<div class="updated">(.*?)</div>', text, re.S)
    if m:
        raw = _html.unescape(_TAG_RE.sub("", m.group(1))).strip()
        # The template prints "Updated <stamp> · auto-refresh 3min · N closed
        # trades · M desk judgements". Keep only the stamp, i.e. everything
        # before the first separator, so regenerating from our own output does
        # not accumulate the suffixes on every run.
        raw = raw.split("·")[0]
        raw = re.sub(r"^Updated\s+", "", raw, flags=re.I)
        out["updated"] = raw.strip(" ·|")

    m = re.search(r'<span class="ver">([^<]+)</span>', text)
    if m:
        out["version"] = m.group(1).strip()
    # the v1 page used class="mode"; this page uses class="badge paper|live"
    m = re.search(r'<span class="mode">([^<]+)</span>', text) or re.search(
        r'<span class="badge (?:paper|live)">([^<]+)</span>', text)
    if m:
        out["mode"] = m.group(1).strip()

    wanted = {
        "MTM Equity": "equity",
        "Available": "available",
        "Realized": "realized",
        "Today P&L": "today_pnl",
    }
    for match in _KPI_RE.finditer(text):
        label = match.group("label").replace("&amp;", "&").strip()
        key = wanted.get(label)
        if not key:
            continue
        raw = _TAG_RE.sub("", match.group("value"))
        num = _NUM_RE.search(raw.replace("$", "").replace(",", ""))
        if num:
            out[key] = float(num.group(0))
    return out


# ---------------------------------------------------------------------------
# analytics
# ---------------------------------------------------------------------------


def trade_stats(trades):
    closed = [t for t in trades if t.get("net_pnl") is not None]
    wins = [t for t in closed if t["net_pnl"] > 0]
    losses = [t for t in closed if t["net_pnl"] <= 0]
    gross_win = sum(t["net_pnl"] for t in wins)
    gross_loss = -sum(t["net_pnl"] for t in losses)
    net = sum(t["net_pnl"] for t in closed)

    # R multiple: realised P&L over the risk that was actually at stake.
    for t in closed:
        try:
            risk_per_share = abs(t["entry"] - t.get("initial_sl", t.get("sl")))
            qty = t.get("qty") or 0
            t["_r"] = (t["net_pnl"] / (risk_per_share * qty)) if risk_per_share and qty else None
        except Exception:
            t["_r"] = None

    rs = [t["_r"] for t in closed if t["_r"] is not None]
    holds = []
    for t in closed:
        try:
            fmt = "%Y-%m-%dT%H:%M:%S%z"
            entry_ts = datetime.strptime(t["date"] + "T" + t["time"], "%Y-%m-%dT%H:%M:%S")
            exit_ts = datetime.strptime(t["exit_time"], fmt).replace(tzinfo=None)
            holds.append((exit_ts - entry_ts).total_seconds() / 60.0)
        except Exception:
            continue

    by_date = OrderedDict()
    for t in sorted(closed, key=lambda x: (x["date"], x["time"])):
        by_date[t["date"]] = by_date.get(t["date"], 0.0) + (t["net_pnl"] or 0)

    equity, curve = 100000.0, [("start", 100000.0)]
    for date, pnl in by_date.items():
        equity += pnl
        curve.append((date, equity))

    by_ticker = {}
    for t in closed:
        row = by_ticker.setdefault(t["ticker"], {"n": 0, "wins": 0, "pnl": 0.0})
        row["n"] += 1
        row["wins"] += 1 if t["net_pnl"] > 0 else 0
        row["pnl"] += t["net_pnl"]

    return {
        "trades": closed,
        "n": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (100.0 * len(wins) / len(closed)) if closed else None,
        "net": net,
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "profit_factor": (gross_win / gross_loss) if gross_loss else None,
        "avg_win": (gross_win / len(wins)) if wins else None,
        "avg_loss": (-gross_loss / len(losses)) if losses else None,
        "expectancy": (net / len(closed)) if closed else None,
        "best": max(closed, key=lambda t: t["net_pnl"]) if closed else None,
        "worst": min(closed, key=lambda t: t["net_pnl"]) if closed else None,
        "avg_r": mean(rs) if rs else None,
        "max_drawdown": _max_drawdown([p for _, p in curve]),
        "curve": curve,
        "by_date": by_date,
        "by_ticker": by_ticker,
        "avg_hold_min": mean(holds) if holds else None,
        "charges": sum((t.get("charges") or {}).get("total", 0) for t in closed),
        "long": len([t for t in closed if t["action"] == "BUY"]),
        "short": len([t for t in closed if t["action"] == "SELL"]),
    }


def _max_drawdown(series):
    peak, worst = None, 0.0
    for value in series:
        peak = value if peak is None else max(peak, value)
        worst = min(worst, value - peak)
    return worst


def desk_stats(history):
    rows = [h for h in history if h.get("verdict")]
    verdicts = Counter(h["verdict"] for h in rows)
    scored = [h for h in rows if h.get("outcome")]
    outcomes = Counter(h["outcome"] for h in scored)
    go = [h for h in rows if h["verdict"] == "GO"]
    go_scored = [h for h in go if h.get("outcome_r") is not None]
    rs = [h["outcome_r"] for h in go_scored]

    # would-be edge: what the skipped signals did, as R multiples
    skip = [h for h in rows if h["verdict"] == "SKIP" and h.get("outcome_r") is not None]
    skip_rs = [h["outcome_r"] for h in skip]

    buckets = OrderedDict()
    for h in rows:
        conf = h.get("confidence")
        if conf is None:
            continue
        band = f"{int(conf // 10) * 10}-{int(conf // 10) * 10 + 9}"
        bucket = buckets.setdefault(band, {"n": 0, "go": 0, "target": 0, "stop": 0, "close": 0, "rs": []})
        bucket["n"] += 1
        if h["verdict"] == "GO":
            bucket["go"] += 1
        if h.get("outcome") == "TARGET":
            bucket["target"] += 1
        if h.get("outcome") == "STOP":
            bucket["stop"] += 1
        if h.get("outcome") == "CLOSE":
            bucket["close"] += 1
        if h.get("outcome_r") is not None:
            bucket["rs"].append(h["outcome_r"])

    by_day = OrderedDict()
    for h in sorted(rows, key=lambda x: (x.get("signal_time") or "")):
        day = (h.get("signal_time") or "")[:10]
        if not day:
            continue
        d = by_day.setdefault(day, {"n": 0, "go": 0, "target": 0, "stop": 0, "close": 0})
        d["n"] += 1
        if h["verdict"] == "GO":
            d["go"] += 1
        d[{"TARGET": "target", "STOP": "stop", "CLOSE": "close"}.get(h.get("outcome") or "", "n")] += 1

    latencies = []
    for h in rows:
        try:
            fmt = "%Y-%m-%dT%H:%M:%S%z"
            sig = datetime.strptime(h["signal_time"], fmt)
            jud = datetime.strptime(h["judged_at"], fmt)
            latencies.append((jud - sig).total_seconds())
        except Exception:
            continue

    return {
        "rows": rows,
        "n": len(rows),
        "verdicts": verdicts,
        "outcomes": outcomes,
        "go": go,
        "go_scored": go_scored,
        "go_n": len(go),
        "go_targets": len([h for h in go_scored if h["outcome"] == "TARGET"]),
        "go_stops": len([h for h in go_scored if h["outcome"] == "STOP"]),
        "go_closes": len([h for h in go_scored if h["outcome"] == "CLOSE"]),
        "go_sum_r": sum(rs) if rs else None,
        "go_avg_r": mean(rs) if rs else None,
        "skip_sum_r": sum(skip_rs) if skip_rs else None,
        "skip_avg_r": mean(skip_rs) if skip_rs else None,
        "skip_n": len(skip_rs),
        "buckets": buckets,
        "by_day": by_day,
        "avg_latency_s": mean(latencies) if latencies else None,
        "first": min((h["signal_time"] for h in rows if h.get("signal_time")), default=None),
        "last": max((h["signal_time"] for h in rows if h.get("signal_time")), default=None),
    }


def scout_stats(rows):
    if not rows:
        return None
    scans = len(rows)
    overlap = sum(len(r.get("overlap") or []) for r in rows)
    qwen_only = sum(len(r.get("qwen_only") or []) for r in rows)
    deep_only = sum(len(r.get("deepseek_only") or []) for r in rows)
    errors = sum(1 for r in rows if r.get("deepseek_error"))
    lats = [r["latency_ms"] for r in rows if isinstance(r.get("latency_ms"), (int, float))]
    qwen_picks = sum(len(r.get("qwen_picks") or []) for r in rows)
    deep_picks = sum(len(r.get("deepseek_picks") or []) for r in rows)
    return {
        "scans": scans,
        "overlap": overlap,
        "qwen_only": qwen_only,
        "deep_only": deep_only,
        "errors": errors,
        "qwen_picks": qwen_picks,
        "deep_picks": deep_picks,
        "avg_latency_ms": mean(lats) if lats else None,
        "agreement": (100.0 * overlap / (overlap + qwen_only + deep_only))
        if (overlap + qwen_only + deep_only)
        else None,
    }


# ---------------------------------------------------------------------------
# svg charts (hand-rolled so the page has zero external dependencies)
# ---------------------------------------------------------------------------


def svg_line(points, width=760, height=170, pad=26, color="#8b93ff", fill="rgba(139,147,255,0.14)", labels=None):
    """points: list of (label, value). Returns an <svg> string."""
    values = [p[1] for p in points]
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi == lo:
        hi += abs(hi) * 0.05 or 1.0
        lo -= abs(lo) * 0.05 or 1.0
    span = hi - lo
    lo -= span * 0.12
    hi += span * 0.12
    span = hi - lo

    def x(i):
        return pad + (width - 2 * pad) * (i / max(1, len(points) - 1))

    def y(v):
        return pad + (height - 2 * pad) * (1 - (v - lo) / span)

    coords = [(x(i), y(v)) for i, (_, v) in enumerate(points)]
    line = " ".join(f"{px:.1f},{py:.1f}" for px, py in coords)
    area = f"{pad},{height - pad} " + line + f" {width - pad},{height - pad}"

    ticks = []
    for frac in (0, 0.5, 1):
        val = lo + span * frac
        gy = y(val)
        ticks.append(
            f'<line x1="{pad}" y1="{gy:.1f}" x2="{width - pad}" y2="{gy:.1f}" stroke="#2a2a3d" stroke-width="1"/>'
            f'<text x="4" y="{gy + 4:.1f}" class="ax">{money(val, 0)}</text>'
        )
    # baseline at the starting value, if it is inside the range
    base = points[0][1]
    if lo <= base <= hi:
        gy = y(base)
        ticks.append(
            f'<line x1="{pad}" y1="{gy:.1f}" x2="{width - pad}" y2="{gy:.1f}" '
            f'stroke="#5a5f7a" stroke-width="1" stroke-dasharray="4 4"/>'
        )

    dots = []
    for (px, py), (lbl, val) in zip(coords, points):
        cls = "pos" if val >= base else "neg"
        dots.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" class="{cls}">'
            f'<title>{esc(lbl)}: {money(val)}</title></circle>'
        )
    xlabels = []
    if labels:
        step = max(1, len(points) // 7)
        labels = labels[:len(points)] + [""] * max(0, len(points) - len(labels))
        for i in range(0, len(points), step):
            xlabels.append(f'<text x="{x(i):.1f}" y="{height - 6}" class="ax" text-anchor="middle">{esc(labels[i])}</text>')

    return (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" class="chart">'
        f'<polygon points="{area}" fill="{fill}"/>'
        f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2"/>'
        + "".join(ticks) + "".join(dots) + "".join(xlabels) + "</svg>"
    )


def svg_bars(items, width=760, height=170, pad=26):
    """items: list of (label, value). Positive bars green, negative red."""
    if not items:
        return ""
    values = [v for _, v in items]
    lo, hi = min(values + [0]), max(values + [0])
    if hi == lo:
        hi += 1
    span = (hi - lo) * 1.15 or 1
    lo -= span * 0.07
    hi += span * 0.07
    span = hi - lo
    zero = pad + (height - 2 * pad) * (1 - (0 - lo) / span)
    bw = max(3.0, (width - 2 * pad) / len(items) - 4)

    rects = []
    for i, (lbl, val) in enumerate(items):
        cx = pad + (width - 2 * pad) * ((i + 0.5) / len(items))
        vy = pad + (height - 2 * pad) * (1 - (val - lo) / span)
        top, h = (vy, zero - vy) if val >= 0 else (zero, vy - zero)
        cls = "barpos" if val >= 0 else "barneg"
        rects.append(
            f'<rect x="{cx - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{max(1, h):.1f}" class="{cls}">'
            f'<title>{esc(lbl)}: {signed_money(val)}</title></rect>'
        )
    step = max(1, len(items) // 8)
    xlabels = [
        f'<text x="{pad + (width - 2 * pad) * ((i + 0.5) / len(items)):.1f}" y="{height - 6}" class="ax" text-anchor="middle">{esc(items[i][0])}</text>'
        for i in range(0, len(items), step)
    ]
    return (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" class="chart">'
        f'<line x1="{pad}" y1="{zero:.1f}" x2="{width - pad}" y2="{zero:.1f}" stroke="#5a5f7a" stroke-dasharray="4 4"/>'
        + "".join(rects) + "".join(xlabels) + "</svg>"
    )


def svg_candles(candles, entry=None, stop=None, target=None, width=760, height=190, pad=28):
    """candles: list of [time, open, high, low, close, volume]."""
    if not candles:
        return ""
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    refs = [v for v in (entry, stop, target) if v is not None]
    lo, hi = min(lows + refs), max(highs + refs)
    if hi == lo:
        hi += 1
    span = (hi - lo) * 1.1 or 1
    lo -= span * 0.05
    hi += span * 0.05
    span = hi - lo
    plot_w = width - 2 * pad
    cw = max(2.0, plot_w / len(candles) * 0.62)

    def y(v):
        return pad + (height - 2 * pad) * (1 - (v - lo) / span)

    out = []
    for i, c in enumerate(candles):
        cx = pad + plot_w * ((i + 0.5) / len(candles))
        _, o, h, l, cl, _v = (list(c) + [0] * 6)[:6]
        up = cl >= o
        cls = "cup" if up else "cdn"
        out.append(f'<line x1="{cx:.1f}" y1="{y(h):.1f}" x2="{cx:.1f}" y2="{y(l):.1f}" class="{cls}" stroke-width="1"/>')
        top = y(max(o, cl))
        hgt = max(1.0, abs(y(o) - y(cl)))
        out.append(f'<rect x="{cx - cw / 2:.1f}" y="{top:.1f}" width="{cw:.1f}" height="{hgt:.1f}" class="{cls}"/>')

    for val, label, col in ((entry, "ENTRY", "#e8e8f0"), (stop, "SL", "#ff6b6b"), (target, "TGT", "#34d17b")):
        if val is None or not (lo <= val <= hi):
            continue
        gy = y(val)
        out.append(
            f'<line x1="{pad}" y1="{gy:.1f}" x2="{width - pad}" y2="{gy:.1f}" stroke="{col}" stroke-width="1" stroke-dasharray="5 4" opacity="0.85"/>'
            f'<text x="{width - pad - 2}" y="{gy - 4:.1f}" class="ax" text-anchor="end" fill="{col}">{label} {val}</text>'
        )
    step = max(1, len(candles) // 8)
    for i in range(0, len(candles), step):
        cx = pad + plot_w * ((i + 0.5) / len(candles))
        out.append(f'<text x="{cx:.1f}" y="{height - 6}" class="ax" text-anchor="middle">{esc(candles[i][0])}</text>')
    return (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" class="chart">' + "".join(out) + "</svg>"
    )


def svg_hbar(rows, width=340, height=None, pad=8):
    """rows: list of (label, value, css_class)."""
    if not rows:
        return ""
    hi = max(abs(v) for _, v, _ in rows) or 1
    rh = 20
    height = height or (len(rows) * rh + 2 * pad)
    mid = pad + (width - 2 * pad) * 0.34
    out = []
    for i, (label, value, cls) in enumerate(rows):
        cy = pad + i * rh
        w = (width - 2 * pad) * 0.62 * (abs(value) / hi)
        if value >= 0:
            x0 = mid
        else:
            x0 = mid - w
        out.append(f'<text x="{pad - 2}" y="{cy + 13}" class="ax" text-anchor="start" width="{mid}">{esc(label)}</text>')
        out.append(f'<rect x="{x0:.1f}" y="{cy + 4}" width="{max(1, w):.1f}" height="11" class="{cls}"/>')
        out.append(f'<text x="{width - 2}" y="{cy + 13}" class="ax" text-anchor="end">{signed_money(value, 0)}</text>')
    out.append(f'<line x1="{mid:.1f}" y1="{pad}" x2="{mid:.1f}" y2="{height - pad}" stroke="#5a5f7a" stroke-dasharray="3 3"/>')
    return f'<svg viewBox="0 0 {width} {height}" class="chart" preserveAspectRatio="none">' + "".join(out) + "</svg>"


def sparkline(values, width=120, height=28, up=True):
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    pts = " ".join(
        f"{(width * i / max(1, len(values) - 1)):.1f},{(height - 2 - (height - 4) * (v - lo) / span):.1f}"
        for i, v in enumerate(values)
    )
    color = "#34d17b" if up else "#ff6b6b"
    return (
        f'<svg viewBox="0 0 {width} {height}" class="spark">'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/></svg>'
    )


# ---------------------------------------------------------------------------
# html
# ---------------------------------------------------------------------------

CSS = """
:root{--bg:#0e0e16;--panel:#171722;--panel2:#1d1d2b;--line:#26263a;--txt:#e8e8f0;--dim:#9aa0b4;--acc:#8b93ff;--pos:#34d17b;--neg:#ff6b6b;--warn:#ffd166}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--txt);font-family:'Segoe UI',system-ui,-apple-system,sans-serif;margin:0;padding:18px 20px 60px;font-size:14px;line-height:1.45}
a{color:var(--acc);text-decoration:none}
h1{font-size:21px;margin:0;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
h2{font-size:15px;margin:26px 0 10px;letter-spacing:.3px;color:#cfd2ff}
h3{font-size:13px;margin:0 0 6px;color:var(--dim);text-transform:uppercase;letter-spacing:.6px}
.ver{color:var(--acc);font-size:14px;font-weight:400}
.badge{border-radius:4px;font-size:11px;padding:2px 7px;border:1px solid;letter-spacing:.4px}
.badge.paper{color:var(--warn);border-color:var(--warn);background:#2c2c44}
.badge.live{color:var(--neg);border-color:var(--neg);background:#3a2230}
.updated{color:var(--dim);font-size:11.5px;margin:4px 0 16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:9px;margin-bottom:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:9px 11px}
.card .lbl{color:var(--dim);font-size:10.5px;text-transform:uppercase;letter-spacing:.55px}
.card .big{font-size:18px;font-weight:600;margin-top:3px;white-space:nowrap}
.card .sub{font-size:10.5px;color:var(--dim);margin-top:2px}
.pos{color:var(--pos)}.neg{color:var(--neg)}.dim{color:var(--dim)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:12px}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px}
table{width:100%;border-collapse:collapse;font-size:12px;background:var(--panel2);border-radius:8px;overflow:hidden}
th{background:#232338;color:var(--acc);text-align:left;padding:7px 8px;white-space:nowrap;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.4px}
td{padding:7px 8px;border-top:1px solid var(--line);vertical-align:top}
tr.detail td{background:#141420;color:#c9c9d6;font-size:11.5px}
tr.click{cursor:pointer}
tr.click:hover td{background:#20202f}
.buy{color:var(--pos);font-weight:600}.sell{color:var(--neg);font-weight:600}
.rat{max-width:340px;color:#c9c9d6;font-size:11px}
.pill{display:inline-block;border-radius:10px;padding:1px 7px;font-size:10.5px;border:1px solid var(--line)}
.pill.go{color:var(--pos);border-color:var(--pos)}
.pill.skip{color:var(--dim)}
.empty{text-align:center;color:var(--dim);padding:18px}
.chart{width:100%;height:auto;display:block}
.spark{width:110px;height:26px;vertical-align:middle}
.ax{font-size:9.5px;fill:#7f859b}
.cup{stroke:#34d17b;fill:#34d17b}.cdn{stroke:#ff6b6b;fill:#ff6b6b}
.barpos{fill:#34d17b}.barneg{fill:#ff6b6b}
.note{background:#141420;border-left:3px solid var(--acc);padding:8px 11px;border-radius:0 6px 6px 0;margin:8px 0;font-size:12px;color:#c9c9d6}
.note b{color:var(--txt)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-size:12px}
.kv div:nth-child(odd){color:var(--dim)}
.meter{height:6px;background:#26263a;border-radius:3px;overflow:hidden;margin-top:4px}
.meter i{display:block;height:100%;background:var(--acc)}
.mkt-panel{padding:10px 12px 12px}
.mkt-head{font-size:12.5px;margin-bottom:8px;display:flex;flex-wrap:wrap;gap:6px 10px;align-items:baseline}
.mkt-status{font-size:10.5px;letter-spacing:.5px;border:1px solid currentColor;border-radius:4px;padding:0 5px}
.mkt-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px}
.mkt-tile{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:8px 10px 6px}
.mkt-tile.stale{opacity:.55}
.mkt-name{font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.5px}
.mkt-sym{margin-left:6px;color:#6f7590;font-size:10px;text-transform:none;letter-spacing:0}
.mkt-px{font-size:20px;font-weight:600;margin-top:2px;font-variant-numeric:tabular-nums}
.mkt-chg{font-size:12.5px;font-variant-numeric:tabular-nums;margin-top:1px;min-height:17px}
.mkt-sub{font-size:10px;color:var(--dim);margin-top:3px}
.mkt-tile .spark{width:100%;height:34px;display:block;margin-top:4px}
.mkt-empty{font-size:13px}
.mkt-tape{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 10px;padding:7px 12px;font-size:12px}
.mkt-tape-items{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 0}
.mkt-tape-item{display:inline-flex;align-items:baseline;gap:5px;padding:0 10px;border-left:1px solid var(--line);white-space:nowrap}
.mkt-tape-item:first-child{border-left:0;padding-left:0}
.mkt-tape-item.stale{opacity:.55}
.mkt-tape-item .mkt-name{color:var(--dim);text-transform:uppercase;font-size:10px;letter-spacing:.5px}
.mkt-tape-item b{font-size:13.5px;font-weight:600;font-variant-numeric:tabular-nums}
.mkt-tape-time{font-size:10px}
.tag{font-size:10.5px;color:var(--dim);border:1px solid var(--line);border-radius:4px;padding:1px 5px;margin-left:4px}
.tag.risk{color:var(--warn);border-color:var(--warn)}
footer{margin-top:28px;color:var(--dim);font-size:11px;border-top:1px solid var(--line);padding-top:12px}
.toggle{background:#20202f;border:1px solid var(--line);color:var(--txt);border-radius:6px;padding:5px 11px;font-size:12px;cursor:pointer}
.toggle:hover{border-color:var(--acc)}
.row-actions{display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
"""

JS = """
function tog(id){var e=document.getElementById(id);if(e)e.classList.toggle('hidden');}
var _fmtN=function(x){return Math.abs(x).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});};
function _setF(root,name,html){var el=root.querySelector('[data-f="'+name+'"]');if(el)el.innerHTML=html;}
function refreshMarket(){
  var panel=document.querySelector('[data-market-url]');
  if(!panel||typeof fetch!=='function')return;
  fetch(panel.dataset.marketUrl+'?t='+Date.now(),{cache:'no-store'}).then(function(r){
    if(!r.ok)throw new Error(r.status);return r.json();
  }).then(function(doc){
    var byTile={};(doc.indices||[]).forEach(function(i){byTile[i.symbol]=i;});
    var anyOpen=false;
    panel.querySelectorAll('.mkt-tile').forEach(function(t){
      var i=byTile[t.dataset.sym];if(!i)return;
      anyOpen=anyOpen||!!i.session_open;
      _setF(t,'px',i.price.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}));
      if(i.change!=null){
        var up=i.change>=0, good=t.dataset.inverse==='1'?!up:up, cls=good?'pos':'neg';
        _setF(t,'chg','<span class="'+cls+'">'+(up?'▲ ':'▼ ')+(up?'+':'-')+_fmtN(i.change)+' ('+(up?'+':'')+i.change_pct.toFixed(2)+'%)</span>');
      }
      _setF(t,'time',(i.quote_time_et||'')+' ET');
      _setF(t,'stale',i.stale?' <span class="tag risk">stale</span>':'');
      t.classList.toggle('stale',!!i.stale);
    });
    var st=panel.querySelector('[data-f="status"]');
    if(st){st.textContent=anyOpen?'LIVE · intraday':'CLOSED · last session';st.className='mkt-status '+(anyOpen?'pos':'dim');}
    var times=(doc.indices||[]).map(function(i){return i.quote_time_et||'';}).sort();
    _setF(panel,'asof','as of '+(times.pop()||'—')+' ET · refreshed '+new Date().toLocaleTimeString());
  }).catch(function(){/* file:// or offline: keep the values already on the page */});
}
if(typeof document.querySelector==='function'&&document.querySelector('[data-market-url]')){refreshMarket();setInterval(refreshMarket,60000);}
document.addEventListener('click',function(ev){
  var row=ev.target.closest('tr.click');
  if(!row)return;
  var d=document.getElementById(row.dataset.detail);
  if(d)d.classList.toggle('hidden');
});
function filterTable(inputId, tableId){
  var q=document.getElementById(inputId).value.toLowerCase().trim();
  document.querySelectorAll('#'+tableId+' tbody tr.click').forEach(function(tr){
    var hay=(tr.dataset.key||tr.textContent).toLowerCase();
    tr.style.display = !q || hay.indexOf(q)>-1 ? '' : 'none';
  });
}
var h=document.createElement('style');h.textContent='.hidden{display:none}';document.head.appendChild(h);
"""


def _hidden(cls=""):
    return f' class="detail hidden{(" " + cls) if cls else ""}"'


# ---------------------------------------------------------------- markets
# Index levels come from market_us.json, written by fetch_market_us.py. This
# function only reads that file, so the page still makes no network requests.
# Tiles follow the NSE/BSE convention: name, level, change in points and %.

# VIX is a fear gauge, so a rise is bad news: colour it inversely.
_INVERSE = {"^VIX"}


def _fmt_ist(utc_iso: str) -> str:
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.strptime(utc_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return dt.astimezone(ZoneInfo("Asia/Kolkata")).strftime("%H:%M IST")
    except Exception:  # noqa: BLE001 — a bad timestamp must not break the page
        return ""


def market_strip(repo: Path, latest: dict) -> str:
    mk = _load(repo / "market_us.json", None)
    indices = [i for i in (mk or {}).get("indices", []) if isinstance(i, dict) and i.get("price") is not None]

    spy_pct = latest.get("spy_day_pct") if isinstance(latest, dict) else None
    spy_txt = f"SPY {spy_pct:+.2f}% at signal time" if isinstance(spy_pct, (int, float)) else ""

    if not indices:
        return (
            '<div class="panel mkt-empty"><b>Index levels not available.</b> '
            'Run <code>fetch_market_us.py</code> on a schedule to write <code>market_us.json</code>.'
            + (f' <span class="dim">{esc(spy_txt)}</span>' if spy_txt else "")
            + "</div>"
        )

    any_open = any(i.get("session_open") for i in indices)
    status_cls = "pos" if any_open else "dim"
    status = "LIVE · intraday" if any_open else "CLOSED · last session"
    ets = sorted({i.get("quote_time_et", "") for i in indices if i.get("quote_time_et")})
    as_of_et = ets[-1] if ets else "—"
    ist = _fmt_ist(mk.get("fetched_at_utc", ""))
    as_of = f"as of {as_of_et} ET" + (f" ({ist})" if ist else "")
    # the page's own JS re-reads market_us.json every 60s and updates these hooks

    tiles = []
    for i in indices:
        chg = i.get("change")
        cpct = i.get("change_pct")
        up = (chg or 0) >= 0
        good = (not up) if i.get("symbol") in _INVERSE else up
        cls = "pos" if good else "neg"
        if chg is None:
            chg_txt = '<span class="dim">no prior close</span>'
        else:
            arrow = "▲" if up else "▼"
            chg_txt = f'<span class="{cls}">{arrow} {chg:+,.2f} ({cpct:+.2f}%)</span>'
        spark_vals = i.get("spark") or []
        spark = sparkline(spark_vals, width=150, height=34, up=good) if len(spark_vals) > 1 else ""
        stale = ' <span class="tag risk">stale</span>' if i.get("stale") else ""
        sym = esc(i.get("symbol", ""))
        inv = 1 if i.get("symbol") in _INVERSE else 0
        tiles.append(
            f'<div class="mkt-tile{" stale" if i.get("stale") else ""}" data-sym="{sym}" data-inverse="{inv}">'
            f'<div class="mkt-name">{esc(i.get("name", ""))}'
            f'<span class="mkt-sym">{sym}</span><span data-f="stale">{stale}</span></div>'
            f'<div class="mkt-px" data-f="px">{i["price"]:,.2f}</div>'
            f'<div class="mkt-chg" data-f="chg">{chg_txt}</div>'
            f'{spark}'
            f'<div class="mkt-sub" data-f="time">{esc(i.get("quote_time_et", ""))} ET</div>'
            "</div>"
        )

    return (
        '<div class="panel mkt-panel" data-market-url="market_us.json">'
        f'<div class="mkt-head"><b>US markets</b> '
        f'<span class="{status_cls} mkt-status" data-f="status">{status}</span> '
        f'<span class="dim" data-f="asof">{esc(as_of)}</span> '
        f'<span class="dim">· points and % vs previous close</span></div>'
        f'<div class="mkt-grid">{"".join(tiles)}</div>'
        "</div>"
    )


def market_tape(repo: Path, latest: dict) -> str:
    """Compact alternative layout: one thin ticker line, no tiles, no sparklines.

    Same market_us.json, same data-sym / data-f hooks, so the page's refresh JS
    drives both layouts without any change.
    """
    mk = _load(repo / "market_us.json", None)
    indices = [i for i in (mk or {}).get("indices", []) if isinstance(i, dict) and i.get("price") is not None]

    spy_pct = latest.get("spy_day_pct") if isinstance(latest, dict) else None
    spy_txt = f"SPY {spy_pct:+.2f}% at signal time" if isinstance(spy_pct, (int, float)) else ""

    if not indices:
        return (
            '<div class="panel mkt-empty"><b>Index levels not available.</b> '
            'Run <code>fetch_market_us.py</code> on a schedule to write <code>market_us.json</code>.'
            + (f' <span class="dim">{esc(spy_txt)}</span>' if spy_txt else "")
            + "</div>"
        )

    any_open = any(i.get("session_open") for i in indices)
    status_cls = "pos" if any_open else "dim"
    status = "LIVE" if any_open else "CLOSED"
    ets = sorted({i.get("quote_time_et", "") for i in indices if i.get("quote_time_et")})
    as_of_et = ets[-1] if ets else "—"
    ist = _fmt_ist(mk.get("fetched_at_utc", ""))
    as_of = f"{as_of_et} ET" + (f" · {ist}" if ist else "")

    items = []
    for i in indices:
        chg = i.get("change")
        cpct = i.get("change_pct")
        up = (chg or 0) >= 0
        good = (not up) if i.get("symbol") in _INVERSE else up
        cls = "pos" if good else "neg"
        chg_txt = (
            '<span class="dim">—</span>' if chg is None
            else f'<span class="{cls}">{chg:+,.2f} ({cpct:+.2f}%)</span>'
        )
        stale = ' <span class="tag risk">stale</span>' if i.get("stale") else ""
        sym = esc(i.get("symbol", ""))
        inv = 1 if i.get("symbol") in _INVERSE else 0
        items.append(
            f'<span class="mkt-tile mkt-tape-item{" stale" if i.get("stale") else ""}" '
            f'data-sym="{sym}" data-inverse="{inv}">'
            f'<span class="mkt-name">{esc(i.get("name", ""))}</span> '
            f'<b data-f="px">{i["price"]:,.2f}</b> '
            f'<span data-f="chg">{chg_txt}</span>'
            f'<span data-f="stale">{stale}</span>'
            f'<span class="dim mkt-tape-time" data-f="time">{esc(i.get("quote_time_et", ""))}</span>'
            "</span>"
        )

    return (
        '<div class="panel mkt-panel mkt-tape" data-market-url="market_us.json">'
        f'<span class="{status_cls} mkt-status" data-f="status">{status}</span> '
        f'<span class="mkt-tape-items">{"".join(items)}</span>'
        f'<span class="dim" data-f="asof">{esc(as_of)}</span>'
        "</div>"
    )


def market_panel(repo: Path, latest: dict, layout: str = "tiles") -> str:
    """Dispatch to the requested layout. tiles = index cards, tape = one line."""
    return market_tape(repo, latest) if layout == "tape" else market_strip(repo, latest)


def kpi(label, value, sub="", cls=""):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f'<div class="card"><div class="lbl">{label}</div><div class="big {cls}">{value}</div>{sub_html}</div>'


def render_html(
    repo: Path | None = None,
    *,
    equity=None,
    available=None,
    realized=None,
    today_pnl=None,
    updated_label="",
    mode="",
    version="",
    layout="tiles",
) -> str:
    """Build the full page from the repo's JSON files.

    Pass the broker-side numbers explicitly when calling from the agent's sync
    step; otherwise they are parsed out of the current index.html.
    """
    repo = Path(repo) if repo else Path(__file__).resolve().parent
    old_index = repo / "index.html"

    if equity is None and old_index.exists():
        parsed = parse_existing_index(old_index)
        equity = parsed["equity"]
        available = parsed["available"]
        realized = parsed["realized"]
        today_pnl = parsed["today_pnl"]
        updated_label = _to_ist_label(updated_label or parsed["updated"])
        mode = mode or parsed["mode"]
        version = version or parsed["version"]

    trades_raw = _load(repo / "trades" / "daily_trade_ledger_us.json", [])
    gemini = _load(repo / "desk_history.json", [])
    mistral = _load(repo / "desk_history_mistral.json", [])
    verdict = _load(repo / "desk_verdict.json", {})
    verdict_m = _load(repo / "desk_verdict_mistral.json", {})
    latest = _load(repo / "signals" / "latest.json", {})
    earnings = _load(repo / "earnings_calendar_us.json", [])
    scout = _load_jsonl(repo / "signals" / "shadow_scout_deepseek.jsonl")
    market_html = market_panel(repo, latest, layout)

    ts = trade_stats(trades_raw)
    gs = desk_stats(gemini)
    ms = desk_stats(mistral)
    sc = scout_stats(scout)

    # fall back to the ledger when the page has never been written
    if realized is None:
        realized = ts["net"]
    if equity is None:
        equity = 100000.0 + ts["net"]
    if available is None:
        available = equity

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    todays = [t for t in ts["trades"] if t["date"] == today]
    if today_pnl is None:
        today_pnl = ts["by_date"].get(today, 0.0)

    mode = mode or "PAPER"
    version = version or "v2.9"
    mode_cls = "paper" if mode.upper() == "PAPER" else "live"
    if not updated_label:
        updated_label = datetime.now(_IST).strftime("%Y-%m-%d %H:%M IST") + " · auto-refresh 3min"

    earn_dates = {e["date"]: [] for e in earnings if e.get("date")}
    for e in earnings:
        if e.get("date"):
            earn_dates[e["date"]].append(e["ticker"])

    # ------------------------------------------------------------------ cards
    cards = [
        kpi("MTM Equity", money(equity), "starting $100,000"),
        kpi("Available", money(available)),
        kpi("Realized", signed_money(realized), pct(100.0 * realized / 100000.0, 2) + " on $100k",
            "pos" if (realized or 0) >= 0 else "neg"),
        kpi("Today P&L", signed_money(today_pnl), f"{len(todays)} trade{'s' if len(todays) != 1 else ''} today",
            "pos" if (today_pnl or 0) > 0 else ("neg" if (today_pnl or 0) < 0 else "")),
        kpi("Win Rate", pct(ts["win_rate"]), f'<span class="pos">{ts["wins"]}</span> / <span class="neg">{ts["losses"]}</span> of {ts["n"]}'),
        kpi("Profit Factor", f'{ts["profit_factor"]:.2f}' if ts["profit_factor"] else "—",
            f'{money(ts["gross_win"], 0)} won / {money(ts["gross_loss"], 0)} lost',
            "pos" if (ts["profit_factor"] or 0) >= 1 else "neg"),
        kpi("Expectancy", signed_money(ts["expectancy"]), "per closed trade",
            "pos" if (ts["expectancy"] or 0) >= 0 else "neg"),
        kpi("Avg R", f'{ts["avg_r"]:.2f}R' if ts["avg_r"] is not None else "—", "P&L ÷ risk at entry",
            "pos" if (ts["avg_r"] or 0) >= 0 else "neg"),
        kpi("Max Drawdown", money(ts["max_drawdown"], 0), "from running peak", "neg" if ts["max_drawdown"] < 0 else ""),
        kpi("Best / Worst",
            f'{signed_money(ts["best"]["net_pnl"], 0)} / {signed_money(ts["worst"]["net_pnl"], 0)}'
            if ts["best"] and ts["worst"] else "—",
            (f'{esc(ts["best"]["ticker"])} / {esc(ts["worst"]["ticker"])}' if ts["best"] and ts["worst"] else "")),
        kpi("Avg Hold", f'{ts["avg_hold_min"]:.0f} min' if ts["avg_hold_min"] else "—",
            "entry → exit", ""),
        kpi("Desk Judged", f'{ms["n"]}',
            f'{ms["verdicts"].get("GO", 0)} GO · {ms["verdicts"].get("SKIP", 0)} SKIP', ""),
    ]
    cards_html = "".join(cards)

    # ------------------------------------------------------------- charts
    curve = ts["curve"]
    curve_labels = [("start" if i == 0 else lbl[5:]) for i, (lbl, _) in enumerate(curve)]
    chart_curve = svg_line(curve, labels=curve_labels) if len(curve) > 1 else ""
    daily = [(d[5:], v) for d, v in ts["by_date"].items()]
    chart_daily = svg_bars(daily)

    ticker_rows = sorted(ts["by_ticker"].items(), key=lambda kv: kv[1]["pnl"])
    chart_ticker = svg_hbar([(t, v["pnl"], "barpos" if v["pnl"] >= 0 else "barneg") for t, v in ticker_rows])

    # ------------------------------------------------------- diagnostics
    findings = []
    if ts["n"]:
        stop_rate = len([t for t in ts["trades"] if (t.get("status") or "").startswith("LOSS")]) / ts["n"]
        findings.append(
            f'<b>{pct(100 * stop_rate, 0)} of closed trades stopped out</b> — {ts["losses"]} of {ts["n"]} '
            f'hit the initial stop, while only {ts["wins"]} reached a profit. Average win '
            f'{signed_money(ts["avg_win"])} vs average loss {signed_money(ts["avg_loss"])}, so the edge has to '
            f'come from a higher hit rate or a wider target.'
        )
    if ms["go_scored"]:
        finding = (
            f'<b>{"Desk GO calls are not adding edge yet" if (ms["go_avg_r"] or 0) <= 0 else "Desk GO calls are working"}</b> '
            f'— {ms["go_n"]} GO verdicts scored, {ms["go_targets"]} target / {ms["go_stops"]} stop / '
            f'{ms["go_closes"]} close, sum {ms["go_sum_r"]:+.2f}R ({ms["go_avg_r"]:+.3f}R average).'
        )
        if ms["skip_n"]:
            finding += (
                f' The {ms["skip_n"]} SKIPs did {ms["skip_sum_r"]:+.1f}R ({ms["skip_avg_r"]:+.3f}R average) — '
            )
            finding += (
                "the vetoes are filtering out the weaker setups."
                if (ms["skip_avg_r"] or 0) < (ms["go_avg_r"] or 0)
                else "the skipped names did <i>better</i> than the GO names, so the veto logic is currently cutting the wrong side."
            )
        findings.append(finding)
    if ms["buckets"]:
        bands = sorted(
            [(band, b) for band, b in ms["buckets"].items() if b["rs"]],
            key=lambda kv: int(kv[0].split("-")[0]),
        )
        if bands:
            detail = ", ".join(
                f'{band} &rarr; {mean(b["rs"]):+.3f}R ({b["n"]} signals)' for band, b in bands
            )
            populated = [(band, b) for band, b in bands if b["n"] >= 10]
            inverted = (
                populated
                and mean(populated[0][1]["rs"]) > mean(populated[-1][1]["rs"])
            )
            findings.append(
                f'<b>Confidence ordering is {"inverted" if inverted else "worth reviewing"}</b> '
                f'— average outcome by band: {detail}. '
                f'{"Higher confidence is currently scoring worse, so it is acting as a veto threshold rather than a predictor." if inverted else "Check whether the bands separate outcomes at all."}'
            )
    if ts["charges"]:
        findings.append(
            f'<b>Charges are eating {money(ts["charges"])} </b> of gross P&L — every trade pays roughly the same '
            f'SEC fee (~$6.5), which is {pct(100 * ts["charges"] / abs(ts["net"]), 1) if ts["net"] else "—"} of net realised.'
        )
    findings_html = "".join(f'<div class="note">{f}</div>' for f in findings)

    # ---------------------------------------------------------- desk table
    def desk_row(h, idx, desk):
        notes = h.get("notes") or {}
        oc = h.get("outcome")
        oc_cls = {"TARGET": "pos", "STOP": "neg"}.get(oc or "", "dim")
        r = h.get("outcome_r")
        detail_id = f"d-{desk}-{idx}"
        note_html = "".join(
            f'<div style="margin-bottom:6px"><b style="color:#8b93ff">{k.title()}</b><br>{esc(v)}</div>'
            for k, v in notes.items() if v
        )
        trig = f'<div style="margin-bottom:6px"><b style="color:#8b93ff">Trigger</b><br>{esc(h.get("trigger"))}</div>' if h.get("trigger") else ""
        key = " ".join(str(x) for x in (
            (h.get("signal_time") or "")[:16], h.get("ticker"), h.get("action"),
            h.get("verdict"), h.get("confidence"), h.get("outcome") or "",
            desk,
        ))
        return (
            f'<tr class="click" data-detail="{detail_id}" data-key="{esc(key)}">'
            f'<td>{esc((h.get("signal_time") or "")[:16].replace("T", " "))}</td>'
            f'<td><b>{esc(h.get("ticker"))}</b></td>'
            f'<td class="{"buy" if h.get("action") == "BUY" else "sell"}">{esc(h.get("action"))}</td>'
            f'<td><span class="pill {"go" if h.get("verdict") == "GO" else "skip"}">{esc(h.get("verdict"))}</span></td>'
            f'<td>{esc(h.get("confidence"))}</td>'
            f'<td>{h.get("entry")}</td><td>{h.get("stop")}</td><td>{h.get("target")}</td>'
            f'<td>{esc(h.get("qty"))}</td>'
            f'<td class="{oc_cls}">{esc(oc or "—")}{f" ({r:+.1f}R)" if r is not None else ""}</td>'
            f'</tr>'
            f'<tr id="{detail_id}" class="hidden"><td colspan="10">{trig}{note_html}</td></tr>'
        )

    gemini_rows = "".join(desk_row(h, i, "g") for i, h in enumerate(sorted(
        gemini, key=lambda x: x.get("signal_time") or "", reverse=True)[:40]))
    mistral_rows = "".join(desk_row(h, i, "m") for i, h in enumerate(sorted(
        mistral, key=lambda x: x.get("signal_time") or "", reverse=True)[:60]))

    # ------------------------------------------------------ verdict cards
    def verdict_card(v, title):
        if not v or not v.get("verdicts"):
            return f'<div class="panel"><h3>{title}</h3><div class="empty">No verdict on file.</div></div>'
        row = v["verdicts"][0]
        notes = row.get("notes") or {}
        notes_html = "".join(
            f'<div style="margin-bottom:6px"><b style="color:#8b93ff">{k.title()}</b><br>{esc(val)}</div>'
            for k, val in notes.items() if val
        )
        return f"""
<div class="panel">
  <h3>{title}</h3>
  <div class="kv">
    <div>Signal</div><div><b>{esc(row.get('ticker'))}</b> · {esc(row.get('action'))} {esc(row.get('qty'))}
      @ {row.get('entry')} · SL {row.get('stop')} · TGT {row.get('target')}</div>
    <div>Verdict</div><div><span class="pill {"go" if row.get('verdict') == 'GO' else 'skip'}">{esc(row.get('verdict'))}</span>
      · confidence {esc(row.get('confidence'))} · judged in {esc(v.get('age_min'))} min</div>
    <div>SPY day</div><div class="{'pos' if (v.get('spy_day_pct') or 0) >= 0 else 'neg'}">{(v.get('spy_day_pct') or 0):+.2f}%</div>
    <div>Trigger</div><div>{esc(row.get('trigger') or '—')}</div>
  </div>
  <div style="margin-top:8px">{notes_html}</div>
</div>"""

    # --------------------------------------------------------- trade rows
    def trade_row(t, idx):
        detail_id = f"t-{idx}"
        pnl = t.get("net_pnl") or 0
        r = t.get("_r")
        events = "".join(
            f'<div>{esc((e.get("time") or "")[11:16])} — {esc(e.get("text"))}</div>'
            for e in (t.get("events") or [])
        )
        charges = (t.get("charges") or {}).get("total")
        key = " ".join(str(x) for x in (
            t["date"], t["time"], t["ticker"], t["action"], t["qty"],
            t.get("status") or "", f"{pnl:.2f}", "win" if pnl > 0 else "loss",
        ))
        return (
            f'<tr class="click" data-detail="{detail_id}" data-key="{esc(key)}">'
            f'<td>{esc(t["date"])}</td><td>{esc(t["time"][:8])}</td>'
            f'<td><b>{esc(t["ticker"])}</b></td>'
            f'<td class="{"buy" if t["action"] == "BUY" else "sell"}">{esc(t["action"])}</td>'
            f'<td>{esc(t["qty"])}</td>'
            f'<td>${t["entry"]:.2f}</td><td>${(t.get("initial_sl") or t.get("sl") or 0):.2f}</td>'
            f'<td>${(t.get("tgt") or 0):.2f}</td><td>${(t.get("exit_price") or t.get("last_price") or 0):.2f}</td>'
            f'<td>{esc(t.get("status"))}</td>'
            f'<td class="{"pos" if pnl >= 0 else "neg"}">{signed_money(pnl)}</td>'
            f'<td>{f"{r:+.2f}R" if r is not None else "—"}</td>'
            f'<td class="rat">{esc((t.get("rationale") or "")[:220])}…</td>'
            f'</tr>'
            f'<tr id="{detail_id}" class="hidden"><td colspan="13">'
            f'<div class="two"><div><b style="color:#8b93ff">Full rationale</b><br>{esc(t.get("rationale"))}</div>'
            f'<div><b style="color:#8b93ff">Event log</b>{events or "<br>—"}'
            f'<div style="margin-top:6px" class="dim">ATR {esc(t.get("atr"))} · charges {money(charges)} · '
            f'BE {esc(t.get("breakeven_price"))} ({"set" if t.get("breakeven_set") else "not set"})</div></div></div>'
            f'</td></tr>'
        )

    ordered = sorted(ts["trades"], key=lambda t: (t["date"], t["time"]), reverse=True)
    trade_rows = "".join(trade_row(t, i) for i, t in enumerate(ordered))
    todays_rows = "".join(
        trade_row(t, 1000 + i) for i, t in enumerate(
            [t for t in ordered if t["date"] == today])) or (
        '<tr><td colspan="13" class="empty">No trades closed today.</td></tr>')

    # ------------------------------------------------------------- signal
    sig_html = ""
    if latest and latest.get("candidates"):
        for c in latest["candidates"]:
            candles = c.get("candles_5m") or []
            ind = c.get("indicators") or {}
            sig_html += f"""
<div class="panel">
  <h3>Latest candidate · {esc(c.get('ticker'))} {esc(c.get('signal'))}</h3>
  <div class="kv">
    <div>Signal</div><div><b>{esc(latest.get('signal_id'))}</b> · {esc((latest.get('generated_at') or '')[:19])}</div>
    <div>Levels</div><div>LTP {esc(c.get('ltp'))} · SL {esc(c.get('stop'))} · TGT {esc(c.get('target'))} · qty {esc(c.get('qty'))}</div>
    <div>Indicators</div><div>RSI {esc(ind.get('rsi'))} · ATR {esc(ind.get('atr'))} · vol ratio {esc(ind.get('vol_ratio'))}</div>
    <div>Scout reason</div><div>{esc(c.get('scout_reason'))}</div>
    <div>SPY day</div><div class="{'pos' if (latest.get('spy_day_pct') or 0) >= 0 else 'neg'}">{(latest.get('spy_day_pct') or 0):+.2f}%</div>
  </div>
  {svg_candles(candles, c.get('ltp'), c.get('stop'), c.get('target'))}
</div>"""
    else:
        sig_html = '<div class="panel"><h3>Latest candidate</h3><div class="empty">signals/latest.json is empty.</div></div>'

    # ----------------------------------------------------------- earnings
    upcoming = sorted([d for d in earn_dates if d >= today])[:8]
    earn_html = "".join(
        f'<div class="card"><div class="lbl">{esc(d)}</div><div class="big" style="font-size:14px">'
        f'{esc(", ".join(earn_dates[d]))}</div></div>' for d in upcoming
    ) or '<div class="empty">No upcoming earnings on file.</div>'

    # -------------------------------------------------------------- scout
    scout_html = ""
    if sc:
        scout_html = f"""
<div class="panel">
  <h3>Shadow scout · Qwen vs DeepSeek</h3>
  <div class="kv">
    <div>Scans logged</div><div>{sc['scans']}</div>
    <div>Qwen picks</div><div>{sc['qwen_picks']}</div>
    <div>DeepSeek picks</div><div>{sc['deep_picks']}</div>
    <div>Agreement</div><div>{sc['overlap']} overlapping · {sc['qwen_only']} Qwen-only · {sc['deep_only']} DeepSeek-only
      → <b>{pct(sc['agreement'])}</b> agreement</div>
    <div>DeepSeek errors</div><div>{sc['errors']}</div>
    <div>Avg latency</div><div>{sc['avg_latency_ms']:.0f} ms</div>
  </div>
</div>"""

    # ---------------------------------------------------------- desk stats
    def desk_kpis(s, name):
        if not s["n"]:
            return f'<div class="panel"><h3>{name}</h3><div class="empty">No judgements on file.</div></div>'
        go_pct = 100.0 * s["go_n"] / s["n"]
        bucket_rows = []
        for band, b in s["buckets"].items():
            avg_r = mean(b["rs"]) if b["rs"] else None
            cell = (
                f'<td class="{"pos" if avg_r >= 0 else "neg"}">{avg_r:+.2f}R</td>'
                if avg_r is not None
                else '<td class="dim">—</td>'
            )
            bucket_rows.append(
                f'<tr><td>{esc(band)}</td><td>{b["n"]}</td><td>{b["go"]}</td>'
                f'<td>{b["target"]}/{b["stop"]}/{b["close"]}</td>{cell}</tr>'
            )
        buckets = "".join(bucket_rows)
        return f"""
<div class="panel">
  <h3>{name}</h3>
  <div class="grid">
    {kpi("Judged", str(s["n"]), f'{esc(s["first"] or "")[:10]} → {esc(s["last"] or "")[:10]}')}
    {kpi("GO rate", pct(go_pct), f'{s["go_n"]} of {s["n"]}')}
    {kpi("GO outcomes", f'{s["go_targets"]}T / {s["go_stops"]}S / {s["go_closes"]}C', "target / stop / close")}
    {kpi("GO sum R", f'{s["go_sum_r"]:+.1f}R' if s["go_sum_r"] is not None else "—",
         f'{s["go_avg_r"]:+.2f}R avg' if s["go_avg_r"] is not None else "",
         "pos" if (s["go_sum_r"] or 0) >= 0 else "neg")}
    {kpi("SKIP sum R", f'{s["skip_sum_r"]:+.1f}R' if s["skip_sum_r"] is not None else "—",
         f'{s["skip_avg_r"]:+.2f}R avg over {s["skip_n"]}' if s["skip_avg_r"] is not None else "",
         "pos" if (s["skip_sum_r"] or 0) >= 0 else "neg")}
    {kpi("Judge latency", f'{s["avg_latency_s"]:.0f}s' if s["avg_latency_s"] else "—", "signal → verdict")}
  </div>
  <table><thead><tr><th>Confidence</th><th>Signals</th><th>GO</th><th>T/S/C</th><th>Avg R</th></tr></thead>
  <tbody>{buckets}</tbody></table>
</div>"""

    # -------------------------------------------------------------- page
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="180">
<title>US Intraday Agent {esc(version)}</title>
<style>{CSS}</style></head><body>
<h1>US Intraday Agent <span class="ver">{esc(version)}</span>
<span class="badge {mode_cls}">{esc(mode)}</span></h1>
<div class="updated">Updated {esc(updated_label)} &middot; auto-refresh 3min &middot;
{ts['n']} closed trades &middot; {ms['n']} desk judgements</div>

{market_html}

<div class="grid">{cards_html}</div>

<div class="two">
  <div class="panel"><h3>Equity curve (closed trades, $100k start)</h3>{chart_curve}</div>
  <div class="panel"><h3>Daily P&amp;L</h3>{chart_daily}</div>
</div>
<div class="two">
  <div class="panel"><h3>P&amp;L by ticker</h3>{chart_ticker}</div>
  <div class="panel"><h3>What the numbers say</h3>{findings_html}</div>
</div>

<h2>Today</h2>
<table><thead><tr><th>Date</th><th>Time (ET)</th><th>Ticker</th><th>Action</th><th>Qty</th><th>Entry</th><th>SL</th>
<th>TGT</th><th>Exit</th><th>Status</th><th>Net P&amp;L</th><th>R</th><th>Rationale</th></tr></thead>
<tbody>{todays_rows}</tbody></table>

<h2>Trade History <span class="tag">click a row for the full rationale and event log</span></h2>
<div class="row-actions">
  <input id="q" class="toggle" placeholder="Filter ticker / status…" onkeyup="filterTable('q','hist')" style="min-width:220px">
</div>
<table id="hist"><thead><tr><th>Date</th><th>Time (ET)</th><th>Ticker</th><th>Action</th><th>Qty</th><th>Entry</th><th>SL</th>
<th>TGT</th><th>Exit</th><th>Status</th><th>Net P&amp;L</th><th>R</th><th>Rationale</th></tr></thead>
<tbody>{trade_rows}</tbody></table>

<h2>Assistant Desk <span class="tag">advisory only — these are vetoes on the Oracle's signals, not trades</span></h2>
<div class="two">{verdict_card(verdict, "Gemini desk · last verdict")}{verdict_card(verdict_m, "Mistral desk · last verdict")}</div>
<div class="two">{desk_kpis(ms, "Mistral desk scorecard")}{desk_kpis(gs, "Gemini desk scorecard")}</div>

<div class="two">
  <div class="panel"><h3>Mistral desk · last 60 judgements</h3>
  <table><thead><tr><th>Signal time</th><th>Ticker</th><th>Action</th><th>Verdict</th><th>Conf</th><th>Entry</th><th>SL</th><th>TGT</th><th>Qty</th><th>Outcome</th></tr></thead>
  <tbody>{mistral_rows or '<tr><td colspan="10" class="empty">No judgements.</td></tr>'}</tbody></table></div>
  <div class="panel"><h3>Gemini desk · last 40 judgements</h3>
  <table><thead><tr><th>Signal time</th><th>Ticker</th><th>Action</th><th>Verdict</th><th>Conf</th><th>Entry</th><th>SL</th><th>TGT</th><th>Qty</th><th>Outcome</th></tr></thead>
  <tbody>{gemini_rows or '<tr><td colspan="10" class="empty">No judgements.</td></tr>'}</tbody></table></div>
</div>

<h2>Live signal</h2>
{sig_html}

<h2>Scout &amp; calendar</h2>
<div class="two">
  <div class="panel"><h3>Upcoming earnings on watched names</h3>
  <div class="grid">{earn_html}</div></div>
  {scout_html}
</div>

<footer>
Generated by <code>build_dashboard.py</code> from this repo's own JSON — no external requests, no CDN.
Advisory / paper-trading output only, not investment advice.
</footer>
<script>{JS}</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="repo root (default: script's directory)")
    ap.add_argument("--out", default=None, help="output file (default: <repo>/index.html)")
    ap.add_argument("--layout", choices=("tiles", "tape"), default="tiles",
                    help="market strip layout: tiles (index cards + sparklines) or tape (one compact line)")
    args = ap.parse_args()

    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parent
    out = Path(args.out).resolve() if args.out else repo / "index.html"
    html_text = render_html(repo, layout=args.layout)
    out.write_text(html_text, encoding="utf-8")
    print(f"wrote {out} ({len(html_text):,} bytes)")


if __name__ == "__main__":
    main()
