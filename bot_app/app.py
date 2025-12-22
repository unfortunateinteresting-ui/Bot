#!/usr/bin/env python3
"""
Auto scalper with a *modern* local GUI + live-updating price chart.

- Uses ccxt for exchange access
- Uses ttkbootstrap (optional) to theme Tkinter with a modern look
- Embeds a Matplotlib chart that updates as you change the trading symbol/coin

WARNING:
- DRY_RUN = False with real API keys can place live orders.
- Use small sizes, and test in DRY_RUN first.
"""

import os
import sys
import time
import traceback
import datetime as dt
import json
import math
import statistics
from typing import Optional, List, Tuple, Dict, Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import threading
import queue

import pandas as pd
import ccxt

# --- GUI / chart deps ---
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

try:
    import ttkbootstrap as tb
    from ttkbootstrap.constants import *  # noqa: F401,F403
    try:
        # Preferred import path in newer ttkbootstrap versions
        from ttkbootstrap.widgets import ScrolledFrame as TBScrolledFrame  # type: ignore
    except Exception:
        # If this fails, just don't use the ttkbootstrap scrolled frame
        TBScrolledFrame = None
except Exception:
    tb = None  # fallback to plain ttk
    TBScrolledFrame = None

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates

from bot_app import config
from bot_app.config import *
from bot_app.exchange_utils import (
    fetch_best_book_price,
    fetch_ohlcv_df,
    init_exchange,
    simulate_dry_run_market_fill,
)
from bot_app.settings_io import load_settings, save_settings
from bot_app.state import AutopilotParam, BotState, OpenOrder, Position, log_event


COINEX_API_BASE = "https://api.coinex.com/v2"
COINEX_PUBLIC_TIMEOUT_SEC = 8.0
AUTO_COIN_NOTIONAL_USDT = 100.0
SMART_MARKET_FRICTION_BPS = 120.0
SMART_MARKET_NOTIONAL_USDT = 100.0
PRICE_SANITY_MAX_RATIO = 10.0


################################################################
# Legacy data structures moved to bot_app.state
################################################################

################################################################
# Settings helpers moved to bot_app.settings_io
################################################################


################################################################
# Exchange helpers moved to bot_app.exchange_utils
################################################################


# ============================================================
# ================== ORDER / PNL HELPERS =====================
# ============================================================


def simulate_dry_run_market_fill(exchange: ccxt.Exchange, symbol: str, side: str, qty_base: float,
                                 price_fallback: float) -> Tuple[float, float, float, float, int]:
    """Simulate a market fill from the current order book.

    - side: 'buy' uses asks, 'sell' uses bids
    - qty_base: amount of base asset to buy/sell
    Returns: (filled_qty, avg_price, best_price, worst_price, used_levels)
    """
    if qty_base <= 0:
        return 0.0, price_fallback, price_fallback, price_fallback, 0

    try:
        ob = exchange.fetch_order_book(symbol, limit=SLIPPAGE_ORDERBOOK_LEVELS)
        levels = (ob.get("asks") if side == "buy" else ob.get("bids")) or []

        remaining = float(qty_base)
        filled = 0.0
        cost = 0.0
        best = None
        worst = None
        used = 0

        for lvl in levels:
            if remaining <= 0:
                break
            if not lvl or len(lvl) < 2:
                continue
            price = float(lvl[0])
            amount = float(lvl[1])
            if amount <= 0:
                continue

            take = min(remaining, amount)
            if best is None:
                best = price
            worst = price

            cost += take * price
            filled += take
            remaining -= take
            used += 1

        # If the book is empty or unusable, fall back to last price estimate
        if filled <= 0:
            return qty_base, price_fallback, price_fallback, price_fallback, 0

        # If requested size is larger than available depth, approximate remainder at a worse price
        if remaining > 0:
            bps = SLIPPAGE_FALLBACK_BPS / 10000.0
            if worst is None:
                worst = price_fallback
            remainder_price = worst * (1 + bps) if side == "buy" else worst * (1 - bps)
            cost += remaining * remainder_price
            filled += remaining
            worst = remainder_price
            remaining = 0.0

        avg_price = cost / filled if filled > 0 else price_fallback
        if best is None:
            best = avg_price
        if worst is None:
            worst = avg_price
        return filled, avg_price, best, worst, used

    except Exception:
        # Network / exchange error: just use the estimate
        return qty_base, price_fallback, price_fallback, price_fallback, 0


def simulate_dry_run_ioc_fill(exchange: ccxt.Exchange, symbol: str, side: str, qty_base: float,
                              limit_price: float) -> Tuple[float, float, float, float, int]:
    """Simulate IOC limit fill; only consumes levels within cap."""
    if qty_base <= 0 or limit_price <= 0:
        return 0.0, limit_price, 0.0, 0.0, 0

    try:
        ob = exchange.fetch_order_book(symbol, limit=SLIPPAGE_ORDERBOOK_LEVELS)
        levels = (ob.get("asks") if side == "buy" else ob.get("bids")) or []

        remaining = float(qty_base)
        filled = 0.0
        cost = 0.0
        best = 0.0
        worst = 0.0
        used = 0

        for lvl in levels:
            if remaining <= 0:
                break
            if not lvl or len(lvl) < 2:
                continue
            price = float(lvl[0])
            amount = float(lvl[1])
            if amount <= 0 or price <= 0:
                continue

            if side == "buy" and price > limit_price:
                break
            if side == "sell" and price < limit_price:
                break

            take = min(remaining, amount)
            if best == 0.0:
                best = price
            worst = price

            cost += take * price
            filled += take
            remaining -= take
            used += 1

        if filled <= 0:
            return 0.0, limit_price, 0.0, 0.0, used

        avg_price = cost / filled if filled > 0 else limit_price
        return filled, avg_price, best or avg_price, worst or avg_price, used

    except Exception:
        return 0.0, limit_price, 0.0, 0.0, 0


def normalize_qty_for_market(exchange: ccxt.Exchange, price: float, qty: float) -> Tuple[float, float, float]:
    """Round qty to market precision and return (qty, min_qty, min_notional)."""
    market = {}
    try:
        market = exchange.markets.get(SYMBOL, {}) if hasattr(exchange, "markets") else {}
    except Exception:
        market = {}

    prec = None
    try:
        prec = market.get("precision", {}).get("amount")
    except Exception:
        prec = None

    if prec is not None:
        try:
            qty = float(round(qty, int(prec)))
        except Exception:
            pass

    limits = market.get("limits", {}) if isinstance(market, dict) else {}
    amount_min = None
    cost_min = None
    try:
        amount_min = limits.get("amount", {}).get("min")
        cost_min = limits.get("cost", {}).get("min")
    except Exception:
        pass

    min_notional = max(MIN_NOTIONAL_USDT, float(cost_min) if cost_min else MIN_NOTIONAL_USDT)
    min_qty = float(amount_min) if amount_min else (min_notional / max(price, 1e-9))
    return qty, min_qty, min_notional


def fetch_best_book_price(exchange: ccxt.Exchange, symbol: str, side: str, price_fallback: float) -> float:
    """Best available price just before a market order.

    - side='buy'  -> best ask
    - side='sell' -> best bid
    Falls back to ticker ask/bid/last or provided fallback.
    """
    try:
        ob = exchange.fetch_order_book(symbol, limit=5)
        if side == "buy":
            asks = ob.get("asks") or []
            if asks and asks[0] and len(asks[0]) >= 1:
                return float(asks[0][0])
        else:
            bids = ob.get("bids") or []
            if bids and bids[0] and len(bids[0]) >= 1:
                return float(bids[0][0])
    except Exception:
        pass

    try:
        t = exchange.fetch_ticker(symbol) or {}
        if side == "buy":
            v = t.get("ask") or t.get("last") or price_fallback
        else:
            v = t.get("bid") or t.get("last") or price_fallback
        return float(v or price_fallback)
    except Exception:
        return float(price_fallback)


def _get_ap(state: BotState, key: str, default: float) -> AutopilotParam:
    ap = state.autopilot.get(key)
    if not ap:
        ap = AutopilotParam(manual=default, auto=default, effective=default, mode="manual")
        state.autopilot[key] = ap
    return ap


def _ap_mode(state: BotState, key: str) -> str:
    try:
        return _get_ap(state, key, 0.0).mode
    except Exception:
        return "manual"


def _safe_float(val: Any) -> float:
    try:
        return float(val)
    except Exception:
        return 0.0


def _coerce_bool(val: Any, default: bool = True) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")


def _calc_spread_bps(bid: float, ask: float) -> float:
    if bid <= 0 or ask <= 0 or ask <= bid:
        return 0.0
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid * 10000.0


def _price_ratio_too_wide(a: float, b: float, max_ratio: float = PRICE_SANITY_MAX_RATIO) -> bool:
    if a <= 0 or b <= 0:
        return False
    ratio = max(a / b, b / a)
    return ratio > max_ratio


def _slip_bps_for_notional(levels: List[List[float]], notional_usdt: float, side: str) -> float:
    if not levels or notional_usdt <= 0:
        return 0.0
    try:
        best = _safe_float(levels[0][0])
    except Exception:
        return 0.0
    if best <= 0:
        return 0.0

    if side == "buy":
        remaining = float(notional_usdt)
        filled_base = 0.0
        cost = 0.0
        worst = best
        for price, amount in levels:
            if remaining <= 0:
                break
            price = _safe_float(price)
            amount = _safe_float(amount)
            if price <= 0 or amount <= 0:
                continue
            level_cost = price * amount
            take_cost = min(remaining, level_cost)
            take_base = take_cost / price
            cost += take_cost
            filled_base += take_base
            remaining -= take_cost
            worst = price
        if remaining > 0:
            bps = SLIPPAGE_FALLBACK_BPS / 10000.0
            remainder_price = worst * (1 + bps)
            take_base = remaining / remainder_price if remainder_price > 0 else 0.0
            cost += remaining
            filled_base += take_base
            remaining = 0.0
        avg = cost / filled_base if filled_base > 0 else best
        slip_bps = (avg / best - 1.0) * 10000.0
        return max(0.0, slip_bps)

    # sell
    base_qty = notional_usdt / best if best > 0 else 0.0
    remaining = base_qty
    proceeds = 0.0
    worst = best
    for price, amount in levels:
        if remaining <= 0:
            break
        price = _safe_float(price)
        amount = _safe_float(amount)
        if price <= 0 or amount <= 0:
            continue
        take = min(remaining, amount)
        proceeds += take * price
        remaining -= take
        worst = price
    if remaining > 0:
        bps = SLIPPAGE_FALLBACK_BPS / 10000.0
        remainder_price = worst * (1 - bps)
        proceeds += remaining * remainder_price
        remaining = 0.0
    avg = proceeds / base_qty if base_qty > 0 else best
    slip_bps = (1.0 - avg / best) * 10000.0
    return max(0.0, slip_bps)


def _get_book_metrics(exchange: ccxt.Exchange, symbol: str, notional_usdt: float) -> Optional[Dict[str, float]]:
    try:
        ob = exchange.fetch_order_book(symbol, limit=20)
    except Exception:
        return None
    asks = ob.get("asks") or []
    bids = ob.get("bids") or []
    if not asks or not bids:
        return None
    best_ask = _safe_float(asks[0][0])
    best_bid = _safe_float(bids[0][0])
    spread_bps = _calc_spread_bps(best_bid, best_ask)
    slip_buy = _slip_bps_for_notional(asks, notional_usdt, "buy")
    slip_sell = _slip_bps_for_notional(bids, notional_usdt, "sell")
    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread_bps": spread_bps,
        "slip_bps_buy": slip_buy,
        "slip_bps_sell": slip_sell,
    }


def _coinex_public_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = params or {}
    url = COINEX_API_BASE + path
    if params:
        url += "?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "bot"})
    try:
        with urlopen(req, timeout=COINEX_PUBLIC_TIMEOUT_SEC) as resp:
            payload = json.load(resp) or {}
    except Exception:
        return {}
    if isinstance(payload, dict):
        code = payload.get("code")
        if code not in (0, "0", None):
            return {}
    return payload


def _coinex_extract_data(payload: Dict[str, Any]) -> Any:
    if isinstance(payload, dict) and "data" in payload:
        return payload.get("data")
    return payload


def _coinex_fetch_markets() -> List[Dict[str, Any]]:
    payload = _coinex_public_get("/spot/market")
    data = _coinex_extract_data(payload)
    if isinstance(data, dict):
        markets = data.get("markets") or data.get("market") or data.get("data") or []
    else:
        markets = data or []
    return markets if isinstance(markets, list) else []


def _coinex_fetch_tickers() -> List[Dict[str, Any]]:
    payload = _coinex_public_get("/spot/ticker")
    data = _coinex_extract_data(payload)
    if isinstance(data, dict):
        raw = data.get("ticker") or data.get("data") or data
    else:
        raw = data or []
    tickers: List[Dict[str, Any]] = []
    if isinstance(raw, dict):
        for market, info in raw.items():
            if not isinstance(info, dict):
                continue
            row = dict(info)
            row.setdefault("market", market)
            tickers.append(row)
        return tickers
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                tickers.append(item)
    return tickers


def _ccxt_fetch_tickers(exchange: ccxt.Exchange) -> List[Dict[str, Any]]:
    try:
        raw = exchange.fetch_tickers() or {}
    except Exception:
        return []
    tickers: List[Dict[str, Any]] = []
    if isinstance(raw, dict):
        for sym, info in raw.items():
            if not isinstance(info, dict):
                continue
            row = dict(info)
            row.setdefault("market", _symbol_to_market(sym))
            tickers.append(row)
        return tickers
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            market = row.get("market") or row.get("symbol") or ""
            if market:
                row["market"] = _symbol_to_market(str(market))
            tickers.append(row)
    return tickers


def _coinex_fetch_depth(market: str, limit: int = 20) -> Tuple[List[List[float]], List[List[float]]]:
    market = _symbol_to_market(market)
    payload = _coinex_public_get("/spot/depth", {"market": market, "limit": limit, "interval": "0"})
    data = _coinex_extract_data(payload)
    if isinstance(data, dict):
        depth = data.get("depth") if isinstance(data.get("depth"), dict) else data
        asks = depth.get("asks") or depth.get("sell") or []
        bids = depth.get("bids") or depth.get("buy") or []
    else:
        asks, bids = [], []
    clean_asks: List[List[float]] = []
    for row in asks or []:
        if not row or len(row) < 2:
            continue
        price = _safe_float(row[0])
        amount = _safe_float(row[1])
        if price > 0 and amount > 0:
            clean_asks.append([price, amount])

    clean_bids: List[List[float]] = []
    for row in bids or []:
        if not row or len(row) < 2:
            continue
        price = _safe_float(row[0])
        amount = _safe_float(row[1])
        if price > 0 and amount > 0:
            clean_bids.append([price, amount])
    return clean_asks, clean_bids


def _coinex_fetch_kline(market: str, limit: int = 120) -> List[Any]:
    market = _symbol_to_market(market)
    payload = _coinex_public_get("/spot/kline", {"market": market, "period": "1min", "limit": int(limit)})
    data = _coinex_extract_data(payload)
    if isinstance(data, dict):
        rows = data.get("candles") or data.get("data") or data.get("kline") or []
    else:
        rows = data or []
    return rows if isinstance(rows, list) else []


def _parse_kline_row(row: Any) -> Optional[Tuple[float, float, float, float]]:
    if isinstance(row, dict):
        o = _safe_float(row.get("open") or row.get("open_price"))
        high = _safe_float(row.get("high") or row.get("high_price"))
        low = _safe_float(row.get("low") or row.get("low_price"))
        close = _safe_float(row.get("close") or row.get("close_price"))
        if close <= 0 or high <= 0 or low <= 0:
            return None
        return o, high, low, close
    if not isinstance(row, (list, tuple)) or len(row) < 5:
        return None
    o = _safe_float(row[1])
    c1 = _safe_float(row[2])
    c2 = _safe_float(row[3])
    c3 = _safe_float(row[4])

    # Format A: [ts, open, high, low, close]
    high_a, low_a, close_a = c1, c2, c3
    ok_a = high_a >= max(o, close_a) and low_a <= min(o, close_a) and high_a >= low_a

    # Format B: [ts, open, close, high, low]
    close_b, high_b, low_b = c1, c2, c3
    ok_b = high_b >= max(o, close_b) and low_b <= min(o, close_b) and high_b >= low_b

    if ok_b and not ok_a:
        return o, high_b, low_b, close_b
    if ok_a and not ok_b:
        return o, high_a, low_a, close_a
    if ok_b and ok_a:
        return o, high_b, low_b, close_b
    return o, high_b, low_b, close_b


def _atr_pct_from_kline(rows: List[Any], lookback: int = 60) -> Tuple[float, float, float]:
    ranges = []
    for row in rows[-lookback:]:
        parsed = _parse_kline_row(row)
        if not parsed:
            continue
        _o, high, low, close = parsed
        if close <= 0:
            continue
        ranges.append((high - low) / close * 100.0)
    if not ranges:
        return 0.0, 0.0, 0.0
    avg_range = statistics.mean(ranges)
    med_range = statistics.median(ranges)
    max_range = max(ranges)
    return float(avg_range), float(med_range), float(max_range)


def _symbol_to_market(symbol: str) -> str:
    return symbol.replace("/", "").upper()


def _market_to_symbol(market: str) -> str:
    market = market.upper()
    if "/" in market:
        return market
    if market.endswith("USDT"):
        return f"{market[:-4]}/USDT"
    return market


def _build_ticker_items(tickers: List[Dict[str, Any]], allowed_markets: Optional[set]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    items: List[Dict[str, Any]] = []
    ticker_map: Dict[str, Dict[str, Any]] = {}
    for t in tickers:
        info = t.get("info") if isinstance(t, dict) else None
        info = info if isinstance(info, dict) else {}
        market = str(t.get("market") or t.get("symbol") or t.get("market_name") or "").upper()
        if not market or not market.endswith("USDT"):
            continue
        if allowed_markets is not None and market not in allowed_markets:
            continue

        last = _safe_float(
            t.get("last") or t.get("close") or t.get("price") or
            info.get("last") or info.get("last_price") or info.get("close") or info.get("close_price")
        )
        open_ = _safe_float(
            t.get("open") or t.get("open_24h") or t.get("open_price") or
            info.get("open") or info.get("open_price") or info.get("open_24h")
        )
        value = _safe_float(
            t.get("value") or t.get("quote_volume") or t.get("quoteVolume") or
            info.get("value") or info.get("quote_volume") or info.get("quoteVolume")
        )
        if value <= 0:
            base_vol = _safe_float(
                t.get("baseVolume") or t.get("volume") or t.get("vol") or t.get("amount") or
                info.get("baseVolume") or info.get("volume") or info.get("vol") or info.get("amount")
            )
            if base_vol > 0 and last > 0:
                value = base_vol * last
        change = (last / open_ - 1.0) if open_ > 0 else 0.0
        row = {
            "market": market,
            "last": last,
            "open": open_,
            "value": value,
            "change": change,
        }
        items.append(row)
        ticker_map[market] = row
    return items, ticker_map


def _rank_trending_candidates(items: List[Dict[str, Any]], min_value: float, candidates_n: int) -> List[Dict[str, Any]]:
    filtered = [x for x in items if x.get("value", 0.0) >= min_value]
    if not filtered:
        filtered = items
    if not filtered:
        return []

    by_value = sorted(filtered, key=lambda x: x.get("value", 0.0), reverse=True)
    by_change = sorted(filtered, key=lambda x: abs(x.get("change", 0.0)), reverse=True)
    for idx, item in enumerate(by_value):
        item["_rank_value"] = idx
    for idx, item in enumerate(by_change):
        item["_rank_change"] = idx
    for item in filtered:
        item["_rank_sum"] = int(item.get("_rank_value", 0)) + int(item.get("_rank_change", 0))
    ranked = sorted(filtered, key=lambda x: (x.get("_rank_sum", 0), -x.get("value", 0.0)))
    return ranked[:max(1, candidates_n)]


def _filter_tradable(candidates: List[Dict[str, Any]], min_value: float, spread_max: float,
                     slip_max: float) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[Dict[str, Any]], List[Dict[str, Any]]]:
    survivors: List[Dict[str, Any]] = []
    counts = {"min_value": 0, "depth_fail": 0, "spread": 0, "slippage": 0}
    depth_samples: List[Dict[str, Any]] = []
    candidate_metrics: List[Dict[str, Any]] = []
    for c in candidates:
        if c.get("value", 0.0) < min_value:
            counts["min_value"] += 1
            continue
        market = c.get("market") or ""
        asks, bids = _coinex_fetch_depth(market, limit=20)
        if not asks or not bids:
            counts["depth_fail"] += 1
            continue
        best_ask = _safe_float(asks[0][0])
        best_bid = _safe_float(bids[0][0])
        spread_bps = _calc_spread_bps(best_bid, best_ask)
        slip_buy = _slip_bps_for_notional(asks, AUTO_COIN_NOTIONAL_USDT, "buy")
        slip_sell = _slip_bps_for_notional(bids, AUTO_COIN_NOTIONAL_USDT, "sell")
        if len(depth_samples) < 3:
            depth_samples.append({
                "market": str(market),
                "spread_bps": spread_bps,
                "slip_bps_buy_100": slip_buy,
            })
        c_metric = dict(c)
        c_metric["spread_bps"] = spread_bps
        c_metric["slip_bps_buy_100"] = slip_buy
        c_metric["slip_bps_sell_100"] = slip_sell
        candidate_metrics.append(c_metric)
        if spread_bps > spread_max:
            counts["spread"] += 1
            continue
        if slip_buy > slip_max:
            counts["slippage"] += 1
            continue
        survivors.append(c_metric)
    return survivors, counts, depth_samples, candidate_metrics


def _score_candidate(c: Dict[str, Any], atr_pct: float, pump_penalty: bool) -> float:
    value = max(1.0, float(c.get("value", 0.0)))
    spread_bps = float(c.get("spread_bps", 0.0))
    slip_bps = float(c.get("slip_bps_buy_100", 0.0))
    score = (
        AUTO_COIN_SCORE_ATR_WEIGHT * atr_pct
        + AUTO_COIN_SCORE_VALUE_WEIGHT * math.log10(value)
        - AUTO_COIN_SCORE_FRICTION_WEIGHT * (spread_bps + slip_bps)
    )
    if pump_penalty:
        score *= 0.7
    return score


def _score_single_market(market: str, ticker: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not market:
        return None
    asks, bids = _coinex_fetch_depth(market, limit=20)
    if not asks or not bids:
        return None
    best_ask = _safe_float(asks[0][0])
    best_bid = _safe_float(bids[0][0])
    spread_bps = _calc_spread_bps(best_bid, best_ask)
    slip_buy = _slip_bps_for_notional(asks, AUTO_COIN_NOTIONAL_USDT, "buy")
    slip_sell = _slip_bps_for_notional(bids, AUTO_COIN_NOTIONAL_USDT, "sell")

    rows = _coinex_fetch_kline(market, limit=120)
    atr_pct, med_range, max_range = _atr_pct_from_kline(rows, lookback=60)
    pump_penalty = med_range > 0 and max_range > (4.0 * med_range)

    value = float(ticker.get("value", 0.0)) if ticker else 0.0
    candidate = {
        "market": market,
        "value": value,
        "spread_bps": spread_bps,
        "slip_bps_buy_100": slip_buy,
        "slip_bps_sell_100": slip_sell,
        "atr_pct": atr_pct,
    }
    candidate["score"] = _score_candidate(candidate, atr_pct, pump_penalty)
    return candidate


def _scan_coinex_trending(state: BotState, exchange: Optional[ccxt.Exchange] = None) -> Dict[str, Any]:
    markets = _coinex_fetch_markets()
    allowed_markets = None
    if markets:
        allowed_markets = set()
        for item in markets:
            if not isinstance(item, dict):
                continue
            market = str(item.get("market") or item.get("symbol") or "").upper()
            if not market:
                continue
            api_ok = _coerce_bool(item.get("is_api_trading_available"), default=True)
            status = str(item.get("status") or item.get("state") or "online").lower()
            if api_ok and (not status or status == "online"):
                allowed_markets.add(market)
        if not allowed_markets:
            allowed_markets = None

    tickers = _coinex_fetch_tickers()
    ticker_source = "coinex"
    if not tickers and exchange is not None:
        tickers = _ccxt_fetch_tickers(exchange)
        ticker_source = "ccxt"
    items, ticker_map = _build_ticker_items(tickers, allowed_markets)
    if not items:
        return {
            "scored": [],
            "ui_candidates": [],
            "best": None,
            "ticker_map": ticker_map,
            "relax_next": state.auto_coin_relax_factor,
            "ticker_source": ticker_source,
            "tickers_count": len(tickers),
            "items_count": 0,
            "candidates_count": 0,
            "survivors_count": 0,
            "scored_count": 0,
            "reject_min_value": 0,
            "reject_depth_fail": 0,
            "reject_spread": 0,
            "reject_slippage": 0,
            "reject_min_value_candidates": 0,
            "kline_fail": 0,
            "depth_samples": [],
        }

    candidates_n = max(5, int(getattr(state, "auto_coin_candidates_n", AUTO_COIN_CANDIDATES_N_DEFAULT) or AUTO_COIN_CANDIDATES_N_DEFAULT))
    base_min_value = AUTO_COIN_MIN_VALUE_USDT
    if sum(1 for x in items if x.get("value", 0.0) >= base_min_value) < max(5, candidates_n):
        base_min_value = AUTO_COIN_MIN_VALUE_FALLBACK_USDT

    relax = max(1.0, float(getattr(state, "auto_coin_relax_factor", 1.0)))
    min_value = base_min_value / relax
    spread_max = AUTO_COIN_SPREAD_BPS_MAX * relax
    slip_max = AUTO_COIN_SLIP_BPS_MAX * relax

    value_candidates = [x for x in items if x.get("value", 0.0) >= min_value]
    value_reject = (len(items) - len(value_candidates)) if value_candidates else 0

    candidates = _rank_trending_candidates(items, min_value, candidates_n)
    survivors, reject_counts, depth_samples, candidate_metrics = _filter_tradable(candidates, min_value, spread_max, slip_max)
    fallback_used = False

    if not survivors and candidates:
        # Immediate relax inside the same scan to avoid 0 survivors.
        relaxed_min_value = max(0.0, min_value * 0.5)
        relaxed_spread_max = spread_max * 1.5
        relaxed_slip_max = slip_max * 1.5
        survivors, _reject2, _depth2, candidate_metrics = _filter_tradable(
            candidates, relaxed_min_value, relaxed_spread_max, relaxed_slip_max
        )
        if survivors:
            fallback_used = True

    if not survivors and candidate_metrics:
        # Final fallback: accept depth-ok candidates even if they exceed caps.
        survivors = candidate_metrics
        fallback_used = True

    if len(survivors) < 3:
        relax_next = min(relax * 1.25, 3.0)
    else:
        relax_next = 1.0

    scored: List[Dict[str, Any]] = []
    kline_fail = 0
    survivors_sorted = sorted(survivors, key=lambda x: x.get("value", 0.0), reverse=True)
    for c in survivors_sorted[:10]:
        rows = _coinex_fetch_kline(c.get("market") or "", limit=120)
        if not rows:
            kline_fail += 1
        atr_pct, med_range, max_range = _atr_pct_from_kline(rows, lookback=60)
        pump_penalty = med_range > 0 and max_range > (4.0 * med_range)
        c = dict(c)
        c["atr_pct"] = atr_pct
        c["score"] = _score_candidate(c, atr_pct, pump_penalty)
        scored.append(c)

    scored = sorted(scored, key=lambda x: x.get("score", 0.0), reverse=True)
    atr_min = float(AUTO_COIN_MIN_ATR_PCT)
    atr_filtered = [c for c in scored if float(c.get("atr_pct", 0.0)) >= atr_min] if atr_min > 0 else []
    atr_filter_used = False
    if atr_min > 0 and atr_filtered:
        scored = atr_filtered
        atr_filter_used = True
    best = scored[0] if scored else None
    ui_candidates: List[Dict[str, Any]] = scored[:5]
    if not ui_candidates and candidate_metrics:
        for c in candidate_metrics[:5]:
            rows = _coinex_fetch_kline(c.get("market") or "", limit=120)
            if not rows:
                kline_fail += 1
            atr_pct, med_range, max_range = _atr_pct_from_kline(rows, lookback=60)
            pump_penalty = med_range > 0 and max_range > (4.0 * med_range)
            c = dict(c)
            c["atr_pct"] = atr_pct
            c["score"] = _score_candidate(c, atr_pct, pump_penalty)
            ui_candidates.append(c)
    if not ui_candidates and candidates:
        for c in candidates[:5]:
            c = dict(c)
            c.setdefault("spread_bps", 0.0)
            c.setdefault("slip_bps_buy_100", 0.0)
            c.setdefault("slip_bps_sell_100", 0.0)
            c.setdefault("atr_pct", 0.0)
            c.setdefault("score", 0.0)
            ui_candidates.append(c)
    return {
        "scored": scored,
        "ui_candidates": ui_candidates,
        "best": best,
        "ticker_map": ticker_map,
        "relax_next": relax_next,
        "atr_min": atr_min,
        "atr_filter_used": atr_filter_used,
        "tickers_count": len(tickers),
        "items_count": len(items),
        "candidates_count": len(candidates),
        "survivors_count": len(survivors),
        "scored_count": len(scored),
        "ticker_source": ticker_source,
        "reject_min_value": int(value_reject),
        "reject_depth_fail": int(reject_counts.get("depth_fail", 0)),
        "reject_spread": int(reject_counts.get("spread", 0)),
        "reject_slippage": int(reject_counts.get("slippage", 0)),
        "reject_min_value_candidates": int(reject_counts.get("min_value", 0)),
        "kline_fail": int(kline_fail),
        "depth_samples": depth_samples,
        "fallback_used": bool(fallback_used),
    }


def _format_usdt_value(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def _fmt_duration(seconds: float) -> str:
    try:
        seconds = float(seconds)
    except Exception:
        return "-"
    if seconds <= 0:
        return "0s"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def place_market_buy(exchange: ccxt.Exchange, state: BotState, cost_usdt: float, price_estimate: float,
                     now: dt.datetime, entry_index: int) -> Optional[Position]:
    if cost_usdt <= 0:
        return None
    symbol = SYMBOL
    use_cap_setting = bool(getattr(state, "use_ioc_slippage_cap", USE_IOC_SLIPPAGE_CAP_DEFAULT))
    max_slip_pct = max(0.01, float(getattr(state, "max_slip_pct", MAX_SLIP_PCT_DEFAULT)))
    force_ioc = (not DRY_RUN and EXCHANGE_NAME.lower() == "coinex")
    use_cap = True if force_ioc else use_cap_setting
    book = None
    if use_cap_setting and not force_ioc:
        book = _get_book_metrics(exchange, symbol, SMART_MARKET_NOTIONAL_USDT)
        if book:
            friction_bps = float(book.get("spread_bps", 0.0) + book.get("slip_bps_buy", 0.0))
            if friction_bps <= SMART_MARKET_FRICTION_BPS:
                use_cap = False
    best_pre = (book or {}).get("best_ask") or fetch_best_book_price(exchange, symbol, "buy", price_estimate)
    limit_price = (best_pre or price_estimate) * (1 + max_slip_pct / 100.0) if use_cap else price_estimate
    if limit_price <= 0:
        return None
    qty = cost_usdt / limit_price
    qty, min_qty, min_notional = normalize_qty_for_market(exchange, limit_price, qty)
    if qty <= 0 or qty < min_qty or qty * limit_price < min_notional:
        log_event(state, f"BUY skipped: qty {qty:.8f} below min (min_qty={min_qty:.8f}, min_notional={min_notional:.2f}).")
        return None
    attempt = 0
    filled_qty = 0.0
    fill_price = price_estimate

    while attempt < (2 if use_cap else 1) and filled_qty <= 0:
        attempt += 1
        if DRY_RUN:
            if SIMULATE_SLIPPAGE_DRY_RUN:
                if use_cap:
                    filled_qty, fill_price, best, worst, used = simulate_dry_run_ioc_fill(
                        exchange, symbol, "buy", qty, limit_price
                    )
                else:
                    filled_qty, fill_price, best, worst, used = simulate_dry_run_market_fill(
                        exchange, symbol, "buy", qty, limit_price
                    )
                slip_bps = 0.0
                if best and best > 0 and filled_qty > 0:
                    slip_bps = (fill_price / best - 1.0) * 10000.0
                try:
                    with state.lock:
                        state.last_slippage_bps = float(slip_bps)
                        state.last_slippage_side = "BUY"
                        state.last_slippage_best = float(best or 0.0)
                        state.last_slippage_fill = float(fill_price)
                        state.last_slippage_levels = int(used or 0)
                        state.last_slippage_ts = time.time()
                except Exception:
                    pass
                label = "IOC" if use_cap else "MARKET"
                log_event(
                    state,
                    f"{label} BUY (DRY_RUN) {filled_qty:.6f} {symbol.split('/')[0]} @ {fill_price:.8f} "
                    f"(cap={limit_price:.8f}, best={best:.8f}, worst={worst:.8f}, slip={slip_bps:.1f} bps, lvls={used}, attempt={attempt})"
                )
            else:
                fill_price = limit_price
                filled_qty = qty
                try:
                    with state.lock:
                        state.last_slippage_bps = 0.0
                        state.last_slippage_side = "BUY"
                        state.last_slippage_best = float(limit_price or 0.0)
                        state.last_slippage_fill = float(fill_price)
                        state.last_slippage_levels = 0
                        state.last_slippage_ts = time.time()
                except Exception:
                    pass
                label = "IOC" if use_cap else "MARKET"
                log_event(state, f"{label} BUY (DRY_RUN) {filled_qty:.6f} {symbol.split('/')[0]} @ {fill_price:.8f} (cap={limit_price:.8f}, attempt={attempt})")
        else:
            try:
                params = {"timeInForce": "IOC"} if use_cap else {}
                order = exchange.create_order(symbol, "limit" if use_cap else "market", "buy", qty, limit_price, params)
                filled_qty = float(order.get("filled") or order.get("amount") or 0.0)
                fill_price = float(order.get("average") or order.get("price") or limit_price)
                slip_bps = 0.0
                if best_pre and best_pre > 0 and filled_qty > 0:
                    slip_bps = (fill_price / best_pre - 1.0) * 10000.0
                try:
                    with state.lock:
                        state.last_slippage_bps = float(slip_bps)
                        state.last_slippage_side = "BUY"
                        state.last_slippage_best = float(best_pre or 0.0)
                        state.last_slippage_fill = float(fill_price)
                        state.last_slippage_levels = 0
                        state.last_slippage_ts = time.time()
                except Exception:
                    pass
                log_event(
                    state,
                    f"{'IOC' if use_cap else 'MARKET'} BUY filled {filled_qty:.6f} {symbol.split('/')[0]} @ {fill_price:.8f} "
                    f"(best={best_pre:.8f}, cap={limit_price:.8f}, slip={slip_bps:.1f} bps, attempt={attempt})"
                )
            except Exception as e:
                log_event(state, f"[ERROR] BUY attempt {attempt} error: {e}")
                filled_qty = 0.0

        if filled_qty <= 0 and use_cap and attempt == 1:
            # widen cap by 50% for one retry; last resort fallback to market after loop
            limit_price = (best_pre or price_estimate) * (1 + (max_slip_pct * 1.5) / 100.0)
            qty = cost_usdt / limit_price
            qty, min_qty, min_notional = normalize_qty_for_market(exchange, limit_price, qty)
            if qty <= 0 or qty < min_qty or qty * limit_price < min_notional:
                log_event(state, f"BUY retry skipped: qty {qty:.8f} below min (min_qty={min_qty:.8f}, min_notional={min_notional:.2f}).")
                return None
            continue

    if filled_qty <= 0 and use_cap:
        if DRY_RUN:
            log_event(state, "IOC BUY (DRY_RUN) unfilled after retry; no market fallback in DRY_RUN.")
            return None
        if force_ioc:
            log_event(state, "IOC BUY unfilled after retry; market fallback disabled for CoinEx.")
            return None
        # final fallback: plain market to avoid missing critical fills
        try:
            order = exchange.create_order(symbol, "market", "buy", qty)
            filled_qty = float(order.get("filled") or order.get("amount") or qty)
            fill_price = float(order.get("average") or order.get("price") or limit_price)
            slip_bps = 0.0
            if best_pre and best_pre > 0 and filled_qty > 0:
                slip_bps = (fill_price / best_pre - 1.0) * 10000.0
            try:
                with state.lock:
                    state.last_slippage_bps = float(slip_bps)
                    state.last_slippage_side = "BUY"
                    state.last_slippage_best = float(best_pre or 0.0)
                    state.last_slippage_fill = float(fill_price)
                    state.last_slippage_levels = 0
                    state.last_slippage_ts = time.time()
            except Exception:
                pass
            log_event(state, f"MARKET BUY fallback filled {filled_qty:.6f} {symbol.split('/')[0]} @ {fill_price:.8f} (best={best_pre:.8f}, slip={slip_bps:.1f} bps)")
        except Exception as e:
            log_event(state, f"[ERROR] Market BUY fallback error: {e}")
            return None

    if DRY_RUN and filled_qty > 0 and _price_ratio_too_wide(fill_price, price_estimate):
        log_event(
            state,
            f"[WARN] DRY_RUN BUY ignored: fill {fill_price:.8f} vs last {price_estimate:.8f} exceeds sanity ratio."
        )
        return None

    cost_notional = filled_qty * fill_price
    entry_fee = cost_notional * TAKER_FEE

    if DRY_RUN:
        with state.lock:
            state.sim_balance_usdt -= (cost_notional + entry_fee)
            state.sim_base_qty += filled_qty

    return Position(
        symbol=symbol,
        entry_price=fill_price,
        qty=filled_qty,
        initial_qty=filled_qty,
        entry_time=now,
        entry_index=entry_index,
        entry_fee=entry_fee,
        peak_price=fill_price,
        trail_armed=False,
        trail_pct_used=float(getattr(state, "effective_trail_pct", TRAIL_STOP_PCT)),
        stop_pct_used=float(getattr(state, "stop_loss_effective_pct", STOP_LOSS_PCT)) if (getattr(state, "stop_loss_effective_pct", STOP_LOSS_PCT) is not None) else None,
        hard_stop_mult_used=float(getattr(state, "effective_hard_stop_mult", HARD_STOP_MULT_DEFAULT)),
        soft_confirms_used=int(getattr(state, "soft_stop_confirms", SOFT_STOP_CONFIRMS_DEFAULT)),
    )


def place_market_sell(exchange: ccxt.Exchange, state: BotState, position: Position, qty_exit: float,
                      price_estimate: float) -> Tuple[float, float]:
    if qty_exit <= 0:
        return 0.0, price_estimate
    symbol = position.symbol
    use_cap_setting = bool(getattr(state, "use_ioc_slippage_cap", USE_IOC_SLIPPAGE_CAP_DEFAULT))
    use_cap = use_cap_setting
    max_slip_pct = max(0.01, float(getattr(state, "max_slip_pct", MAX_SLIP_PCT_DEFAULT)))
    book = None
    if use_cap_setting:
        book = _get_book_metrics(exchange, symbol, SMART_MARKET_NOTIONAL_USDT)
        if book:
            friction_bps = float(book.get("spread_bps", 0.0) + book.get("slip_bps_sell", 0.0))
            if friction_bps <= SMART_MARKET_FRICTION_BPS:
                use_cap = False
    qty_exit, min_qty, min_notional = normalize_qty_for_market(exchange, price_estimate, qty_exit)
    if qty_exit <= 0:
        log_event(state, f"[WARN] Market SELL skipped: qty normalized to 0 (min_qty={min_qty:.8f}).")
        return 0.0, price_estimate

    if DRY_RUN:
        if SIMULATE_SLIPPAGE_DRY_RUN:
            if use_cap:
                best_bid = fetch_best_book_price(exchange, symbol, "sell", price_estimate)
                limit_price = (best_bid or price_estimate) * (1 - max_slip_pct / 100.0)
                filled_qty, fill_price, best, worst, used = simulate_dry_run_ioc_fill(
                    exchange, symbol, "sell", qty_exit, limit_price
                )
            else:
                filled_qty, fill_price, best, worst, used = simulate_dry_run_market_fill(
                    exchange, symbol, "sell", qty_exit, price_estimate
                )
            slip_bps = 0.0
            if best and best > 0 and filled_qty > 0:
                # for sells, worse fill is LOWER than best bid => positive bps
                slip_bps = (1.0 - fill_price / best) * 10000.0
            if _price_ratio_too_wide(fill_price, price_estimate):
                log_event(
                    state,
                    f"[WARN] DRY_RUN SELL ignored: fill {fill_price:.8f} vs last {price_estimate:.8f} exceeds sanity ratio."
                )
                return 0.0, price_estimate
            try:
                with state.lock:
                    state.last_slippage_bps = float(slip_bps)
                    state.last_slippage_side = "SELL"
                    state.last_slippage_best = float(best or 0.0)
                    state.last_slippage_fill = float(fill_price)
                    state.last_slippage_levels = int(used or 0)
                    state.last_slippage_ts = time.time()
            except Exception:
                pass
            label = "IOC" if use_cap else "MARKET"
            log_event(
                state,
                f"{label} SELL (DRY_RUN) {filled_qty:.6f} {symbol.split('/')[0]} @ {fill_price:.8f} "
                f"(book best={best:.8f}, worst={worst:.8f}, slip={slip_bps:.1f} bps, lvls={used})"
            )
            return filled_qty, fill_price
        else:
            fill_price = price_estimate
            filled_qty = qty_exit
            try:
                with state.lock:
                    state.last_slippage_bps = 0.0
                    state.last_slippage_side = "SELL"
                    state.last_slippage_best = float(price_estimate or 0.0)
                    state.last_slippage_fill = float(fill_price)
                    state.last_slippage_levels = 0
                    state.last_slippage_ts = time.time()
            except Exception:
                pass
            label = "IOC" if use_cap else "MARKET"
            log_event(state, f"{label} SELL (DRY_RUN) {filled_qty:.6f} {symbol.split('/')[0]} @ {fill_price:.8f}")
            return filled_qty, fill_price

    base_asset = symbol.split("/")[0]
    try:
        balance = exchange.fetch_balance()
        free_base = float(balance.get("free", {}).get(base_asset) or 0.0)
    except Exception as e:
        log_event(state, f"[ERROR] Market SELL: balance check failed: {e}")
        return 0.0, price_estimate

    if free_base <= 0:
        log_event(state, f"[WARN] Market SELL skipped: no free {base_asset} on exchange.")
        return 0.0, price_estimate

    qty_to_sell = min(qty_exit, free_base) * 0.999  # safety margin
    if qty_to_sell * price_estimate < min_notional and qty_exit < free_base:
        qty_to_sell = min(free_base, max(qty_to_sell, min_qty))
    if qty_to_sell <= 0:
        log_event(state, "[WARN] Market SELL skipped: quantity <= 0 after safety margin.")
        return 0.0, price_estimate

    best_pre = (book or {}).get("best_bid") or fetch_best_book_price(exchange, symbol, "sell", price_estimate)
    limit_price = (best_pre or price_estimate) * (1 - max_slip_pct / 100.0) if use_cap else price_estimate
    attempt = 0
    filled_qty = 0.0
    fill_price = price_estimate

    while attempt < (2 if use_cap else 1) and filled_qty <= 0:
        attempt += 1
        try:
            params = {"timeInForce": "IOC"} if use_cap else {}
            order = exchange.create_order(symbol, "limit" if use_cap else "market", "sell", qty_to_sell, limit_price, params)
            filled_qty = float(order.get("filled") or order.get("amount") or 0.0)
            fill_price = float(order.get("average") or order.get("price") or limit_price)
            slip_bps = 0.0
            if best_pre and best_pre > 0 and filled_qty > 0:
                slip_bps = (1.0 - fill_price / best_pre) * 10000.0
            try:
                with state.lock:
                    state.last_slippage_bps = float(slip_bps)
                    state.last_slippage_side = "SELL"
                    state.last_slippage_best = float(best_pre or 0.0)
                    state.last_slippage_fill = float(fill_price)
                    state.last_slippage_levels = 0
                    state.last_slippage_ts = time.time()
            except Exception:
                pass
            log_event(
                state,
                f"{'IOC' if use_cap else 'MARKET'} SELL filled {filled_qty:.6f} {symbol.split('/')[0]} @ {fill_price:.8f} "
                f"(best={best_pre:.8f}, cap={limit_price:.8f}, slip={slip_bps:.1f} bps, attempt={attempt})"
            )
        except Exception as e:
            log_event(state, f"[ERROR] SELL attempt {attempt} error: {e}")
            filled_qty = 0.0

        if filled_qty <= 0 and use_cap and attempt == 1:
            limit_price = (best_pre or price_estimate) * (1 - (max_slip_pct * 1.5) / 100.0)
            continue

    if filled_qty <= 0 and use_cap:
        try:
            order = exchange.create_order(symbol, "market", "sell", qty_to_sell)
            filled_qty = float(order.get("filled") or order.get("amount") or qty_to_sell)
            fill_price = float(order.get("average") or order.get("price") or price_estimate)
            slip_bps = 0.0
            if best_pre and best_pre > 0:
                slip_bps = (1.0 - fill_price / best_pre) * 10000.0
            try:
                with state.lock:
                    state.last_slippage_bps = float(slip_bps)
                    state.last_slippage_side = "SELL"
                    state.last_slippage_best = float(best_pre or 0.0)
                    state.last_slippage_fill = float(fill_price)
                    state.last_slippage_levels = 0
                    state.last_slippage_ts = time.time()
            except Exception:
                pass
            log_event(state, f"MARKET SELL fallback filled {filled_qty:.6f} {symbol.split('/')[0]} @ {fill_price:.8f} (best={best_pre:.8f}, slip={slip_bps:.1f} bps)")
        except Exception as e:
            log_event(state, f"[ERROR] Market SELL fallback error: {e}")
            return 0.0, price_estimate
    return filled_qty, fill_price


def place_limit_buy(exchange: ccxt.Exchange, state: BotState, price: float, usdt_to_spend: float,
                    now: dt.datetime, entry_index: int) -> Optional[OpenOrder]:
    if usdt_to_spend <= 0 or price <= 0:
        return None
    qty = usdt_to_spend / price
    qty, min_qty, min_notional = normalize_qty_for_market(exchange, price, qty)
    if qty <= 0 or qty < min_qty or qty * price < min_notional:
        log_event(state, f"LIMIT BUY skipped: qty {qty:.8f} below min (min_qty≈{min_qty:.8f}, min_notional={min_notional:.2f}).")
        return None
    if qty <= 0:
        return None
    symbol = SYMBOL

    if DRY_RUN:
        log_event(state, f"LIMIT BUY (DRY_RUN) {qty:.6f} {symbol.split('/')[0]} @ {price:.8f} - waiting to fill")
        return OpenOrder(id=None, side="buy", type="limit", price=price, qty=qty, status="open",
                         created_time=now, simulated=True)

    try:
        order = exchange.create_order(symbol, "limit", "buy", qty, price)
        oid = order.get("id")
        actual_price = float(order.get("price") or price)
        log_event(state, f"LIMIT BUY sent {qty:.6f} {symbol.split('/')[0]} @ {actual_price:.8f} (id={oid})")
        return OpenOrder(id=oid, side="buy", type="limit", price=actual_price, qty=qty, status="open",
                         created_time=now, simulated=False)
    except Exception as e:
        log_event(state, f"[ERROR] Limit BUY error: {e}")
        return None


def place_limit_sell(exchange: ccxt.Exchange, state: BotState, position: Position, price: float,
                     now: dt.datetime) -> Optional[OpenOrder]:
    qty_exit = position.qty
    if qty_exit <= 0 or price <= 0:
        return None
    symbol = position.symbol
    qty_exit, min_qty, min_notional = normalize_qty_for_market(exchange, price, qty_exit)
    if qty_exit <= 0 or qty_exit < min_qty or qty_exit * price < min_notional:
        log_event(state, f"LIMIT SELL skipped: qty {qty_exit:.8f} below min (min_qty≈{min_qty:.8f}, min_notional={min_notional:.2f}).")
        return None

    if DRY_RUN:
        log_event(state, f"LIMIT SELL (DRY_RUN) {qty_exit:.6f} {symbol.split('/')[0]} @ {price:.8f} - waiting to fill")
        return OpenOrder(id=None, side="sell", type="limit", price=price, qty=qty_exit, status="open",
                         created_time=now, simulated=True)

    try:
        order = exchange.create_order(symbol, "limit", "sell", qty_exit, price)
        oid = order.get("id")
        actual_price = float(order.get("price") or price)
        log_event(state, f"LIMIT SELL sent {qty_exit:.6f} {symbol.split('/')[0]} @ {actual_price:.8f} (id={oid})")
        return OpenOrder(id=oid, side="sell", type="limit", price=actual_price, qty=qty_exit, status="open",
                         created_time=now, simulated=False)
    except Exception as e:
        log_event(state, f"[ERROR] Limit SELL error: {e}")
        return None


def realize_pnl_for_exit(state: BotState, position: Position, qty_exit: float, exit_price: float) -> float:
    if qty_exit <= 0 or position.qty <= 0:
        return 0.0

    initial_qty = getattr(position, "initial_qty", position.qty)
    denom_qty = initial_qty if initial_qty > 0 else position.qty
    if denom_qty <= 0:
        denom_qty = qty_exit
    qty_ratio = min(1.0, max(0.0, qty_exit / denom_qty))
    allocated_entry_fee = position.entry_fee * qty_ratio

    exit_value = qty_exit * exit_price
    exit_fee = exit_value * TAKER_FEE
    pnl_gross = qty_exit * (exit_price - position.entry_price)
    pnl_net = pnl_gross - allocated_entry_fee - exit_fee

    if DRY_RUN:
        with state.lock:
            state.sim_balance_usdt += exit_value - exit_fee
            state.sim_base_qty -= qty_exit
            if state.sim_base_qty < 0:
                state.sim_base_qty = 0.0

    position.qty -= qty_exit
    if position.qty < 0:
        position.qty = 0.0

    position.realized_pnl += pnl_net

    # Update lifetime PnL (persisted)
    try:
        with state.lock:
            state.total_realized_pnl = float(getattr(state, 'total_realized_pnl', 0.0)) + float(pnl_net)
    except Exception:
        pass
    return pnl_net


# ============================================================
# ==================== LIMIT ORDER CHECKER ===================
# ============================================================

def check_open_order(exchange: ccxt.Exchange, state: BotState, df: pd.DataFrame):
    oo = state.open_order
    if not oo or oo.status != "open":
        return

    last_row = df.iloc[-1]

    if oo.side == "sell" and getattr(state, "use_tp1_trailing", True):
        try:
            if not oo.simulated and oo.id:
                exchange.cancel_order(oo.id, SYMBOL)
        except Exception as e:
            log_event(state, f"[WARN] Canceling stale SELL limit order failed: {e}")
        oo.status = "canceled"
        state.open_order = None
        state.pending_sell_price = None
        return

    low = float(last_row["low"])
    high = float(last_row["high"])
    last_price = float(last_row["close"])
    now_ts = time.time()

    if oo.simulated:
        age = now_ts - oo.created_time.timestamp()
        if oo.side == "buy":
            if low <= oo.price <= high or (age > 1.0 and last_price <= oo.price * 1.002):
                fill_qty = oo.qty
                fill_price = oo.price
                cost_notional = fill_qty * fill_price
                fee = cost_notional * TAKER_FEE

                with state.lock:
                    state.sim_balance_usdt -= (cost_notional + fee)
                    state.sim_base_qty += fill_qty

                pos = Position(
                    symbol=SYMBOL,
                    entry_price=fill_price,
                    qty=fill_qty,
                    initial_qty=fill_qty,
                    entry_time=dt.datetime.now(dt.timezone.utc),
                    entry_index=len(df) - 1,
                    entry_fee=fee,
                    peak_price=fill_price,
                    trail_armed=False,
                    trail_pct_used=float(getattr(state, "effective_trail_pct", TRAIL_STOP_PCT)),
                    stop_pct_used=float(getattr(state, "stop_loss_effective_pct", STOP_LOSS_PCT)) if (getattr(state, "stop_loss_effective_pct", STOP_LOSS_PCT) is not None) else None,
                    hard_stop_mult_used=float(getattr(state, "effective_hard_stop_mult", HARD_STOP_MULT_DEFAULT)),
                    soft_confirms_used=int(getattr(state, "soft_stop_confirms", SOFT_STOP_CONFIRMS_DEFAULT)),
                )
                state.position = pos
                state.last_trade_price = fill_price
                state.last_trade_side = "BUY"
                state.anchor_timestamp = time.time()
                state.status = "IN_POSITION"

                oo.status = "filled"
                state.open_order = None
                log_event(state, f"LIMIT BUY filled (DRY_RUN) {fill_qty:.6f} {SYMBOL.split('/')[0]} @ {fill_price:.8f}")
        elif oo.side == "sell":
            if low <= oo.price <= high or (age > 1.0 and last_price >= oo.price * 0.998):
                position = state.position
                if position:
                    fill_qty = min(oo.qty, position.qty)
                    fill_price = oo.price
                    pnl = realize_pnl_for_exit(state, position, fill_qty, fill_price)
                    state.daily_realized_pnl += pnl
                    position.last_exit_reason = "limit_take_profit"

                    state.position = None
                    state.status = "WAIT_DIP"
                    state.last_trade_price = fill_price
                    state.last_trade_side = "SELL"
                    state.anchor_timestamp = time.time()

                    log_event(state, f"LIMIT SELL filled (DRY_RUN) {fill_qty:.6f} {SYMBOL.split('/')[0]} @ {fill_price:.8f}, PnL: {pnl:.2f} USDT")
                oo.status = "filled"
                state.open_order = None
        return

    try:
        order = exchange.fetch_order(oo.id, SYMBOL)
    except Exception as e:
        log_event(state, f"[ERROR] fetch_order failed: {e}")
        return

    status = (order.get("status") or "").lower()
    filled_qty = float(order.get("filled") or order.get("amount") or 0.0)
    avg_price = float(order.get("average") or order.get("price") or oo.price)

    if status == "closed" or filled_qty >= oo.qty * 0.999:
        filled_qty = min(filled_qty, oo.qty)
        if oo.side == "buy":
            pos = Position(
                symbol=SYMBOL,
                entry_price=avg_price,
                qty=filled_qty,
                initial_qty=filled_qty,
                entry_time=dt.datetime.now(dt.timezone.utc),
                entry_index=len(df) - 1,
                entry_fee=filled_qty * avg_price * TAKER_FEE,
                peak_price=avg_price,
                trail_armed=False,
                trail_pct_used=float(getattr(state, "effective_trail_pct", TRAIL_STOP_PCT)),
                stop_pct_used=float(getattr(state, "stop_loss_effective_pct", STOP_LOSS_PCT)) if (getattr(state, "stop_loss_effective_pct", STOP_LOSS_PCT) is not None) else None,
                hard_stop_mult_used=float(getattr(state, "effective_hard_stop_mult", HARD_STOP_MULT_DEFAULT)),
                soft_confirms_used=int(getattr(state, "soft_stop_confirms", SOFT_STOP_CONFIRMS_DEFAULT)),
            )
            state.position = pos
            state.last_trade_price = avg_price
            state.last_trade_side = "BUY"
            state.anchor_timestamp = time.time()
            state.status = "IN_POSITION"
            log_event(state, f"LIMIT BUY filled {filled_qty:.6f} {SYMBOL.split('/')[0]} @ {avg_price:.8f}")
        elif oo.side == "sell":
            position = state.position
            if position:
                fill_qty = min(filled_qty, position.qty)
                pnl = realize_pnl_for_exit(state, position, fill_qty, avg_price)
                state.daily_realized_pnl += pnl
                position.last_exit_reason = "limit_take_profit"

                state.position = None
                state.status = "WAIT_DIP"
                state.last_trade_price = avg_price
                state.last_trade_side = "SELL"
                state.anchor_timestamp = time.time()

                log_event(state, f"LIMIT SELL filled {fill_qty:.6f} {SYMBOL.split('/')[0]} @ {avg_price:.8f}, PnL: {pnl:.2f} USDT")
        oo.status = "filled"
        state.open_order = None
    elif status in ("canceled", "rejected", "expired"):
        oo.status = status
        state.open_order = None
        log_event(state, f"Limit order {oo.side} {status} on exchange.")




# ============================================================
# ============== ADAPTIVE THRESHOLDS (DATA) ==================
# ============================================================

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _tf_to_minutes(tf: str) -> int:
    tf = (tf or "").strip().lower()
    try:
        if tf.endswith("m"):
            return max(1, int(float(tf[:-1])))
        if tf.endswith("h"):
            return max(1, int(float(tf[:-1])) * 60)
        if tf.endswith("d"):
            return max(1, int(float(tf[:-1])) * 1440)
    except Exception:
        pass
    return 1

def _derive_weights(timeframes: List[str], override: Optional[List[float]] = None) -> List[float]:
    """Weights for blending multiple horizons.

    - If override is provided and valid, it is used.
    - Otherwise, custom weights favor faster timeframes automatically (1/sqrt(minutes)).
    """
    if override and len(override) == len(timeframes) and all(isinstance(x, (int, float)) for x in override):
        w = [float(x) for x in override]
    else:
        w = []
        for tf in timeframes:
            m = float(_tf_to_minutes(tf))
            w.append(1.0 / (m ** 0.5))
    s = sum(w) or 1.0
    return [x / s for x in w]

def _atr_per_min_pct(exchange: ccxt.Exchange, symbol: str, timeframe: str, lookback: int) -> Tuple[float, pd.DataFrame]:
    """Compute a robust ATR-like volatility (True Range %) and normalize it per-minute.

    Returns:
      atr_per_min_pct, df
    """
    lb = max(30, int(lookback))
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=max(lb + 5, 80))
    if not ohlcv:
        raise RuntimeError(f"No OHLCV returned for {timeframe}")

    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "vol"])
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)

    prev_close = df["close"].shift(1)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    tr_pct = (tr / df["close"]) * 100.0
    # Robustify: cap extreme candles (wicks/spikes) to reduce "one wick ruins everything".
    try:
        cap = float(tr_pct.quantile(0.90))
        tr_pct = tr_pct.clip(upper=cap)
    except Exception:
        pass

    tr_pct = tr_pct.tail(lb)
    atr_pct = float(tr_pct.mean()) if len(tr_pct) else 0.0

    minutes = float(_tf_to_minutes(timeframe))
    atr_per_min = atr_pct / (minutes ** 0.5)
    return atr_per_min, df

def compute_adaptive_thresholds(exchange: ccxt.Exchange, state: BotState) -> Dict[str, Any]:
    """Compute effective BUY/SELL thresholds using multi-horizon volatility + trend.

    - For each selected timeframe, we compute a robust TR% (ATR-like) and normalize it to a per-minute scale.
    - We blend horizons with preset/custom weights.
    - Convert blended volatility into BUY/SELL thresholds via k_buy/k_sell, then apply trend multipliers.
    """
    with state.lock:
        profile = str(getattr(state, "adaptive_profile", ADAPTIVE_PROFILE_DEFAULT) or ADAPTIVE_PROFILE_DEFAULT)
        custom_tfs = list(getattr(state, "adaptive_timeframes", []) or [])
        custom_w = list(getattr(state, "adaptive_weights", []) or [])
        lookback = int(getattr(state, "data_lookback", DATA_LOOKBACK_CANDLES_DEFAULT) or DATA_LOOKBACK_CANDLES_DEFAULT)
        k_buy = float(getattr(state, "adaptive_k_buy", ADAPTIVE_PRESETS[ADAPTIVE_PROFILE_DEFAULT]["k_buy"]))
        k_sell = float(getattr(state, "adaptive_k_sell", ADAPTIVE_PRESETS[ADAPTIVE_PROFILE_DEFAULT]["k_sell"]))
        use_edge = bool(getattr(state, "use_edge_aware_thresholds", USE_EDGE_AWARE_THRESHOLDS_DEFAULT))

    if profile in ADAPTIVE_PRESETS and profile != "Custom":
        preset = ADAPTIVE_PRESETS[profile]
        timeframes = list(preset["timeframes"])
        weights = list(preset["weights"])
        lookback = int(preset["lookback"])
        k_buy = float(preset["k_buy"])
        k_sell = float(preset["k_sell"])
    else:
        # Custom: user-selected timeframes (at least one), weights derived automatically unless provided.
        timeframes = [tf for tf in custom_tfs if tf] or [DATA_TIMEFRAME_DEFAULT]
        weights = _derive_weights(timeframes, custom_w)

    # Normalize weights (safety)
    if len(weights) != len(timeframes) or not weights:
        weights = _derive_weights(timeframes, None)

    # Compute per-horizon volatility
    vols: List[float] = []
    dfs: Dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        v, df_tf = _atr_per_min_pct(exchange, SYMBOL, tf, lookback)
        vols.append(max(0.0, float(v)))
        dfs[tf] = df_tf

    blended_v = 0.0
    wsum = 0.0
    for w, v in zip(weights, vols):
        blended_v += float(w) * float(v)
        wsum += float(w)
    blended_v = blended_v / (wsum or 1.0)

    # Trend on the *slowest* selected timeframe (stable regime signal)
    trend_tf = sorted(timeframes, key=lambda x: _tf_to_minutes(x))[-1]
    df_tr = dfs.get(trend_tf)
    trend = "SIDE"
    if df_tr is not None and not df_tr.empty:
        closes = df_tr["close"].astype(float).tolist()
        s_ma = sum(closes[-10:]) / max(1, min(10, len(closes)))
        l_ma = sum(closes[-30:]) / max(1, min(30, len(closes)))
        if s_ma > l_ma * 1.001:
            trend = "UP"
        elif s_ma < l_ma * 0.999:
            trend = "DOWN"
        else:
            trend = "SIDE"

    # For compatibility with previous UI fields, keep avg_high/avg_low from the trend timeframe.
    avg_high = float(df_tr["high"].astype(float).tail(lookback).mean()) if df_tr is not None and not df_tr.empty else 0.0
    avg_low = float(df_tr["low"].astype(float).tail(lookback).mean()) if df_tr is not None and not df_tr.empty else 0.0

    # Convert volatility (per-minute %) into effective thresholds
    raw_buy = max(0.0, blended_v) * k_buy
    raw_sell = max(0.0, blended_v) * k_sell

    # Edge-aware clamp to cover costs (fees + spread + slippage + buffer)
    fee_bps = max(0.0, TAKER_FEE * 10000.0)
    spread_pct = EDGE_SPREAD_FALLBACK_PCT
    try:
        ticker = exchange.fetch_ticker(SYMBOL) or {}
        bid = float(ticker.get("bid") or 0.0)
        ask = float(ticker.get("ask") or 0.0)
        if bid > 0 and ask > 0 and ask > bid:
            spread_pct = max(spread_pct, (ask - bid) / bid * 100.0)
    except Exception:
        pass

    slip_pct = EDGE_SLIPPAGE_FALLBACK_PCT
    try:
        slip_bps = float(getattr(state, "last_slippage_bps", 0.0) or 0.0)
        slip_pct = max(slip_pct, slip_bps / 100.0)
    except Exception:
        pass

    if use_edge:
        required_edge_bps = (2 * fee_bps) + (spread_pct * 100.0) + (2 * slip_pct * 100.0) + (EDGE_BUFFER_PCT_DEFAULT * 100.0)
        min_required_move_pct = required_edge_bps / 100.0
        raw_sell = max(raw_sell, min_required_move_pct)
        raw_buy = max(raw_buy, min_required_move_pct * 0.5)

    # Apply trend bias (reuse existing multipliers)
    if trend == "UP":
        buy_mult, sell_mult = DATA_BUY_MULT_UP, DATA_SELL_MULT_UP
    elif trend == "DOWN":
        buy_mult, sell_mult = DATA_BUY_MULT_DOWN, DATA_SELL_MULT_DOWN
    else:
        buy_mult, sell_mult = DATA_BUY_MULT_SIDE, DATA_SELL_MULT_SIDE

    eff_buy = _clamp(raw_buy * buy_mult, DATA_BUY_MIN_PCT, DATA_BUY_MAX_PCT)
    eff_sell = _clamp(raw_sell * sell_mult, DATA_SELL_MIN_PCT, DATA_SELL_MAX_PCT)

    return {
        "profile": profile,
        "timeframes": timeframes,
        "weights": weights,
        "trend_tf": trend_tf,
        "trend": trend,
        "blended_atr_per_min_pct": blended_v,
        "avg_high": avg_high,
        "avg_low": avg_low,
        "eff_buy": eff_buy,
        "eff_sell": eff_sell,
        "k_buy": k_buy,
        "k_sell": k_sell,
    }


def _rate_limit(current: float, target: float, step: float) -> float:
    if step <= 0:
        return target
    if target > current:
        return min(target, current + step)
    if target < current:
        return max(target, current - step)
    return current


def maybe_update_autopilot(exchange: ccxt.Exchange, state: BotState, last_price: float, new_candle: bool) -> None:
    """Update autopilot auto/effective values on new candles; respects modes and locks."""
    if not new_candle or last_price <= 0:
        return

    now_ts = time.time()
    vol = float(getattr(state, "blended_atr_per_min_pct", 0.0) or 0.0)
    trend = str(getattr(state, "data_trend", "SIDE") or "SIDE")

    # Spread estimate
    spread_pct = EDGE_SPREAD_FALLBACK_PCT
    try:
        ticker = exchange.fetch_ticker(SYMBOL) or {}
        bid = float(ticker.get("bid") or 0.0)
        ask = float(ticker.get("ask") or 0.0)
        if bid > 0 and ask > 0 and ask > bid:
            spread_pct = max(spread_pct, (ask - bid) / bid * 100.0)
    except Exception:
        pass

    slip_pct = EDGE_SLIPPAGE_FALLBACK_PCT
    try:
        slip_bps = float(getattr(state, "last_slippage_bps", 0.0) or 0.0)
        slip_pct = max(slip_pct, slip_bps / 100.0)
    except Exception:
        pass

    fee_pct = max(0.0, TAKER_FEE * 100.0)
    edge_buffer = EDGE_BUFFER_PCT_DEFAULT

    def _update_param(key: str, auto_val: float, min_val: float, max_val: float, step: float) -> float:
        ap = _get_ap(state, key, auto_val)
        ap.auto = max(min_val, min(auto_val, max_val))
        ap.auto = round(ap.auto, 6)
        if (now_ts - ap.last_manual_ts) < 300:  # 5 min lock after manual edit
            ap.effective = ap.manual
        else:
            target = ap.auto
            if ap.mode == "manual":
                target = ap.manual
            elif ap.mode == "hybrid":
                # Clamp auto within +/-20% of manual as guardrails
                lo = ap.manual * 0.8
                hi = ap.manual * 1.2
                target = min(max(ap.auto, lo), hi)
            ap.effective = _rate_limit(ap.effective, target, step)
        ap.effective = max(min_val, min(ap.effective, max_val))
        ap.last_update_ts = now_ts
        state.autopilot[key] = ap
        return ap.effective

    # Trail %
    trail_target = max(0.15, min(1.5, 0.20 + vol * 0.6))
    state.effective_trail_pct = _update_param("trail_pct", trail_target, 0.1, 3.0, 0.1)

    # TP1 fraction
    if trend == "UP":
        tp1_target = 0.40
    elif trend == "DOWN":
        tp1_target = 0.60
    else:
        tp1_target = 0.50
    state.effective_tp1_frac = _update_param("tp1_frac", tp1_target, 0.2, 0.8, 0.05)

    # Soft stop confirms (int but store float, apply int later)
    soft_target = 1.0 if vol < 0.6 else 2.0
    eff_soft = _update_param("soft_stop_confirms", soft_target, 1.0, 3.0, 1.0)
    state.soft_stop_confirms = int(round(eff_soft))

    # Hard stop multiplier
    hard_target = max(1.2, min(2.0, 1.4 + vol * 0.1))
    state.effective_hard_stop_mult = _update_param("hard_stop_mult", hard_target, 1.2, 2.5, 0.1)

    # Max slippage %
    slip_target = max(0.15, min(1.2, slip_pct * 1.3))
    eff_slip = _update_param("max_slip_pct", slip_target, 0.05, 2.0, 0.1)
    state.max_slip_pct = eff_slip

    # Stop loss pct (hybrid): auto based on ATR, clamp against user input if provided
    auto_stop_pct = max(0.30, min(1.20, 1.2 * vol))
    manual_stop = STOP_LOSS_PCT
    stop_min = float(manual_stop) if manual_stop is not None else 0.0
    stop_max = 1.5 * stop_min if stop_min > 0 else 2.0
    stop_min = stop_min if stop_min > 0 else 0.0
    eff_stop = _update_param("stop_pct", auto_stop_pct, max(0.3, stop_min), max(2.0, stop_max), 0.05)
    state.stop_loss_effective_pct = eff_stop if eff_stop > 0 else manual_stop

    # Auto buy pct (%)
    free_usdt = state.sim_balance_usdt if DRY_RUN else 0.0
    if not DRY_RUN:
        try:
            bal = exchange.fetch_balance()
            free_usdt = float(bal.get("free", {}).get("USDT") or 0.0)
        except Exception:
            free_usdt = state.sim_balance_usdt
    equity = max(0.0, float(state.wallet_equity or 0.0))
    risk_pct = 0.8  # percent of equity
    risk_usdt = equity * (risk_pct / 100.0)
    stop_assumed = state.stop_loss_effective_pct if state.stop_loss_effective_pct and state.stop_loss_effective_pct > 0 else max(0.35, 0.9 * vol)
    stop_assumed = max(0.35, stop_assumed)
    size_usdt = risk_usdt / (max(stop_assumed, 0.01) / 100.0)
    auto_buy_pct = 100.0
    if free_usdt > 0:
        auto_buy_pct = max(20.0, min(100.0, (size_usdt / free_usdt) * 100.0))
    # small-cap: if funds barely above min, go full size
    _, _, min_notional = normalize_qty_for_market(exchange, last_price, 1.0)
    if free_usdt < 2 * min_notional:
        auto_buy_pct = 100.0
    eff_auto_buy = _update_param("auto_buy_pct", auto_buy_pct, 20.0, 100.0, 5.0)
    state.auto_buy_pct = eff_auto_buy / 100.0
    state.auto_buy_pct_effective = state.auto_buy_pct

    # Edge clamp helper: store min required move if you want to display
    min_required = (2 * fee_pct) + spread_pct + slip_pct + edge_buffer
    state.effective_edge_floor = min_required  # type: ignore


def _force_flat_for_switch(exchange: ccxt.Exchange, state: BotState, last_price: float) -> bool:
    if state.open_order and state.open_order.status == "open":
        try:
            if not state.open_order.simulated and state.open_order.id:
                exchange.cancel_order(state.open_order.id, SYMBOL)
        except Exception as e:
            log_event(state, f"[WARN] Auto-coin cancel failed: {e}")
        state.open_order.status = "canceled"
        state.open_order = None

    position = state.position
    if not position:
        return True

    price = last_price
    if price <= 0:
        try:
            ticker = exchange.fetch_ticker(SYMBOL) or {}
            price = float(ticker.get("last") or 0.0)
        except Exception:
            price = 0.0
    if price <= 0:
        log_event(state, "[WARN] Auto-coin switch blocked: cannot fetch price to close position.")
        return False

    filled_qty, exit_price = place_market_sell(exchange, state, position, position.qty, price)
    if filled_qty <= 0:
        log_event(state, "[WARN] Auto-coin switch blocked: failed to close position.")
        return False

    pnl = realize_pnl_for_exit(state, position, filled_qty, exit_price)
    state.daily_realized_pnl += pnl
    position.last_exit_reason = "auto_coin_switch"
    log_event(state, f"Auto-coin closed {filled_qty:.6f} @ {exit_price:.8f}, PnL: {pnl:.2f} USDT")

    state.position = None
    state.status = "WAIT_DIP"
    state.last_trade_price = exit_price
    state.last_trade_side = "SELL"
    state.anchor_timestamp = time.time()
    state.pending_sell_price = None
    state.next_trade_time = time.time() + POLL_INTERVAL_SECONDS
    return True


def maybe_update_auto_coin(exchange: ccxt.Exchange, state: BotState, last_price: float,
                           force_switch: bool = False, ignore_interval: bool = False) -> bool:
    """Scan CoinEx trending candidates and switch symbols when conditions are met."""
    if EXCHANGE_NAME.lower() != "coinex":
        return False

    with state.lock:
        enabled = bool(getattr(state, "auto_coin_enabled", AUTO_COIN_ENABLED_DEFAULT))
        scan_interval = float(getattr(state, "auto_coin_scan_interval_sec", AUTO_COIN_SCAN_INTERVAL_SEC_DEFAULT))
        last_scan = float(getattr(state, "auto_coin_last_scan_ts", 0.0))
        policy = str(getattr(state, "auto_coin_policy", AUTO_COIN_POLICY_DEFAULT) or AUTO_COIN_POLICY_DEFAULT)
        dwell_min = float(getattr(state, "auto_coin_dwell_min", AUTO_COIN_DWELL_MIN_DEFAULT))
        hysteresis = float(getattr(state, "auto_coin_hysteresis_pct", AUTO_COIN_HYSTERESIS_PCT_DEFAULT))
        last_switch = float(getattr(state, "auto_coin_last_switch_ts", 0.0))

    if not enabled and not force_switch and not ignore_interval:
        return False

    now_ts = time.time()
    if not ignore_interval and (now_ts - last_scan) < max(10.0, scan_interval):
        return False

    with state.lock:
        state.auto_coin_last_scan_ts = now_ts

    result = _scan_coinex_trending(state, exchange)
    scored = result.get("scored") or []
    ui_candidates = result.get("ui_candidates") or []
    best = result.get("best")
    ticker_map = result.get("ticker_map") or {}
    relax_next = float(result.get("relax_next", 1.0))
    tickers_count = int(result.get("tickers_count", 0) or 0)
    items_count = int(result.get("items_count", 0) or 0)
    candidates_count = int(result.get("candidates_count", 0) or 0)
    survivors_count = int(result.get("survivors_count", 0) or 0)
    scored_count = int(result.get("scored_count", 0) or 0)
    ticker_source = str(result.get("ticker_source", "coinex") or "coinex")
    reject_min_value = int(result.get("reject_min_value", 0) or 0)
    reject_depth_fail = int(result.get("reject_depth_fail", 0) or 0)
    reject_spread = int(result.get("reject_spread", 0) or 0)
    reject_slippage = int(result.get("reject_slippage", 0) or 0)
    reject_min_value_candidates = int(result.get("reject_min_value_candidates", 0) or 0)
    kline_fail = int(result.get("kline_fail", 0) or 0)
    depth_samples = result.get("depth_samples") or []
    fallback_used = bool(result.get("fallback_used", False))

    with state.lock:
        state.auto_coin_relax_factor = relax_next
        state.auto_coin_top_candidates = ui_candidates[:5]
        state.auto_coin_last_scan_info = (
            f"tickers={tickers_count} items={items_count} candidates={candidates_count} "
            f"survivors={survivors_count} scored={scored_count} "
            f"rej[value={reject_min_value}/cand_value={reject_min_value_candidates}/depth={reject_depth_fail}/"
            f"spread={reject_spread}/slip={reject_slippage}/kline={kline_fail}]"
        )
        if fallback_used:
            state.auto_coin_last_scan_info += " fallback=1"

    if tickers_count == 0:
        log_event(state, f"Auto-coin scan: no tickers from {ticker_source} API.")
    else:
        log_event(
            state,
            f"Auto-coin scan: tickers={tickers_count} items={items_count} candidates={candidates_count} "
            f"survivors={survivors_count} scored={scored_count} "
            f"rej[value={reject_min_value}/cand_value={reject_min_value_candidates}/depth={reject_depth_fail}/"
            f"spread={reject_spread}/slip={reject_slippage}/kline={kline_fail}]"
            f"{' fallback=1' if fallback_used else ''}."
        )
        if depth_samples:
            sample_str = ", ".join(
                f"{_market_to_symbol(str(s.get('market')))} spread={float(s.get('spread_bps', 0.0)):.1f} bps slip={float(s.get('slip_bps_buy_100', 0.0)):.1f} bps"
                for s in depth_samples[:3]
            )
            log_event(state, f"Auto-coin depth samples: {sample_str}")

    if not best and force_switch and ui_candidates:
        best = ui_candidates[0]

    best_market = str(best.get("market") or "") if best else ""
    best_score = float(best.get("score", 0.0)) if best else 0.0

    current_market = _symbol_to_market(SYMBOL)
    current_score = 0.0
    for c in scored:
        if str(c.get("market") or "") == current_market:
            current_score = float(c.get("score", 0.0))
            break
    if current_score <= 0 and current_market:
        current_info = _score_single_market(current_market, ticker_map.get(current_market))
        if current_info:
            current_score = float(current_info.get("score", 0.0))

    winner_streak = 0
    with state.lock:
        state.auto_coin_current_score = current_score
        if best_market:
            if best_market == state.auto_coin_last_winner:
                state.auto_coin_winner_streak += 1
            else:
                state.auto_coin_last_winner = best_market
                state.auto_coin_winner_streak = 1
            winner_streak = state.auto_coin_winner_streak

    block_reasons: List[str] = []
    if not best_market:
        block_reasons.append("no_best")
    elif best_market == current_market:
        block_reasons.append("same_symbol")
    if not force_switch:
        if winner_streak < 2:
            block_reasons.append("not_consecutive")
        if dwell_min > 0 and (now_ts - last_switch) < (dwell_min * 60.0):
            block_reasons.append("dwell")
        if current_score > 0 and best_score < (current_score * (1.0 + hysteresis / 100.0)):
            block_reasons.append("hysteresis")

    if force_switch:
        policy = "force"
    if policy not in ("flat", "force"):
        policy = "flat"

    if policy == "flat":
        if state.position is not None or (state.open_order and state.open_order.status == "open"):
            block_reasons.append("not_flat")

    delta_str = "n/a"
    if current_score > 0:
        delta_pct = (best_score - current_score) / current_score * 100.0
        delta_str = f"{delta_pct:+.1f}%"
    reason_str = "switch" if not block_reasons else ",".join(block_reasons)
    log_event(
        state,
        f"Auto-coin decision: best={_market_to_symbol(best_market) if best_market else '-'}({best_score:.2f}) "
        f"current={SYMBOL}({current_score:.2f}) delta={delta_str} reason={reason_str}"
    )

    if block_reasons:
        return False

    if policy == "force":
        if not _force_flat_for_switch(exchange, state, last_price):
            log_event(state, "Auto-coin decision: force_close_failed")
            return False

    new_symbol = _market_to_symbol(best_market)
    log_event(
        state,
        f"Auto-coin switch: {SYMBOL} -> {new_symbol} (ATR%={best.get('atr_pct', 0.0):.2f}, "
        f"spread={best.get('spread_bps', 0.0):.1f} bps, slip={best.get('slip_bps_buy_100', 0.0):.1f} bps, "
        f"value={best.get('value', 0.0):.0f}, score={best_score:.2f})"
    )
    prev_symbol = SYMBOL
    change_symbol(exchange, state, new_symbol)
    if SYMBOL == new_symbol and prev_symbol != new_symbol:
        with state.lock:
            state.auto_coin_last_switch_ts = now_ts
        return True
    return False

def maybe_refresh_adaptive(exchange: ccxt.Exchange, state: BotState) -> None:
    """Called from trading loop. Keeps effective thresholds updated."""
    with state.lock:
        if not state.data_driven:
            state.effective_buy_threshold_pct = state.buy_threshold_pct
            state.effective_sell_threshold_pct = state.sell_threshold_pct
            return

        now = time.time()

        # refresh cadence depends on profile (or custom refresh)
        profile = str(getattr(state, "adaptive_profile", ADAPTIVE_PROFILE_DEFAULT) or ADAPTIVE_PROFILE_DEFAULT)
        refresh_sec = int(getattr(state, "data_refresh_sec", DATA_REFRESH_SEC_DEFAULT) or DATA_REFRESH_SEC_DEFAULT)
        alpha = float(getattr(state, "adaptive_alpha", ADAPTIVE_PRESETS[ADAPTIVE_PROFILE_DEFAULT]["alpha"]))
        if profile in ADAPTIVE_PRESETS and profile != "Custom":
            refresh_sec = int(ADAPTIVE_PRESETS[profile]["refresh"])
            alpha = float(ADAPTIVE_PRESETS[profile]["alpha"])

        if (now - state._last_data_refresh_ts) < max(5, refresh_sec):
            return

        # force update timestamp now to avoid stampede on failures
        state._last_data_refresh_ts = now

    try:
        stats = compute_adaptive_thresholds(exchange, state)
        with state.lock:
            state.data_trend = stats["trend"]
            state.avg_high = stats["avg_high"]
            state.avg_low = stats["avg_low"]

            # This field used to be avg candle range %. Now it's blended ATR% per minute (still in % units).
            state.avg_range_pct = float(stats["blended_atr_per_min_pct"])
            state.blended_atr_per_min_pct = float(stats["blended_atr_per_min_pct"])

            # smooth thresholds (EMA)
            old_b = float(state.effective_buy_threshold_pct or 0.0)
            old_s = float(state.effective_sell_threshold_pct or 0.0)
            new_b = float(stats["eff_buy"])
            new_s = float(stats["eff_sell"])

            if old_b <= 0:
                state.effective_buy_threshold_pct = new_b
            else:
                state.effective_buy_threshold_pct = (alpha * new_b) + ((1.0 - alpha) * old_b)

            if old_s <= 0:
                state.effective_sell_threshold_pct = new_s
            else:
                state.effective_sell_threshold_pct = (alpha * new_s) + ((1.0 - alpha) * old_s)

            # Floors/edge clamps
            buy_floor = max(0.08, 0.35 * state.blended_atr_per_min_pct)
            state.effective_buy_threshold_pct = max(state.effective_buy_threshold_pct, buy_floor)

            # Keep state fields aligned (for persistence/UI)
            state.adaptive_profile = str(stats.get("profile") or state.adaptive_profile)
            state.adaptive_timeframes = list(stats.get("timeframes") or state.adaptive_timeframes)
            state.adaptive_weights = list(stats.get("weights") or state.adaptive_weights)
            state.adaptive_k_buy = float(stats.get("k_buy") or state.adaptive_k_buy)
            state.adaptive_k_sell = float(stats.get("k_sell") or state.adaptive_k_sell)
            state.data_timeframe = str(stats.get("trend_tf") or state.data_timeframe)

        save_settings(state)
    except Exception as e:
        log_event(state, f"[WARN] Adaptive thresholds failed: {e}")


# ============================================================
# ============== BUY-LOW / SELL-HIGH LOGIC ===================
# ============================================================

def maybe_open_position(exchange: ccxt.Exchange, state: BotState, df: pd.DataFrame):
    if state.position is not None:
        return

    now_ts = time.time()
    if now_ts < state.next_trade_time:
        return

    if state.trading_paused_until and now_ts < state.trading_paused_until:
        return

    if state.daily_realized_pnl <= DAILY_MAX_LOSS_USDT or state.daily_realized_pnl >= DAILY_MAX_PROFIT_USDT:
        state.paused_reason = "Daily PnL limit reached"
        return

    last_row = df.iloc[-1]
    last_price = float(last_row["close"])
    last_idx = len(df) - 1
    buy_thr = (state.effective_buy_threshold_pct if state.data_driven else state.buy_threshold_pct)

    if state.last_trade_price is None or state.last_trade_side is None:
        state.last_trade_price = last_price
        state.last_trade_side = "SELL"
        state.anchor_timestamp = now_ts
        state.status = "WAIT_DIP"
        state.dip_wait_start_ts = 0.0
        state.dip_low = 0.0
        state.dip_higher_close_count = 0
        buy_trigger = state.last_trade_price * (1 - buy_thr / 100.0)
        state.pending_buy_price = buy_trigger
        log_event(state, f"Initialized. Waiting for dip: BUY trigger at {buy_trigger:.8f}")
        return

    if state.last_trade_side != "SELL":
        return

    age = (now_ts - state.anchor_timestamp) if state.anchor_timestamp is not None else None
    price_above_anchor = last_price > state.last_trade_price * (1 + ANCHOR_DRIFT_REARM_PCT / 100.0)

    if (age is not None and age > ANCHOR_MAX_AGE_SEC) or price_above_anchor:
        old_anchor = state.last_trade_price
        state.last_trade_price = last_price
        state.anchor_timestamp = now_ts
        state.pending_buy_price = None
        state.dip_wait_start_ts = 0.0
        state.dip_low = 0.0
        state.dip_higher_close_count = 0
        log_event(state, f"Re-anchoring BUY reference {old_anchor:.8f} -> {last_price:.8f} (age={0 if age is None else age:.0f}s).")

    buy_trigger_price = state.last_trade_price * (1 - buy_thr / 100.0)

    if state.pending_buy_price is None or abs(buy_trigger_price - state.pending_buy_price) > 1e-12:
        state.pending_buy_price = buy_trigger_price
        # keep log quieter; comment back in if you want it noisy:
        # log_event(state, f"New BUY trigger set at {buy_trigger_price:.8f} (waiting for dip)")

    use_rebound = bool(getattr(state, "use_dip_rebound", True))
    rebound_pct = max(0.0, float(getattr(state, "rebound_confirm_pct", REBOUND_CONFIRM_PCT_DEFAULT)))
    rebound_timeout = max(5.0, float(getattr(state, "dip_rebound_timeout_sec", DIP_REBOUND_TIMEOUT_SEC_DEFAULT)))
    confirm_closes = max(1, int(getattr(state, "dip_confirm_closes", DIP_CONFIRM_CLOSES_DEFAULT)))
    no_new_low_buf = max(0.0, float(DIP_NO_NEW_LOW_BUFFER_PCT))

    if last_price > buy_trigger_price:
        state.status = "WAIT_DIP"
        state.dip_wait_start_ts = 0.0
        state.dip_low = 0.0
        state.dip_higher_close_count = 0
        return

    # Dip reached: wait for bounce confirmation if enabled
    if use_rebound:
        if state.dip_wait_start_ts <= 0:
            state.dip_wait_start_ts = now_ts
            state.dip_low = last_price
            state.dip_higher_close_count = 0
            log_event(state, f"Dip hit {last_price:.8f}. Waiting for bounce to confirm BUY.")
        else:
            state.dip_low = min(state.dip_low if state.dip_low > 0 else last_price, last_price)

        # Track consecutive higher closes
        if len(df) >= 2:
            prev_close = float(df.iloc[-2]["close"])
            if last_price > prev_close:
                state.dip_higher_close_count += 1
            elif last_price < prev_close:
                state.dip_higher_close_count = 0
                state.dip_low = min(state.dip_low, last_price)

        rebound_ok = False
        if state.dip_low > 0 and last_price >= state.dip_low * (1 + rebound_pct / 100.0):
            rebound_ok = True
        if state.dip_higher_close_count >= confirm_closes:
            rebound_ok = True

        timeout_hit = (now_ts - state.dip_wait_start_ts) >= rebound_timeout
        if timeout_hit and state.dip_low > 0 and last_price >= state.dip_low * (1 + no_new_low_buf / 100.0):
            rebound_ok = True

        if not rebound_ok and not timeout_hit:
            state.status = "WAIT_BOUNCE"
            return
        # proceed; clear counters
        state.dip_wait_start_ts = 0.0
        state.dip_low = 0.0
        state.dip_higher_close_count = 0

    # size
    if DRY_RUN:
        available_usdt = state.sim_balance_usdt
        if available_usdt <= 0:
            state.sim_balance_usdt = float(SIMULATED_BALANCE_USDT)
            available_usdt = state.sim_balance_usdt
            log_event(state, f"Paper balance reset to {available_usdt:.2f} USDT for DRY_RUN.")
    else:
        try:
            balance = exchange.fetch_balance()
            available_usdt = float(balance.get("free", {}).get("USDT") or 0.0)
        except Exception as e:
            log_event(state, f"[ERROR] Error fetching balance for buy sizing: {e}")
            return

    if available_usdt <= 0:
        log_event(state, "No sufficient USDT to buy.")
        return

    spend_usdt = available_usdt * max(0.0, min(state.auto_buy_pct, 1.0))
    if spend_usdt < MIN_NOTIONAL_USDT:
        log_event(state, f"Auto BUY skipped: spend {spend_usdt:.2f} < min_notional {MIN_NOTIONAL_USDT:.2f}.")
        return

    if state.order_mode == "market":
        now = dt.datetime.now(dt.timezone.utc)
        pos = place_market_buy(exchange, state, spend_usdt, last_price, now, last_idx)
        if not pos:
            return

        state.position = pos
        state.status = "IN_POSITION"
        state.next_trade_time = time.time() + POLL_INTERVAL_SECONDS

        state.last_trade_price = pos.entry_price
        state.last_trade_side = "BUY"
        state.anchor_timestamp = time.time()
        state.pending_buy_price = None
        state.dip_wait_start_ts = 0.0
        state.dip_low = 0.0
        state.dip_higher_close_count = 0
        log_event(state, f"Entered position (market) @ {pos.entry_price:.8f}, qty={pos.qty:.6f}")
    else:
        if state.open_order and state.open_order.status == "open" and state.open_order.side == "buy":
            return

        now = dt.datetime.now(dt.timezone.utc)
        oo = place_limit_buy(exchange, state, last_price, spend_usdt, now, last_idx)
        if oo:
            state.open_order = oo
            state.next_trade_time = time.time() + POLL_INTERVAL_SECONDS
            state.dip_wait_start_ts = 0.0
            state.dip_low = 0.0
            state.dip_higher_close_count = 0


def manage_position(exchange: ccxt.Exchange, state: BotState, df: pd.DataFrame):
    position = state.position
    if position is None:
        return

    last_row = df.iloc[-1]
    last_price = float(last_row["close"])
    sell_thr = (state.effective_sell_threshold_pct if state.data_driven else state.sell_threshold_pct)

    use_trailing = bool(getattr(state, "use_tp1_trailing", True))
    use_soft_stop = bool(getattr(state, "use_soft_stop", True))
    soft_confirms = max(1, int(getattr(position, "soft_confirms_used", getattr(state, "soft_stop_confirms", SOFT_STOP_CONFIRMS_DEFAULT))))
    hard_stop_mult = max(1.0, float(getattr(position, "hard_stop_mult_used", getattr(state, "effective_hard_stop_mult", HARD_STOP_MULT_DEFAULT))))
    stop_pct_used = position.stop_pct_used if getattr(position, "stop_pct_used", None) else (state.stop_loss_effective_pct if state.stop_loss_effective_pct else STOP_LOSS_PCT)
    anchor_price = state.last_trade_price if state.last_trade_price is not None else position.entry_price
    sell_trigger = anchor_price * (1 + sell_thr / 100.0)
    tp1_fraction = max(0.2, min(float(getattr(state, "effective_tp1_frac", TP1_SELL_FRACTION)), 0.8))

    # Legacy: single full take-profit with optional stop-loss/daily guardrails.
    if not use_trailing:
        if state.order_mode == "limit":
            if state.open_order and state.open_order.status == "open" and state.open_order.side == "sell":
                return
            if last_price >= sell_trigger:
                now = dt.datetime.now(dt.timezone.utc)
                oo = place_limit_sell(exchange, state, position, last_price, now)
                if oo:
                    state.open_order = oo
                    state.pending_sell_price = last_price
            return

        full_exit = False
        reason = None

        if last_price >= sell_trigger:
            full_exit = True
            reason = "take_profit_threshold"

        if (not full_exit) and (stop_pct_used is not None):
            stop_price = position.entry_price * (1 - stop_pct_used / 100.0)
            hard_stop_price = position.entry_price * (1 - (stop_pct_used * hard_stop_mult) / 100.0)

            if last_price > stop_price:
                state.soft_stop_counter = 0

            if last_price <= hard_stop_price:
                full_exit = True
                reason = "hard_stop"
                state.soft_stop_counter = 0
            elif use_soft_stop:
                if last_price <= stop_price:
                    state.soft_stop_counter = min(soft_confirms, state.soft_stop_counter + 1)
                    if state.soft_stop_counter >= soft_confirms:
                        full_exit = True
                        reason = "soft_stop"
                        state.soft_stop_counter = 0
                else:
                    state.soft_stop_counter = 0
            else:
                if last_price <= stop_price:
                    full_exit = True
                    reason = "stop_loss"
                    state.soft_stop_counter = 0
        else:
            state.soft_stop_counter = 0

        if (not full_exit) and (
            state.daily_realized_pnl <= DAILY_MAX_LOSS_USDT or state.daily_realized_pnl >= DAILY_MAX_PROFIT_USDT
        ):
            full_exit = True
            reason = "daily_limit"

        if full_exit and position.qty > 0:
            qty_exit = position.qty
            filled_qty, exit_price = place_market_sell(exchange, state, position, qty_exit, last_price)
            if filled_qty > 0:
                pnl = realize_pnl_for_exit(state, position, filled_qty, exit_price)
                state.daily_realized_pnl += pnl

                if reason == "stop_loss":
                    now_ts = time.time()
                    state.stop_loss_timestamps.append(now_ts)
                    while state.stop_loss_timestamps and now_ts - state.stop_loss_timestamps[0] > 3600:
                        state.stop_loss_timestamps.popleft()
                    if len(state.stop_loss_timestamps) >= MAX_STOPS_PER_HOUR:
                        state.trading_paused_until = now_ts + CIRCUIT_STOP_COOLDOWN_SEC
                        state.paused_reason = "Too many stop losses in last hour"
                        log_event(state, "Circuit breaker: too many stop losses. Pausing trading.")

                position.last_exit_reason = reason
                log_event(state, f"Exited position @ {exit_price:.8f} (reason: {reason}), PnL: {pnl:.2f} USDT")

            state.position = None
            state.status = "WAIT_DIP"
            state.next_trade_time = time.time() + POLL_INTERVAL_SECONDS
            state.last_trade_price = exit_price
            state.last_trade_side = "SELL"
            state.anchor_timestamp = time.time()
            state.pending_sell_price = None
            state.open_order = None
        return

    if getattr(position, "peak_price", 0.0) <= 0:
        position.peak_price = position.entry_price
    position.peak_price = max(position.peak_price, last_price)

    def _compute_trail_pct() -> float:
        trail_base = getattr(position, "trail_pct_used", getattr(state, "effective_trail_pct", TRAIL_STOP_PCT))
        trail = max(trail_base, sell_thr * 0.5)
        atr_pct = float(getattr(state, "blended_atr_per_min_pct", 0.0) or 0.0)
        if TRAIL_ATR_MULT and atr_pct > 0:
            trail = max(trail, atr_pct * TRAIL_ATR_MULT)
        return max(trail, 0.0)

    def _arm_trail_if_ready():
        if getattr(position, "trail_armed", False):
            return True
        fee_pct = max(0.0, TAKER_FEE * 100.0)
        slip_pct = max(EDGE_SLIPPAGE_FALLBACK_PCT, float(getattr(state, "last_slippage_bps", 0.0) or 0.0) / 100.0)
        buffer_pct = 0.05
        arm_pct = (2 * fee_pct) + slip_pct + buffer_pct
        if position.tp1_done or last_price >= position.entry_price * (1 + arm_pct / 100.0):
            position.trail_armed = True
            return True
        return False

    def _cancel_open_sell_if_any():
        oo = state.open_order
        if oo and oo.status == "open" and oo.side == "sell":
            try:
                if not oo.simulated and oo.id:
                    exchange.cancel_order(oo.id, SYMBOL)
            except Exception as e:
                log_event(state, f"[WARN] Failed to cancel open SELL limit: {e}")
            oo.status = "canceled"
            state.open_order = None
            state.pending_sell_price = None

    def _finalize_close(exit_price: float, reason: str):
        state.position = None
        state.status = "WAIT_DIP"
        cooldown = POLL_INTERVAL_SECONDS
        if reason in ("hard_stop", "stop_loss", "soft_stop"):
            cooldown = max(POLL_INTERVAL_SECONDS, POLL_INTERVAL_SECONDS * 2)
        elif reason == "trailing_stop":
            cooldown = POLL_INTERVAL_SECONDS
        elif reason in ("take_profit_threshold", "tp1_partial", "trailing_exit"):
            cooldown = 0.0
        state.next_trade_time = time.time() + cooldown
        state.last_trade_price = exit_price
        state.last_trade_side = "SELL"
        state.anchor_timestamp = time.time()
        state.pending_sell_price = None
        state.open_order = None
        state.soft_stop_counter = 0

    def _record_stop_loss_timestamp():
        now_ts = time.time()
        state.stop_loss_timestamps.append(now_ts)
        while state.stop_loss_timestamps and now_ts - state.stop_loss_timestamps[0] > 3600:
            state.stop_loss_timestamps.popleft()
        if len(state.stop_loss_timestamps) >= MAX_STOPS_PER_HOUR:
            state.trading_paused_until = now_ts + CIRCUIT_STOP_COOLDOWN_SEC
            state.paused_reason = "Too many stop losses in last hour"
            log_event(state, "Circuit breaker: too many stop losses. Pausing trading.")

    def _execute_market_exit(qty_exit: float, reason: str) -> Tuple[bool, float]:
        if qty_exit <= 0 or position.qty <= 0:
            return False, 0.0

        _cancel_open_sell_if_any()
        filled_qty, exit_price = place_market_sell(exchange, state, position, qty_exit, last_price)
        if filled_qty <= 0:
            return False, 0.0

        pnl = realize_pnl_for_exit(state, position, filled_qty, exit_price)
        state.daily_realized_pnl += pnl

        if reason in ("stop_loss", "soft_stop", "hard_stop"):
            _record_stop_loss_timestamp()

        position.last_exit_reason = reason
        remaining_qty = position.qty

        if remaining_qty > 0:
            log_event(state, f"Partial exit ({reason}) {filled_qty:.6f} @ {exit_price:.8f}, remaining={remaining_qty:.6f}, PnL: {pnl:.2f} USDT")
            position.peak_price = max(position.peak_price, exit_price, last_price)
            return False, filled_qty

        log_event(state, f"Exited position @ {exit_price:.8f} (reason: {reason}), PnL: {pnl:.2f} USDT")
        _finalize_close(exit_price, reason)
        return True, filled_qty

    # Forced exits first (risk limits)
    stop_pct_used = position.stop_pct_used if getattr(position, "stop_pct_used", None) else (state.stop_loss_effective_pct if state.stop_loss_effective_pct else STOP_LOSS_PCT)
    if stop_pct_used is not None:
        stop_price = position.entry_price * (1 - stop_pct_used / 100.0)
        hard_stop_price = position.entry_price * (1 - (stop_pct_used * hard_stop_mult) / 100.0)

        if last_price > stop_price:
            state.soft_stop_counter = 0

        if last_price <= hard_stop_price:
            state.soft_stop_counter = 0
            closed, _ = _execute_market_exit(position.qty, "hard_stop")
            if closed or state.position is None:
                return
        elif use_soft_stop:
            if last_price <= stop_price:
                state.soft_stop_counter = min(soft_confirms, state.soft_stop_counter + 1)
                if state.soft_stop_counter >= soft_confirms:
                    closed, _ = _execute_market_exit(position.qty, "soft_stop")
                    state.soft_stop_counter = 0
                    if closed or state.position is None:
                        return
            else:
                state.soft_stop_counter = 0
        else:
            if last_price <= stop_price:
                state.soft_stop_counter = 0
                closed, _ = _execute_market_exit(position.qty, "stop_loss")
                if closed or state.position is None:
                    return

    else:
        state.soft_stop_counter = 0

    if state.daily_realized_pnl <= DAILY_MAX_LOSS_USDT or state.daily_realized_pnl >= DAILY_MAX_PROFIT_USDT:
        closed, _ = _execute_market_exit(position.qty, "daily_limit")
        if closed or state.position is None:
            return

    # Take-profit 1: partial exit
    if (not position.tp1_done) and last_price >= sell_trigger:
        qty_exit = position.qty * tp1_fraction
        min_qty = MIN_NOTIONAL_USDT / max(last_price, 1e-9)
        if qty_exit * last_price < MIN_NOTIONAL_USDT or qty_exit < min_qty:
            position.tp1_done = True  # skip TP1, go to trailing/full logic
        else:
            closed, filled_qty = _execute_market_exit(qty_exit, "tp1_partial")
            if state.position is None or closed:
                return
            if filled_qty > 0:
                position.tp1_done = True
                position.peak_price = max(position.peak_price, last_price)

    # Trailing stop on the remainder
    if position.tp1_done and position.qty > 0:
        if not _arm_trail_if_ready():
            return
        position.peak_price = max(position.peak_price, last_price)
        trail_stop_price = position.peak_price * (1 - _compute_trail_pct() / 100.0)
        trail_stop_price = max(trail_stop_price, position.entry_price)
        if last_price <= trail_stop_price:
            closed, _ = _execute_market_exit(position.qty, "trailing_stop")
            if closed or state.position is None:
                return


# ============================================================
# ====================== WALLET VIEW =========================
# ============================================================

def fetch_wallet_live(exchange: ccxt.Exchange) -> Tuple[List[Dict[str, Any]], float]:
    rows: List[Dict[str, Any]] = []
    total_equity = 0.0
    try:
        balance = exchange.fetch_balance()
    except Exception as e:
        print(f"Error fetching balance: {e}", flush=True)
        return rows, total_equity

    free = balance.get("free", {}) or balance.get("total", {})
    for asset, amount in free.items():
        if not amount:
            continue
        amount = float(amount)
        if amount == 0:
            continue

        est_usdt = 0.0
        if asset == "USDT":
            est_usdt = amount
        else:
            sym = f"{asset}/USDT"
            if sym in exchange.markets:
                try:
                    ticker = exchange.fetch_ticker(sym)
                    est_usdt = amount * float(ticker["last"])
                except Exception:
                    est_usdt = 0.0

        total_equity += est_usdt
        rows.append({"asset": asset, "amount": amount, "est_usdt": est_usdt})

    rows.sort(key=lambda r: r["est_usdt"], reverse=True)
    return rows, total_equity


def build_wallet_view(exchange: ccxt.Exchange, state: BotState, last_price: float) -> Tuple[List[Dict[str, Any]], float]:
    has_keys = bool(API_KEY) and bool(SECRET_KEY)
    if SHOW_REAL_WALLET and has_keys and not DRY_RUN:
        return fetch_wallet_live(exchange)

    base_asset = SYMBOL.split("/")[0]
    usdt_row = {"asset": "USDT", "amount": state.sim_balance_usdt, "est_usdt": state.sim_balance_usdt}
    base_row = {"asset": base_asset, "amount": state.sim_base_qty, "est_usdt": state.sim_base_qty * last_price}
    rows = [usdt_row]
    if state.sim_base_qty > 0:
        rows.append(base_row)
    total_equity = usdt_row["est_usdt"] + base_row["est_usdt"]
    return rows, total_equity


def adopt_position_from_wallet(state: BotState, last_price: float):
    if DRY_RUN or last_price <= 0:
        return

    base_asset = SYMBOL.split("/")[0]
    qty_wallet = None
    for row in state.wallet_rows:
        if row["asset"] == base_asset:
            qty_wallet = float(row["amount"])
            break

    min_qty = MIN_NOTIONAL_USDT / last_price

    if qty_wallet is None or qty_wallet < min_qty:
        if state.position is not None:
            log_event(state, f"Wallet shows no usable {base_asset}; clearing internal position.")
            state.position = None
        return

    if state.position is None:
        now = dt.datetime.now(dt.timezone.utc)
        pos = Position(
            symbol=SYMBOL,
            entry_price=last_price,
            qty=qty_wallet,
            initial_qty=qty_wallet,
            entry_time=now,
            entry_index=-1,
            entry_fee=0.0,
            peak_price=last_price,
            trail_armed=False,
            trail_pct_used=float(getattr(state, "effective_trail_pct", TRAIL_STOP_PCT)),
            stop_pct_used=float(getattr(state, "stop_loss_effective_pct", STOP_LOSS_PCT)) if (getattr(state, "stop_loss_effective_pct", STOP_LOSS_PCT) is not None) else None,
            hard_stop_mult_used=float(getattr(state, "effective_hard_stop_mult", HARD_STOP_MULT_DEFAULT)),
            soft_confirms_used=int(getattr(state, "soft_stop_confirms", SOFT_STOP_CONFIRMS_DEFAULT)),
        )
        state.position = pos
        state.last_trade_price = last_price
        state.last_trade_side = "BUY"
        state.anchor_timestamp = time.time()
        state.status = "IN_POSITION"
        log_event(state, f"Adopted wallet holdings as position: {qty_wallet:.6f} {base_asset} @ ~{last_price:.8f}")
    else:
        if abs(state.position.qty - qty_wallet) > max(min_qty * 0.1, qty_wallet * 0.01):
            log_event(state, f"Adjusting position qty {state.position.qty:.6f} -> {qty_wallet:.6f} {base_asset} to match wallet.")
            state.position.qty = qty_wallet
            state.position.initial_qty = max(getattr(state.position, "initial_qty", qty_wallet), qty_wallet)


# ============================================================
# ================== SYMBOL SWITCHING ========================
# ============================================================

def change_symbol(exchange: ccxt.Exchange, state: BotState, new_symbol: str):
    global SYMBOL
    new_symbol = new_symbol.upper().strip()
    if not new_symbol:
        log_event(state, "[ERROR] Empty symbol string.")
        return

    if new_symbol == SYMBOL:
        log_event(state, f"Symbol is already {SYMBOL}.")
        return

    if new_symbol not in exchange.markets:
        log_event(state, f"[ERROR] Symbol {new_symbol} not found on exchange.")
        return

    if state.position is not None or (state.open_order and state.open_order.status == "open"):
        log_event(state, "Cannot change symbol while there is an open position or open order.")
        return

    SYMBOL = new_symbol
    config.SYMBOL = new_symbol
    state.last_trade_price = None
    state.last_trade_side = None
    state.anchor_timestamp = None
    state.pending_buy_price = None
    state.pending_sell_price = None
    state.last_candle_ts = None
    state.status = "INIT"
    state.sim_base_qty = 0.0
    state.dip_wait_start_ts = 0.0
    state.dip_low = 0.0
    state.dip_higher_close_count = 0

    with state.lock:
        state.chart_times = []
        state.chart_close = []
        state.chart_high = []
        state.chart_low = []

        # reset adaptive stats for new symbol (will refresh automatically if enabled)
        state.avg_high = 0.0
        state.avg_low = 0.0
        state.avg_range_pct = 0.0
        state.data_trend = "?"
        state._last_data_refresh_ts = 0.0
    log_event(state, f"Trading symbol changed to {SYMBOL}.")
    save_settings(state)


# ============================================================
# ================== COMMAND HANDLING ========================
# ============================================================

def handle_command(cmd: str, state: BotState, exchange: ccxt.Exchange):
    cmd = cmd.strip()
    low = cmd.lower()

    if low.startswith("dry_run:"):
        # Switch between DRY_RUN (paper trading) and live trading.
        # For safety, only allow switching while flat and no open limit order.
        payload = low.split(":", 1)[1].strip()
        want = payload in ("1", "true", "on", "yes", "y")
        if state.position is not None or state.open_order is not None:
            log_event(state, "[WARN] Cannot toggle DRY RUN while a position/order exists. Close/cancel first.")
            return
        global DRY_RUN
        DRY_RUN = want
        with state.lock:
            state.dry_run = want
            # Reset anchors/order state to avoid mixing modes
            state.last_trade_price = None
            state.last_trade_side = None
            state.anchor_timestamp = None
            state.pending_buy_price = None
            state.pending_sell_price = None
            state.last_candle_ts = None
            state.open_order = None
            state.position = None
            state.sim_base_qty = 0.0
            if want:
                # Seed sim USDT with current equity if available, else default.
                seed = state.wallet_equity if state.wallet_equity > 0 else SIMULATED_BALANCE_USDT
                state.sim_balance_usdt = float(seed)
        log_event(state, f"DRY RUN set to {want}.")
        save_settings(state)
        return

    if low == "reset_dry_run":
        if not state.dry_run:
            log_event(state, "[WARN] Dry-run reset ignored: DRY RUN is OFF.")
            return
        with state.lock:
            state.sim_balance_usdt = float(SIMULATED_BALANCE_USDT)
            state.sim_base_qty = 0.0
            state.daily_realized_pnl = 0.0
            state.total_realized_pnl = 0.0
            state.position = None
            state.open_order = None
            state.last_trade_price = None
            state.last_trade_side = None
            state.anchor_timestamp = None
            state.pending_buy_price = None
            state.pending_sell_price = None
            state.last_candle_ts = None
            state.soft_stop_counter = 0
            state.dip_wait_start_ts = 0.0
            state.dip_low = 0.0
            state.dip_higher_close_count = 0
            state.trading_paused_until = None
            state.paused_reason = None
            state.status = "INIT"
        log_event(state, f"Dry-run reset: balance={SIMULATED_BALANCE_USDT:.2f} USDT, PnL cleared.")
        save_settings(state)
        return


    if low == "market":
        state.order_mode = "market"
        log_event(state, "Order mode: MARKET.")
        return

    if low == "limit":
        state.order_mode = "limit"
        log_event(state, "Order mode: LIMIT.")
        return

    if low == "cancel":
        if state.open_order and state.open_order.status == "open":
            log_event(state, f"Open {state.open_order.side.upper()} LIMIT canceled.")
            state.open_order.status = "canceled"
            state.open_order = None
        else:
            log_event(state, "No open limit order to cancel.")
        return

    if low == "pause":
        state.trading_paused_until = time.time() + 10**9
        state.paused_reason = "Manually paused"
        log_event(state, "Trading PAUSED.")
        return

    if low == "resume":
        state.trading_paused_until = None
        state.paused_reason = None
        log_event(state, "Trading RESUMED.")
        return

    if low == "auto_coin_scan":
        maybe_update_auto_coin(exchange, state, state.last_price or 0.0, force_switch=False, ignore_interval=True)
        return

    if low == "auto_coin_force":
        maybe_update_auto_coin(exchange, state, state.last_price or 0.0, force_switch=True, ignore_interval=True)
        return

    if low == "help":
        log_event(state, "Use the fields for BUY/SELL thresholds, set symbol or pick a wallet coin, use buttons for manual orders.")
        return

    if low == "manual_buy":
        if state.position is not None:
            log_event(state, "Manual BUY ignored: already in position.")
            return
        if state.trading_paused_until and time.time() < state.trading_paused_until:
            log_event(state, "Manual BUY blocked: trading is paused.")
            return

        last_price = state.last_price
        if last_price <= 0:
            try:
                ticker = exchange.fetch_ticker(SYMBOL)
                last_price = float(ticker["last"])
                state.last_price = last_price
            except Exception as e:
                log_event(state, f"[ERROR] Manual BUY: cannot fetch price: {e}")
                return

        if DRY_RUN:
            available_usdt = state.sim_balance_usdt
            if available_usdt <= 0:
                state.sim_balance_usdt = float(SIMULATED_BALANCE_USDT)
                available_usdt = state.sim_balance_usdt
                log_event(state, f"Paper balance reset to {available_usdt:.2f} USDT for DRY_RUN.")
        else:
            try:
                balance = exchange.fetch_balance()
                available_usdt = float(balance.get("free", {}).get("USDT") or 0.0)
            except Exception as e:
                log_event(state, f"[ERROR] Manual BUY: balance error: {e}")
                return

        spend_pct = max(0.0, min(state.manual_buy_pct, 1.0))
        try:
            if _ap_mode(state, "auto_buy_pct") == "hybrid":
                spend_pct = min(spend_pct, max(0.0, min(state.auto_buy_pct, 1.0)))
        except Exception:
            pass
        spend = available_usdt * spend_pct
        if spend < MIN_NOTIONAL_USDT:
            log_event(state, f"Manual BUY skipped: not enough USDT (have {available_usdt:.2f}).")
            return

        now = dt.datetime.now(dt.timezone.utc)
        pos = place_market_buy(exchange, state, spend, last_price, now, -1)
        if not pos:
            return

        state.position = pos
        state.status = "IN_POSITION"
        state.last_trade_price = pos.entry_price
        state.last_trade_side = "BUY"
        state.anchor_timestamp = time.time()
        state.pending_buy_price = None
        log_event(state, f"Manual BUY opened @ {pos.entry_price:.8f}, qty={pos.qty:.6f}")
        return

    if low == "manual_sell":
        position = state.position
        if position is None:
            log_event(state, "Manual SELL ignored: no open position.")
            return

        last_price = state.last_price
        if last_price <= 0:
            try:
                ticker = exchange.fetch_ticker(SYMBOL)
                last_price = float(ticker["last"])
                state.last_price = last_price
            except Exception as e:
                log_event(state, f"[ERROR] Manual SELL: cannot fetch price: {e}")
                return

        qty_exit = position.qty * max(0.0, min(getattr(state, 'manual_sell_pct', 1.0), 1.0))
        if qty_exit <= 0:
            log_event(state, "Manual SELL ignored: position size is zero.")
            return

        filled_qty, exit_price = place_market_sell(exchange, state, position, qty_exit, last_price)
        if filled_qty > 0:
            pnl = realize_pnl_for_exit(state, position, filled_qty, exit_price)
            state.daily_realized_pnl += pnl
            position.last_exit_reason = "manual_market_sell"
            log_event(state, f"Manual SELL closed {filled_qty:.6f} @ {exit_price:.8f}, PnL: {pnl:.2f} USDT")

        state.position = None
        state.status = "WAIT_DIP"
        state.last_trade_price = exit_price
        state.last_trade_side = "SELL"
        state.anchor_timestamp = time.time()
        state.pending_sell_price = None
        state.next_trade_time = time.time() + POLL_INTERVAL_SECONDS
        return

    if low.startswith("symbol:"):
        new_symbol = cmd.split(":", 1)[1]
        change_symbol(exchange, state, new_symbol)
        return

    if low.startswith("wallet_coin:"):
        asset = cmd.split(":", 1)[1].strip().upper()
        if not asset:
            log_event(state, "[ERROR] Empty wallet coin.")
        else:
            change_symbol(exchange, state, f"{asset}/USDT")
        return

    if low.startswith("set_api:"):
        payload = cmd[len("set_api:"):]
        parts = payload.split("|", 1)
        if len(parts) != 2:
            log_event(state, "[ERROR] Invalid API details format.")
            return
        new_key, new_secret = parts[0].strip(), parts[1].strip()
        if not new_key or not new_secret:
            log_event(state, "[ERROR] API key/secret cannot be empty.")
            return

        global API_KEY, SECRET_KEY
        API_KEY = new_key
        SECRET_KEY = new_secret
        exchange.apiKey = new_key
        exchange.secret = new_secret
        try:
            exchange.load_markets()
            log_event(state, "API credentials updated and markets reloaded.")
        except Exception as e:
            log_event(state, f"[ERROR] Failed to reload markets after API update: {e}")
        return

    if low == "restart":
        log_event(state, "Restart command received (handled by GUI).")
        return

    log_event(state, f"Unknown command: {cmd}")


# ============================================================
# =================== TRADING LOOP (THREAD) ==================
# ============================================================

def trading_loop(exchange: ccxt.Exchange, state: BotState, cmd_queue: "queue.Queue[str]",
                 stop_event: threading.Event):
    log_event(state, f"Trading loop started on {EXCHANGE_NAME} ({TIMEFRAME}).")

    while not stop_event.is_set():
        loop_start = time.time()
        try:
            df = fetch_ohlcv_df(exchange)
            if df is None or df.empty:
                raise RuntimeError("No OHLCV data returned")

            last_row = df.iloc[-1]
            last_ts = int(last_row["timestamp"])
            last_price = float(last_row["close"])

            with state.lock:
                state.last_price = last_price
                state.chart_times = df["datetime"].tolist()
                state.chart_close = df["close"].astype(float).tolist()
                state.chart_high = df["high"].astype(float).tolist()
                state.chart_low = df["low"].astype(float).tolist()

            wallet_rows, wallet_equity = build_wallet_view(exchange, state, last_price)
            with state.lock:
                state.wallet_rows = wallet_rows
                state.wallet_equity = wallet_equity

            adopt_position_from_wallet(state, last_price)

            # Update adaptive thresholds (if enabled)
            maybe_refresh_adaptive(exchange, state)

            new_candle = state.last_candle_ts is None or last_ts > state.last_candle_ts
            maybe_update_autopilot(exchange, state, last_price, new_candle)

            now_dt = dt.datetime.now(dt.timezone.utc)
            if now_dt.date() != state.day_start_date:
                state.day_start_date = now_dt.date()
                state.daily_realized_pnl = 0.0
                state.stop_loss_timestamps.clear()
                state.paused_reason = None
                state.trading_paused_until = None
                log_event(state, "New day detected. Daily PnL & counters reset.")

            check_open_order(exchange, state, df)

            if state.position:
                manage_position(exchange, state, df)

            switched_symbol = maybe_update_auto_coin(exchange, state, last_price)

            if new_candle and not switched_symbol:
                state.last_candle_ts = last_ts
                if not state.position:
                    state.status = "WAIT_DIP"
                    maybe_open_position(exchange, state, df)

            now_ts = time.time()
            if state.trading_paused_until and now_ts < state.trading_paused_until:
                state.status = "PAUSED"
            elif state.daily_realized_pnl <= DAILY_MAX_LOSS_USDT or state.daily_realized_pnl >= DAILY_MAX_PROFIT_USDT:
                state.status = "PAUSED (daily)"

            while not cmd_queue.empty():
                handle_command(cmd_queue.get_nowait(), state, exchange)

            elapsed = time.time() - loop_start
            time.sleep(max(POLL_INTERVAL_SECONDS - elapsed, 0.5))

        except Exception as e:
            state.last_error = repr(e)
            log_event(state, f"[ERROR] Main loop exception: {e}")
            traceback.print_exc()
            time.sleep(5)


# ============================================================
# ============================ GUI ===========================
# ============================================================

class BotGUI:
    def __init__(self, root: tk.Tk, state: BotState, cmd_queue: "queue.Queue[str]", stop_event: threading.Event):
        self.root = root
        self.state = state
        self.cmd_queue = cmd_queue
        self.stop_event = stop_event

        self.root.title(f"{SYMBOL} · Auto Scalper")
        self.root.geometry("1400x820")

        # --- Layout: left sidebar (scrollable) + right main (chart/info) + bottom events ---
        self.SIDEBAR_WIDTH = 520
        self.root.columnconfigure(0, weight=0, minsize=self.SIDEBAR_WIDTH)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        # Sidebar
        self.sidebar = self._make_scrolled_frame(self.root, width=self.SIDEBAR_WIDTH, height=780)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(10, 8), pady=10)

        # Main area (chart + info + events stacked vertically)
        self.main = ttk.Frame(self.root)
        self.main.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        self.main.columnconfigure(0, weight=1)
        # Chart should dominate vertical space; info + events share the rest
        self.main.rowconfigure(0, weight=3)   # chart
        self.main.rowconfigure(1, weight=1)   # position + wallet panels
        self.main.rowconfigure(2, weight=1)   # events log

        # Events under the chart on the right side only
        self.events_frame = ttk.LabelFrame(self.main, text="Events")
        self.events_frame.grid(row=2, column=0, sticky="nsew", padx=0, pady=(8, 0))
        self.events_frame.columnconfigure(0, weight=1)
        self.events_frame.rowconfigure(0, weight=1)

        # Shorter log height so sidebar content has more space
        self.events_text = ScrolledText(self.events_frame, height=6, state="disabled", wrap="word")
        self.events_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        # --- Sidebar content: single scrollable control panel box ---
        sidebar_parent = self.sidebar.container if hasattr(self.sidebar, "container") else self.sidebar
        self.control_panel = ttk.LabelFrame(sidebar_parent, text="Control Panel")
        self.control_panel.pack(fill="both", expand=True)

        # Tabs: keep the existing controls, and add a new tab for adaptive/data-driven logic.
        self.tabs = ttk.Notebook(self.control_panel)
        self.tabs.pack(fill="both", expand=True, padx=2, pady=2)

        self.tab_trading = ttk.Frame(self.tabs)
        self.tab_adaptive = ttk.Frame(self.tabs)
        self.tab_trending = ttk.Frame(self.tabs)
        self.tab_toggles = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_trading, text="Trading")
        self.tabs.add(self.tab_adaptive, text="Adaptive")
        self.tabs.add(self.tab_trending, text="Trending")
        self.tabs.add(self.tab_toggles, text="Toggles")

        self._build_status_card(self.tab_trading)
        self._build_time_card(self.tab_trading)
        self._build_api_card(self.tab_trading)
        self._build_params_card(self.tab_trading)
        self._build_buttons_card(self.tab_trading)

        self._build_adaptive_tab(self.tab_adaptive)
        self._build_trending_tab(self.tab_trending)
        self._build_toggles_tab(self.tab_toggles)


        # --- Chart + info on the right ---
        self._build_chart(self.main)
        self._build_info_panels(self.main)

        self.root.after(250, self.update_ui)
        self.root.protocol("WM_DELETE_WINDOW", self.quit_bot)

    # ---------- widgets ----------
    def _make_scrolled_frame(self, parent, width=460, height=780):
        # ttkbootstrap has a nicer scrolled frame; fallback to Canvas+Frame
        if tb is not None and TBScrolledFrame is not None:
            # Newer ttkbootstrap supports width/height; older may not.
            try:
                sf = TBScrolledFrame(parent, autohide=True, width=width, height=height)  # type: ignore
            except TypeError:
                sf = TBScrolledFrame(parent, autohide=True, width=width)  # type: ignore

            # Different versions expose the inner container differently
            if not hasattr(sf, "container"):
                sf.container = sf  # type: ignore

            # Keep a stable width for the sidebar
            try:
                sf.configure(width=width)
            except Exception:
                pass
            return sf

        # fallback: manual scrolled frame
        outer = ttk.Frame(parent, width=width, height=height)
        outer.grid_propagate(False)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)

        # Make the inner frame always match the canvas width (so content isn't cramped)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            try:
                canvas.itemconfigure(win_id, width=event.width)
            except Exception:
                pass

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        outer.container = inner
        return outer

    def _card(self, parent, title: str):
        """Create a section inside the single left control panel box.
        This returns an inner frame; the outer scrollable panel is one big LabelFrame.
        """
        section = ttk.Frame(parent)
        section.pack(fill="x", padx=6, pady=(6, 8))
        header = ttk.Label(section, text=title)
        header.pack(anchor="w", padx=4, pady=(0, 2))
        inner = ttk.Frame(section)
        inner.pack(fill="x", padx=4, pady=(0, 0))
        return inner


    def _build_status_card(self, parent):
        card = self._card(parent, "Status")

        self.status_var = tk.StringVar(value="-")
        self.order_mode_var = tk.StringVar(value="-")
        self.price_var = tk.StringVar(value="-")
        self.equity_var = tk.StringVar(value="-")
        self.pnl_var = tk.StringVar(value="-")
        self.pnl_total_var = tk.StringVar(value="-")
        self.uptime_var = tk.StringVar(value="-")
        self.slippage_var = tk.StringVar(value="-")
        self.buy_dist_var = tk.StringVar(value="-")
        self.sell_dist_var = tk.StringVar(value="-")

        rows = [
            ("State", self.status_var),
            ("Order mode", self.order_mode_var),
            ("Last price", self.price_var),
            ("Equity (USDT)", self.equity_var),
            ("Daily PnL (USDT)", self.pnl_var),
            ("All-time PnL (USDT)", self.pnl_total_var),
            ("Uptime (total)", self.uptime_var),
            ("Last slippage", self.slippage_var),
            ("Dist to BUY", self.buy_dist_var),
            ("Dist to SELL", self.sell_dist_var),
        ]
        for r, (lbl, var) in enumerate(rows):
            ttk.Label(card, text=lbl + ":").grid(row=r, column=0, sticky="w", padx=8, pady=2)
            ttk.Label(card, textvariable=var).grid(row=r, column=1, sticky="w", padx=8, pady=2)

        # DRY RUN toggle (paper trading)
        self.dry_run_var = tk.BooleanVar(value=DRY_RUN)
        ttk.Label(card, text="Dry Run:").grid(row=len(rows), column=0, sticky="w", padx=8, pady=(6, 2))
        ttk.Checkbutton(card, variable=self.dry_run_var, command=self._toggle_dry_run).grid(
            row=len(rows), column=1, sticky="w", padx=8, pady=(6, 2)
        )
        self.reset_dry_run_btn = ttk.Button(card, text="Reset Dry-Run PnL + Balance", command=self._on_reset_dry_run)
        self.reset_dry_run_btn.grid(row=len(rows) + 1, column=0, columnspan=2, sticky="ew", padx=8, pady=(4, 2))

        card.columnconfigure(1, weight=1)

    def _build_time_card(self, parent):
        card = self._card(parent, "Time / Timers")

        self.time_now_var = tk.StringVar(value="-")
        self.time_candle_var = tk.StringVar(value="-")
        self.time_anchor_var = tk.StringVar(value="-")
        self.time_dip_var = tk.StringVar(value="-")
        self.time_cooldown_var = tk.StringVar(value="-")
        self.time_auto_scan_var = tk.StringVar(value="-")
        self.time_auto_dwell_var = tk.StringVar(value="-")
        self.time_adaptive_var = tk.StringVar(value="-")
        self.time_day_reset_var = tk.StringVar(value="-")

        rows = [
            ("Now (local)", self.time_now_var),
            ("Candle age", self.time_candle_var),
            ("Anchor age", self.time_anchor_var),
            ("Dip wait", self.time_dip_var),
            ("Cooldown", self.time_cooldown_var),
            ("Auto scan", self.time_auto_scan_var),
            ("Auto dwell", self.time_auto_dwell_var),
            ("Adaptive refresh", self.time_adaptive_var),
            ("Day reset (UTC)", self.time_day_reset_var),
        ]
        for r, (lbl, var) in enumerate(rows):
            ttk.Label(card, text=lbl + ":").grid(row=r, column=0, sticky="w", padx=8, pady=2)
            ttk.Label(card, textvariable=var).grid(row=r, column=1, sticky="w", padx=8, pady=2)

        card.columnconfigure(1, weight=1)

    def _build_api_card(self, parent):
        card = self._card(parent, "API / Connection")

        self.api_key_var = tk.StringVar(value=API_KEY)
        self.api_secret_var = tk.StringVar(value=SECRET_KEY)

        ttk.Label(card, text="API Key").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(card, textvariable=self.api_key_var, show="*", width=34).grid(row=0, column=1, sticky="ew", padx=8, pady=4)

        ttk.Label(card, text="API Secret").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(card, textvariable=self.api_secret_var, show="*", width=34).grid(row=1, column=1, sticky="ew", padx=8, pady=4)

        btn = ttk.Button(card, text="Apply API", command=self.on_apply_api)
        btn.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=(2, 8))
        card.columnconfigure(1, weight=1)


    
    def _build_params_card(self, parent):
        card = self._card(parent, "Live Parameters")

        # --- Core thresholds ---
        self.buy_threshold_var = tk.StringVar(value=f"{self.state.buy_threshold_pct:.3f}")
        self.sell_threshold_var = tk.StringVar(value=f"{self.state.sell_threshold_pct:.3f}")

        ttk.Label(card, text="Buy threshold (% dip)").grid(row=0, column=0, sticky="w", padx=8, pady=(6, 2))
        e_buy = ttk.Entry(card, textvariable=self.buy_threshold_var, width=10)
        e_buy.grid(row=0, column=1, sticky="e", padx=8, pady=(6, 2))
        e_buy.bind("<Return>", self.on_buy_threshold_entry)
        e_buy.bind("<FocusOut>", self.on_buy_threshold_entry)
        self.buy_threshold_entry = e_buy

        ttk.Label(card, text="Sell threshold (% pump)").grid(row=1, column=0, sticky="w", padx=8, pady=(4, 2))
        e_sell = ttk.Entry(card, textvariable=self.sell_threshold_var, width=10)
        e_sell.grid(row=1, column=1, sticky="e", padx=8, pady=(4, 2))
        e_sell.bind("<Return>", self.on_sell_threshold_entry)
        e_sell.bind("<FocusOut>", self.on_sell_threshold_entry)
        self.sell_threshold_entry = e_sell

        # --- Position sizing (percent of USDT) ---
        self.auto_buy_pct_var = tk.StringVar(value=f"{self.state.auto_buy_pct * 100.0:.1f}")
        self.manual_buy_pct_var = tk.StringVar(value=f"{self.state.manual_buy_pct * 100.0:.1f}")

        ttk.Label(card, text="Auto BUY size (% of USDT)").grid(row=2, column=0, sticky="w", padx=8, pady=(10, 2))
        e_auto = ttk.Entry(card, textvariable=self.auto_buy_pct_var, width=10)
        e_auto.grid(row=2, column=1, sticky="e", padx=8, pady=(10, 2))
        e_auto.bind("<Return>", self.on_auto_buy_entry)
        e_auto.bind("<FocusOut>", self.on_auto_buy_entry)
        self.auto_buy_entry = e_auto

        self.auto_buy_display_var = tk.StringVar(value="")
        ttk.Label(card, textvariable=self.auto_buy_display_var).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 8)
        )

        ttk.Label(card, text="Manual BUY size (% of USDT)").grid(row=4, column=0, sticky="w", padx=8, pady=(6, 2))
        e_manual = ttk.Entry(card, textvariable=self.manual_buy_pct_var, width=10)
        e_manual.grid(row=4, column=1, sticky="e", padx=8, pady=(6, 2))
        e_manual.bind("<Return>", self.on_manual_buy_entry)
        e_manual.bind("<FocusOut>", self.on_manual_buy_entry)
        self.manual_buy_entry = e_manual

        self.manual_buy_display_var = tk.StringVar(value="")
        ttk.Label(card, textvariable=self.manual_buy_display_var).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 8)
        )

        # Manual SELL sizing (percent of position)
        self.manual_sell_pct_var = tk.StringVar(value=f"{getattr(self.state, 'manual_sell_pct', 1.0) * 100.0:.1f}")
        ttk.Label(card, text="Manual SELL size (% of position)").grid(row=6, column=0, sticky="w", padx=8, pady=(6, 2))
        e_msell = ttk.Entry(card, textvariable=self.manual_sell_pct_var, width=10)
        e_msell.grid(row=6, column=1, sticky="e", padx=8, pady=(6, 2))
        e_msell.bind("<Return>", self.on_manual_sell_entry)
        e_msell.bind("<FocusOut>", self.on_manual_sell_entry)
        self.manual_sell_entry = e_msell


        # --- Advanced trading parameters (limits, SL, timing, manual sell) ---
        ttk.Separator(card).grid(row=7, column=0, columnspan=2, sticky="ew", padx=8, pady=(10, 4))

        self.stop_loss_var = tk.StringVar(value="" if STOP_LOSS_PCT is None else f"{STOP_LOSS_PCT}")
        self.daily_loss_var = tk.StringVar(value=f"{DAILY_MAX_LOSS_USDT}")
        self.daily_profit_var = tk.StringVar(value=f"{DAILY_MAX_PROFIT_USDT}")
        self.anchor_age_var = tk.StringVar(value=f"{ANCHOR_MAX_AGE_SEC}")
        self.anchor_drift_var = tk.StringVar(value=f"{ANCHOR_DRIFT_REARM_PCT}")
        self.poll_interval_var = tk.StringVar(value=f"{POLL_INTERVAL_SECONDS}")
        self.min_notional_var = tk.StringVar(value=f"{MIN_NOTIONAL_USDT}")
        self.max_stops_var = tk.StringVar(value=f"{MAX_STOPS_PER_HOUR}")
        self.circuit_cooldown_var = tk.StringVar(value=f"{CIRCUIT_STOP_COOLDOWN_SEC}")

        adv_fields = [
            ("Stop loss % (blank=off)", self.stop_loss_var),
            ("Daily max loss (USDT)", self.daily_loss_var),
            ("Daily max profit (USDT)", self.daily_profit_var),
            ("Anchor max age (sec)", self.anchor_age_var),
            ("Anchor drift re-arm %", self.anchor_drift_var),
            ("Poll interval (sec)", self.poll_interval_var),
            ("Min notional (USDT)", self.min_notional_var),
            ("Max stops per hour", self.max_stops_var),
            ("Circuit cooldown (sec)", self.circuit_cooldown_var),
        ]
        base_row = 8
        for idx, (lbl, var) in enumerate(adv_fields):
            r = base_row + idx
            ttk.Label(card, text=lbl).grid(row=r, column=0, sticky="w", padx=8, pady=3)
            ent = ttk.Entry(card, textvariable=var, width=12)
            ent.grid(row=r, column=1, sticky="e", padx=8, pady=3)
            if lbl.startswith("Stop loss"):
                self.stop_loss_entry = ent

        ttk.Button(card, text="Apply Trading Config", command=self.on_apply_config).grid(
            row=base_row + len(adv_fields), column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 8)
        )

        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)

    def _build_advanced_card(self, parent):
        card = self._card(parent, "Advanced Config")

        self.stop_loss_var = tk.StringVar(value="" if STOP_LOSS_PCT is None else f"{STOP_LOSS_PCT}")
        self.daily_loss_var = tk.StringVar(value=f"{DAILY_MAX_LOSS_USDT}")
        self.daily_profit_var = tk.StringVar(value=f"{DAILY_MAX_PROFIT_USDT}")
        self.anchor_age_var = tk.StringVar(value=f"{ANCHOR_MAX_AGE_SEC}")
        self.anchor_drift_var = tk.StringVar(value=f"{ANCHOR_DRIFT_REARM_PCT}")
        self.poll_interval_var = tk.StringVar(value=f"{POLL_INTERVAL_SECONDS}")
        self.min_notional_var = tk.StringVar(value=f"{MIN_NOTIONAL_USDT}")

        fields = [
            ("Stop loss % (blank=off)", self.stop_loss_var),
            ("Daily max loss (USDT)", self.daily_loss_var),
            ("Daily max profit (USDT)", self.daily_profit_var),
            ("Anchor max age (sec)", self.anchor_age_var),
            ("Anchor drift re-arm %", self.anchor_drift_var),
            ("Poll interval (sec)", self.poll_interval_var),
            ("Min notional (USDT)", self.min_notional_var),
        ]

        for r, (lbl, var) in enumerate(fields):
            ttk.Label(card, text=lbl).grid(row=r, column=0, sticky="w", padx=8, pady=3)
            ttk.Entry(card, textvariable=var, width=12).grid(row=r, column=1, sticky="e", padx=8, pady=3)

        ttk.Button(card, text="Apply Advanced Config", command=self.on_apply_config).grid(row=len(fields), column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 8))
        card.columnconfigure(0, weight=1)



    def _build_adaptive_tab(self, parent):
        """Adaptive/data-driven thresholds tab (multi-horizon volatility profiles).

        You can:
        - Toggle Manual vs Data-driven thresholds
        - Pick a profile: Scalping / Short / Medium / Long / Custom
        - In Custom: choose which timeframes to blend (scroll + click)
        - Set lookback + refresh cadence
        - (Custom) tweak k_buy / k_sell and smoothing alpha
        """
        card = self._card(parent, "Adaptive thresholds")

        # Manual vs Data-driven toggle
        self.threshold_mode_var = tk.StringVar(value="data" if self.state.data_driven else "manual")
        rb_frame = ttk.Frame(card)
        rb_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(6, 2))
        ttk.Radiobutton(
            rb_frame, text="Manual % thresholds", value="manual",
            variable=self.threshold_mode_var, command=self.on_threshold_mode_change
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            rb_frame, text="Data-driven (volatility profiles)", value="data",
            variable=self.threshold_mode_var, command=self.on_threshold_mode_change
        ).pack(side="left")

        # Live readout of what's being used
        self.adaptive_info_var = tk.StringVar(value="-")
        ttk.Label(card, textvariable=self.adaptive_info_var).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 10)
        )

        ttk.Separator(card).grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=(2, 8))

        # Profile selector
        ttk.Label(card, text="Profile").grid(row=3, column=0, sticky="w", padx=8, pady=(2, 2))
        self.profile_var = tk.StringVar(value=str(getattr(self.state, "adaptive_profile", ADAPTIVE_PROFILE_DEFAULT)))
        self.profile_combo = ttk.Combobox(card, values=ADAPTIVE_PROFILES, state="readonly", textvariable=self.profile_var)
        self.profile_combo.grid(row=3, column=1, sticky="e", padx=8, pady=(2, 2))
        self.profile_combo.bind("<<ComboboxSelected>>", self.on_profile_change)

        # Timeframe selector (scroll + click)
        ttk.Label(card, text="Timeframes to blend (scroll + click)").grid(
            row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 2)
        )

        tf_wrap = ttk.Frame(card)
        tf_wrap.grid(row=5, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))
        tf_wrap.columnconfigure(0, weight=1)

        self.tf_list = tk.Listbox(tf_wrap, height=7, exportselection=False, selectmode="multiple")
        tf_sb = ttk.Scrollbar(tf_wrap, orient="vertical", command=self.tf_list.yview)
        self.tf_list.configure(yscrollcommand=tf_sb.set)

        self.tf_list.grid(row=0, column=0, sticky="ew")
        tf_sb.grid(row=0, column=1, sticky="ns")

        for tf in DATA_TIMEFRAMES:
            self.tf_list.insert("end", tf)

        # Click-to-toggle selection (no Ctrl needed)
        self.tf_list.bind("<Button-1>", self.on_tf_click)

        # Lookback + refresh
        self.data_lookback_var = tk.StringVar(value=str(getattr(self.state, "data_lookback", DATA_LOOKBACK_CANDLES_DEFAULT)))
        self.data_refresh_var = tk.StringVar(value=str(getattr(self.state, "data_refresh_sec", DATA_REFRESH_SEC_DEFAULT)))

        ttk.Label(card, text="Lookback candles").grid(row=6, column=0, sticky="w", padx=8, pady=(2, 2))
        self.lb_entry = ttk.Entry(card, textvariable=self.data_lookback_var, width=10)
        self.lb_entry.grid(row=6, column=1, sticky="e", padx=8, pady=(2, 2))
        self.lb_entry.bind("<Return>", self.on_data_params_change)
        self.lb_entry.bind("<FocusOut>", self.on_data_params_change)

        ttk.Label(card, text="Refresh seconds").grid(row=7, column=0, sticky="w", padx=8, pady=(2, 2))
        self.rf_entry = ttk.Entry(card, textvariable=self.data_refresh_var, width=10)
        self.rf_entry.grid(row=7, column=1, sticky="e", padx=8, pady=(2, 2))
        self.rf_entry.bind("<Return>", self.on_data_params_change)
        self.rf_entry.bind("<FocusOut>", self.on_data_params_change)

        # Custom coefficients
        ttk.Separator(card).grid(row=8, column=0, columnspan=2, sticky="ew", padx=8, pady=(10, 6))
        ttk.Label(card, text="Custom coefficients (Custom profile)").grid(
            row=9, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 4)
        )

        self.k_buy_var = tk.StringVar(value=f"{float(getattr(self.state, 'adaptive_k_buy', ADAPTIVE_PRESETS[ADAPTIVE_PROFILE_DEFAULT]['k_buy'])):.3f}")
        self.k_sell_var = tk.StringVar(value=f"{float(getattr(self.state, 'adaptive_k_sell', ADAPTIVE_PRESETS[ADAPTIVE_PROFILE_DEFAULT]['k_sell'])):.3f}")
        self.alpha_var = tk.StringVar(value=f"{float(getattr(self.state, 'adaptive_alpha', ADAPTIVE_PRESETS[ADAPTIVE_PROFILE_DEFAULT]['alpha'])):.3f}")

        ttk.Label(card, text="k_buy").grid(row=10, column=0, sticky="w", padx=8, pady=2)
        self.k_buy_entry = ttk.Entry(card, textvariable=self.k_buy_var, width=10)
        self.k_buy_entry.grid(row=10, column=1, sticky="e", padx=8, pady=2)
        self.k_buy_entry.bind("<Return>", self.on_custom_coeff_change)
        self.k_buy_entry.bind("<FocusOut>", self.on_custom_coeff_change)

        ttk.Label(card, text="k_sell").grid(row=11, column=0, sticky="w", padx=8, pady=2)
        self.k_sell_entry = ttk.Entry(card, textvariable=self.k_sell_var, width=10)
        self.k_sell_entry.grid(row=11, column=1, sticky="e", padx=8, pady=2)
        self.k_sell_entry.bind("<Return>", self.on_custom_coeff_change)
        self.k_sell_entry.bind("<FocusOut>", self.on_custom_coeff_change)

        ttk.Label(card, text="Smoothing alpha (0..1)").grid(row=12, column=0, sticky="w", padx=8, pady=2)
        self.alpha_entry = ttk.Entry(card, textvariable=self.alpha_var, width=10)
        self.alpha_entry.grid(row=12, column=1, sticky="e", padx=8, pady=2)
        self.alpha_entry.bind("<Return>", self.on_custom_coeff_change)
        self.alpha_entry.bind("<FocusOut>", self.on_custom_coeff_change)

        ttk.Button(card, text="Refresh adaptive stats now", command=self.force_adaptive_refresh).grid(
            row=13, column=0, columnspan=2, sticky="ew", padx=8, pady=(10, 8)
        )

        card.columnconfigure(1, weight=1)

        # Sync selection/UI states from current state
        self._sync_profile_to_ui()

        # If data-driven is enabled from settings, reflect that in Trading tab inputs.
        if self.state.data_driven:
            try:
                self.buy_threshold_entry.configure(state="disabled")
                self.sell_threshold_entry.configure(state="disabled")
            except Exception:
                pass

    def _build_toggles_tab(self, parent):
        card = self._card(parent, "Feature Toggles")

        self.tp1_toggle_var = tk.BooleanVar(value=bool(getattr(self.state, "use_tp1_trailing", True)))
        ttk.Label(card, text="TP1 partial + trailing exit").grid(row=0, column=0, sticky="w", padx=8, pady=(4, 2))
        ttk.Checkbutton(card, variable=self.tp1_toggle_var, command=self._on_toggle_tp1_trailing).grid(
            row=0, column=1, sticky="w", padx=8, pady=(4, 2)
        )
        ttk.Label(
            card,
            text="Disable to revert to single full take-profit exit (still honors SL/daily guards).",
            wraplength=self.SIDEBAR_WIDTH - 60,
            justify="left",
            foreground="#666",
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))

        self.soft_stop_toggle_var = tk.BooleanVar(value=bool(getattr(self.state, "use_soft_stop", True)))
        ttk.Label(card, text="Soft stop (confirm) + Hard stop").grid(row=2, column=0, sticky="w", padx=8, pady=(8, 2))
        ttk.Checkbutton(card, variable=self.soft_stop_toggle_var, command=self._on_toggle_soft_stop).grid(
            row=2, column=1, sticky="w", padx=8, pady=(8, 2)
        )
        ttk.Label(
            card,
            text="Soft stop waits for multiple closes under stop to avoid wick-outs; hard stop hits deeper drop immediately.",
            wraplength=self.SIDEBAR_WIDTH - 60,
            justify="left",
            foreground="#666",
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 10))

        self.dip_rebound_toggle_var = tk.BooleanVar(value=bool(getattr(self.state, "use_dip_rebound", True)))
        ttk.Label(card, text="Dip ?+' bounce confirmation").grid(row=4, column=0, sticky="w", padx=8, pady=(4, 2))
        ttk.Checkbutton(card, variable=self.dip_rebound_toggle_var, command=self._on_toggle_dip_rebound).grid(
            row=4, column=1, sticky="w", padx=8, pady=(4, 2)
        )
        ttk.Label(card, text="Rebound confirm %").grid(row=5, column=0, sticky="w", padx=8, pady=(2, 2))
        self.rebound_confirm_var = tk.StringVar(value=f"{getattr(self.state, 'rebound_confirm_pct', REBOUND_CONFIRM_PCT_DEFAULT):.3f}")
        e_reb = ttk.Entry(card, textvariable=self.rebound_confirm_var, width=10)
        e_reb.grid(row=5, column=1, sticky="w", padx=8, pady=(2, 2))
        e_reb.bind("<Return>", self._on_rebound_confirm_pct)
        e_reb.bind("<FocusOut>", self._on_rebound_confirm_pct)

        ttk.Label(card, text="Rebound timeout (sec)").grid(row=6, column=0, sticky="w", padx=8, pady=(2, 2))
        self.rebound_timeout_var = tk.StringVar(value=f"{getattr(self.state, 'dip_rebound_timeout_sec', DIP_REBOUND_TIMEOUT_SEC_DEFAULT):.0f}")
        e_rt = ttk.Entry(card, textvariable=self.rebound_timeout_var, width=10)
        e_rt.grid(row=6, column=1, sticky="w", padx=8, pady=(2, 2))
        e_rt.bind("<Return>", self._on_rebound_timeout)
        e_rt.bind("<FocusOut>", self._on_rebound_timeout)

        ttk.Label(card, text="Higher closes to confirm").grid(row=7, column=0, sticky="w", padx=8, pady=(2, 2))
        self.rebound_confirms_var = tk.StringVar(value=str(getattr(self.state, "dip_confirm_closes", DIP_CONFIRM_CLOSES_DEFAULT)))
        e_rc = ttk.Entry(card, textvariable=self.rebound_confirms_var, width=10)
        e_rc.grid(row=7, column=1, sticky="w", padx=8, pady=(2, 2))
        e_rc.bind("<Return>", self._on_rebound_confirms)
        e_rc.bind("<FocusOut>", self._on_rebound_confirms)

        ttk.Label(
            card,
            text="Wait for a tiny bounce or a few higher closes before buying the dip; timeout to avoid waiting forever.",
            wraplength=self.SIDEBAR_WIDTH - 60,
            justify="left",
            foreground="#666",
        ).grid(row=8, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 10))

        self.edge_toggle_var = tk.BooleanVar(value=bool(getattr(self.state, "use_edge_aware_thresholds", USE_EDGE_AWARE_THRESHOLDS_DEFAULT)))
        ttk.Label(card, text="Edge-aware thresholds").grid(row=9, column=0, sticky="w", padx=8, pady=(4, 2))
        ttk.Checkbutton(card, variable=self.edge_toggle_var, command=self._on_toggle_edge_aware).grid(
            row=9, column=1, sticky="w", padx=8, pady=(4, 2)
        )
        ttk.Label(
            card,
            text="Clamp SELL (and optionally BUY) thresholds to cover fees + spread + slippage + buffer.",
            wraplength=self.SIDEBAR_WIDTH - 60,
            justify="left",
            foreground="#666",
        ).grid(row=10, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 10))

        self.ioc_toggle_var = tk.BooleanVar(value=bool(getattr(self.state, "use_ioc_slippage_cap", USE_IOC_SLIPPAGE_CAP_DEFAULT)))
        ttk.Label(card, text="IOC with slippage cap").grid(row=11, column=0, sticky="w", padx=8, pady=(4, 2))
        ttk.Checkbutton(card, variable=self.ioc_toggle_var, command=self._on_toggle_ioc_cap).grid(
            row=11, column=1, sticky="w", padx=8, pady=(4, 2)
        )
        ttk.Label(
            card,
            text="Use IOC limits with a slip cap to avoid runaway market fills; adjust max slip in Autopilot.",
            wraplength=self.SIDEBAR_WIDTH - 60,
            justify="left",
            foreground="#666",
        ).grid(row=12, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))
        card.columnconfigure(1, weight=1)

        # Autopilot controls
        ap_card = ttk.LabelFrame(parent, text="Autopilot (manual / auto / hybrid)")
        ap_card.pack(fill="x", padx=6, pady=(4, 6))
        self.ap_controls = {}

        def _ap_row(parent_row: int, label: str, key: str, fmt: str = ".3f"):
            mode_var = tk.StringVar(value=_get_ap(self.state, key, 0).mode)
            manual_var = tk.StringVar(value=f"{_get_ap(self.state, key, 0).manual:{fmt}}")
            auto_var = tk.StringVar(value="-")
            eff_var = tk.StringVar(value="-")
            ttk.Label(ap_card, text=label).grid(row=parent_row, column=0, sticky="w", padx=6, pady=2)
            cb = ttk.Combobox(ap_card, values=["manual", "auto", "hybrid"], state="readonly", width=8, textvariable=mode_var)
            cb.grid(row=parent_row, column=1, sticky="w", padx=4, pady=2)
            cb.bind("<<ComboboxSelected>>", lambda _e, k=key, v=mode_var: self._on_autopilot_mode(k, v))
            ent = ttk.Entry(ap_card, textvariable=manual_var, width=8)
            ent.grid(row=parent_row, column=2, sticky="w", padx=4, pady=2)
            ent.bind("<Return>", lambda _e, k=key, v=manual_var: self._on_autopilot_manual(k, v))
            ent.bind("<FocusOut>", lambda _e, k=key, v=manual_var: self._on_autopilot_manual(k, v))
            ttk.Label(ap_card, textvariable=auto_var, width=10).grid(row=parent_row, column=3, sticky="w", padx=4, pady=2)
            ttk.Label(ap_card, textvariable=eff_var, width=10).grid(row=parent_row, column=4, sticky="w", padx=4, pady=2)
            self.ap_controls[key] = {"mode": mode_var, "manual": manual_var, "auto": auto_var, "eff": eff_var, "entry": ent}

        ttk.Label(ap_card, text="Mode").grid(row=0, column=1, padx=4)
        ttk.Label(ap_card, text="Manual").grid(row=0, column=2, padx=4)
        ttk.Label(ap_card, text="Auto").grid(row=0, column=3, padx=4)
        ttk.Label(ap_card, text="Effective").grid(row=0, column=4, padx=4)

        _ap_row(1, "Auto BUY %", "auto_buy_pct", ".3f")
        _ap_row(2, "Trail %", "trail_pct", ".3f")
        _ap_row(3, "TP1 fraction", "tp1_frac", ".3f")
        _ap_row(4, "Soft stop closes", "soft_stop_confirms", ".0f")
        _ap_row(5, "Hard stop x", "hard_stop_mult", ".2f")
        _ap_row(6, "Stop %", "stop_pct", ".3f")
        _ap_row(7, "Max slip %", "max_slip_pct", ".3f")
        for c in range(5):
            ap_card.columnconfigure(c, weight=1)

    def _build_trending_tab(self, parent):
        card = self._card(parent, "Auto Coin (Trending Scanner)")

        self.auto_coin_enabled_var = tk.BooleanVar(value=bool(getattr(self.state, "auto_coin_enabled", AUTO_COIN_ENABLED_DEFAULT)))
        ttk.Label(card, text="Auto Coin (CoinEx Trending Scanner)").grid(row=0, column=0, sticky="w", padx=8, pady=(4, 2))
        ttk.Checkbutton(card, variable=self.auto_coin_enabled_var, command=self._on_auto_coin_toggle).grid(
            row=0, column=1, sticky="w", padx=8, pady=(4, 2)
        )

        ttk.Label(card, text="Switch policy").grid(row=1, column=0, sticky="w", padx=8, pady=(6, 2))
        policy_default = "Switch only when flat"
        if str(getattr(self.state, "auto_coin_policy", AUTO_COIN_POLICY_DEFAULT)) == "force":
            policy_default = "Force close then switch"
        self.auto_coin_policy_var = tk.StringVar(value=policy_default)
        policy_combo = ttk.Combobox(
            card,
            values=["Switch only when flat", "Force close then switch"],
            state="readonly",
            width=24,
            textvariable=self.auto_coin_policy_var,
        )
        policy_combo.grid(row=1, column=1, sticky="w", padx=8, pady=(6, 2))
        policy_combo.bind("<<ComboboxSelected>>", self._on_auto_coin_policy)

        ttk.Label(card, text="Scan interval (sec)").grid(row=2, column=0, sticky="w", padx=8, pady=(4, 2))
        self.auto_coin_scan_var = tk.StringVar(value=f"{float(getattr(self.state, 'auto_coin_scan_interval_sec', AUTO_COIN_SCAN_INTERVAL_SEC_DEFAULT)):.0f}")
        e_scan = ttk.Entry(card, textvariable=self.auto_coin_scan_var, width=10)
        e_scan.grid(row=2, column=1, sticky="w", padx=8, pady=(4, 2))
        e_scan.bind("<Return>", self._on_auto_coin_scan_interval)
        e_scan.bind("<FocusOut>", self._on_auto_coin_scan_interval)

        ttk.Label(card, text="Min dwell (min)").grid(row=3, column=0, sticky="w", padx=8, pady=(2, 2))
        self.auto_coin_dwell_var = tk.StringVar(value=f"{float(getattr(self.state, 'auto_coin_dwell_min', AUTO_COIN_DWELL_MIN_DEFAULT)):.0f}")
        e_dwell = ttk.Entry(card, textvariable=self.auto_coin_dwell_var, width=10)
        e_dwell.grid(row=3, column=1, sticky="w", padx=8, pady=(2, 2))
        e_dwell.bind("<Return>", self._on_auto_coin_dwell)
        e_dwell.bind("<FocusOut>", self._on_auto_coin_dwell)

        ttk.Label(card, text="Hysteresis %").grid(row=4, column=0, sticky="w", padx=8, pady=(2, 2))
        self.auto_coin_hyst_var = tk.StringVar(value=f"{float(getattr(self.state, 'auto_coin_hysteresis_pct', AUTO_COIN_HYSTERESIS_PCT_DEFAULT)):.1f}")
        e_hyst = ttk.Entry(card, textvariable=self.auto_coin_hyst_var, width=10)
        e_hyst.grid(row=4, column=1, sticky="w", padx=8, pady=(2, 2))
        e_hyst.bind("<Return>", self._on_auto_coin_hysteresis)
        e_hyst.bind("<FocusOut>", self._on_auto_coin_hysteresis)

        ttk.Label(card, text="Candidates N").grid(row=5, column=0, sticky="w", padx=8, pady=(2, 2))
        self.auto_coin_candidates_var = tk.StringVar(value=str(int(getattr(self.state, "auto_coin_candidates_n", AUTO_COIN_CANDIDATES_N_DEFAULT))))
        e_cand = ttk.Entry(card, textvariable=self.auto_coin_candidates_var, width=10)
        e_cand.grid(row=5, column=1, sticky="w", padx=8, pady=(2, 2))
        e_cand.bind("<Return>", self._on_auto_coin_candidates)
        e_cand.bind("<FocusOut>", self._on_auto_coin_candidates)

        ttk.Separator(card).grid(row=6, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 6))

        ttk.Label(card, text="Current symbol").grid(row=7, column=0, sticky="w", padx=8, pady=(2, 2))
        self.auto_coin_symbol_var = tk.StringVar(value=SYMBOL)
        ttk.Label(card, textvariable=self.auto_coin_symbol_var).grid(row=7, column=1, sticky="w", padx=8, pady=(2, 2))

        ttk.Label(card, text="Last scan").grid(row=8, column=0, sticky="w", padx=8, pady=(2, 6))
        self.auto_coin_last_scan_var = tk.StringVar(value="-")
        ttk.Label(card, textvariable=self.auto_coin_last_scan_var).grid(row=8, column=1, sticky="w", padx=8, pady=(2, 6))

        ttk.Label(card, text="Last scan summary").grid(row=9, column=0, sticky="w", padx=8, pady=(2, 2))
        self.auto_coin_summary_var = tk.StringVar(value="-")
        ttk.Label(card, textvariable=self.auto_coin_summary_var).grid(row=9, column=1, sticky="w", padx=8, pady=(2, 2))

        ttk.Label(card, text="Top candidates").grid(row=10, column=0, sticky="w", padx=8, pady=(2, 2))
        cols = ("symbol", "atr", "spread", "slip", "value", "score")
        self.auto_coin_table = ttk.Treeview(card, columns=cols, show="headings", height=5)
        self.auto_coin_table.heading("symbol", text="Symbol")
        self.auto_coin_table.heading("atr", text="ATR%")
        self.auto_coin_table.heading("spread", text="Spread bps")
        self.auto_coin_table.heading("slip", text="Slip bps")
        self.auto_coin_table.heading("value", text="24h value")
        self.auto_coin_table.heading("score", text="Score")
        self.auto_coin_table.column("symbol", width=80, anchor="w")
        self.auto_coin_table.column("atr", width=60, anchor="e")
        self.auto_coin_table.column("spread", width=80, anchor="e")
        self.auto_coin_table.column("slip", width=70, anchor="e")
        self.auto_coin_table.column("value", width=80, anchor="e")
        self.auto_coin_table.column("score", width=60, anchor="e")
        self.auto_coin_table.grid(row=11, column=0, columnspan=2, sticky="ew", padx=8, pady=(2, 6))
        self.auto_coin_table.bind("<<TreeviewSelect>>", self._on_auto_coin_table_select)

        ttk.Label(card, text="Manual select").grid(row=12, column=0, sticky="w", padx=8, pady=(2, 2))
        self.auto_coin_select_var = tk.StringVar(value="")
        self.auto_coin_select_combo = ttk.Combobox(
            card, textvariable=self.auto_coin_select_var, values=[], state="readonly", width=20
        )
        self.auto_coin_select_combo.grid(row=12, column=1, sticky="w", padx=8, pady=(2, 2))

        self.auto_coin_switch_btn = ttk.Button(card, text="Switch to selected", command=self._on_auto_coin_manual_switch)
        self.auto_coin_switch_btn.grid(row=13, column=0, columnspan=2, sticky="ew", padx=8, pady=(2, 6))

        ttk.Button(card, text="Scan now", command=self._on_auto_coin_scan).grid(
            row=14, column=0, sticky="ew", padx=8, pady=(2, 6)
        )
        ttk.Button(card, text="Force switch now", command=self._on_auto_coin_force).grid(
            row=14, column=1, sticky="ew", padx=8, pady=(2, 6)
        )

        card.columnconfigure(1, weight=1)

    def _sync_tf_selection(self, tfs: List[str]):
        try:
            self.tf_list.selection_clear(0, "end")
            for tf in tfs:
                if tf in DATA_TIMEFRAMES:
                    idx = DATA_TIMEFRAMES.index(tf)
                    self.tf_list.selection_set(idx)
                    self.tf_list.see(idx)
        except Exception:
            pass

    def _sync_profile_to_ui(self):
        """Apply the current profile to UI widgets (timeframe selection + enabled/disabled fields)."""
        try:
            profile = str(self.profile_var.get() or ADAPTIVE_PROFILE_DEFAULT)
        except Exception:
            profile = ADAPTIVE_PROFILE_DEFAULT

        if profile in ADAPTIVE_PRESETS and profile != "Custom":
            preset = ADAPTIVE_PRESETS[profile]
            self._sync_tf_selection(list(preset["timeframes"]))
            self.data_lookback_var.set(str(int(preset["lookback"])))
            self.data_refresh_var.set(str(int(preset["refresh"])))

            # Show preset coefficients (read-only)
            self.k_buy_var.set(f"{float(preset['k_buy']):.3f}")
            self.k_sell_var.set(f"{float(preset['k_sell']):.3f}")
            self.alpha_var.set(f"{float(preset['alpha']):.3f}")

            try:
                self.tf_list.configure(state="disabled")
            except Exception:
                pass
            # disable custom coeff entries
            self._set_custom_entries_state("disabled")
            try:
                self.lb_entry.configure(state="disabled")
                self.rf_entry.configure(state="disabled")
            except Exception:
                pass
        else:
            # Custom
            with self.state.lock:
                tfs = list(getattr(self.state, "adaptive_timeframes", []) or [])
            if not tfs:
                tfs = [DATA_TIMEFRAME_DEFAULT]
            self._sync_tf_selection(tfs)
            try:
                self.tf_list.configure(state="normal")
            except Exception:
                pass
            self._set_custom_entries_state("normal")
            try:
                self.lb_entry.configure(state="normal")
                self.rf_entry.configure(state="normal")
            except Exception:
                pass

    def _set_custom_entries_state(self, st: str):
        st = "normal" if st not in ("normal", "disabled") else st
        try:
            self.k_buy_entry.configure(state=st)
            self.k_sell_entry.configure(state=st)
            self.alpha_entry.configure(state=st)
        except Exception:
            pass

    def on_threshold_mode_change(self):
        want_data = (self.threshold_mode_var.get() == "data")
        with self.state.lock:
            self.state.data_driven = bool(want_data)
            # Force refresh when turning on
            if want_data:
                self.state._last_data_refresh_ts = 0.0

        # Disable manual threshold boxes while data-driven is active
        try:
            st = "disabled" if want_data else "normal"
            self.buy_threshold_entry.configure(state=st)
            self.sell_threshold_entry.configure(state=st)
        except Exception:
            pass

        save_settings(self.state)

    def on_profile_change(self, _evt=None):
        profile = str(self.profile_var.get() or ADAPTIVE_PROFILE_DEFAULT)

        with self.state.lock:
            self.state.adaptive_profile = profile
            # Apply preset defaults into state (so persistence + trading loop are consistent)
            if profile in ADAPTIVE_PRESETS and profile != "Custom":
                p = ADAPTIVE_PRESETS[profile]
                self.state.adaptive_timeframes = list(p["timeframes"])
                self.state.adaptive_weights = list(p.get("weights") or [])
                self.state.data_lookback = int(p["lookback"])
                self.state.data_refresh_sec = int(p["refresh"])
                self.state.adaptive_k_buy = float(p["k_buy"])
                self.state.adaptive_k_sell = float(p["k_sell"])
                self.state.adaptive_alpha = float(p["alpha"])
            else:
                # Custom: keep whatever the user already selected (ensure >= 1 tf)
                tfs = list(getattr(self.state, "adaptive_timeframes", []) or [])
                if not tfs:
                    self.state.adaptive_timeframes = [DATA_TIMEFRAME_DEFAULT]

            self.state._last_data_refresh_ts = 0.0

        # Update UI widgets to match the selected profile
        self._sync_profile_to_ui()
        save_settings(self.state)

    def on_tf_click(self, event):
        """Click-to-toggle timeframes (Custom profile only)."""
        profile = str(self.profile_var.get() or ADAPTIVE_PROFILE_DEFAULT)
        if profile in ADAPTIVE_PRESETS and profile != "Custom":
            return "break"

        try:
            idx = self.tf_list.nearest(event.y)
        except Exception:
            return "break"

        # Toggle selection
        if idx in self.tf_list.curselection():
            self.tf_list.selection_clear(idx)
        else:
            self.tf_list.selection_set(idx)

        selected = [self.tf_list.get(i) for i in self.tf_list.curselection()]
        if not selected:
            # Always keep at least one timeframe
            self.tf_list.selection_set(idx)
            selected = [self.tf_list.get(idx)]

        with self.state.lock:
            self.state.adaptive_timeframes = [str(x) for x in selected]
            self.state._last_data_refresh_ts = 0.0

        save_settings(self.state)
        return "break"

    def on_data_params_change(self, _evt=None):
        try:
            lb = int(float(self.data_lookback_var.get().strip()))
            rf = int(float(self.data_refresh_var.get().strip()))
        except Exception:
            return

        lb = max(30, min(lb, 600))
        rf = max(5, min(rf, 900))

        with self.state.lock:
            self.state.data_lookback = lb
            self.state.data_refresh_sec = rf
            self.state._last_data_refresh_ts = 0.0

        self.data_lookback_var.set(str(lb))
        self.data_refresh_var.set(str(rf))
        save_settings(self.state)

    def on_custom_coeff_change(self, _evt=None):
        profile = str(self.profile_var.get() or ADAPTIVE_PROFILE_DEFAULT)
        if profile in ADAPTIVE_PRESETS and profile != "Custom":
            return

        try:
            kb = float(self.k_buy_var.get().strip())
            ks = float(self.k_sell_var.get().strip())
            a = float(self.alpha_var.get().strip())
        except Exception:
            return

        kb = max(0.1, min(kb, 25.0))
        ks = max(0.1, min(ks, 25.0))
        a = max(0.01, min(a, 0.90))

        with self.state.lock:
            self.state.adaptive_k_buy = kb
            self.state.adaptive_k_sell = ks
            self.state.adaptive_alpha = a
            self.state._last_data_refresh_ts = 0.0

        self.k_buy_var.set(f"{kb:.3f}")
        self.k_sell_var.set(f"{ks:.3f}")
        self.alpha_var.set(f"{a:.3f}")
        save_settings(self.state)

    def force_adaptive_refresh(self):
        with self.state.lock:
            self.state._last_data_refresh_ts = 0.0
        save_settings(self.state)
        log_event(self.state, "Adaptive refresh requested (will update in trading loop).")


    def _build_buttons_card(self, parent):
        card = self._card(parent, "Controls")

        def themed_button(text, cmd, style=None):
            if tb is not None and style:
                return tb.Button(card, text=text, command=cmd, bootstyle=style)
            return ttk.Button(card, text=text, command=cmd)

        # row 0
        themed_button("Use Market Orders", lambda: self.send_cmd("market"), "success-outline").grid(row=0, column=0, sticky="ew", padx=8, pady=4)
        themed_button("Use Limit Orders", lambda: self.send_cmd("limit"), "info-outline").grid(row=0, column=1, sticky="ew", padx=8, pady=4)

        # row 1
        themed_button("Manual BUY", lambda: self.send_cmd("manual_buy"), "success").grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        themed_button("Manual SELL", lambda: self.send_cmd("manual_sell"), "danger").grid(row=1, column=1, sticky="ew", padx=8, pady=4)

        # row 2
        themed_button("Cancel Limit", lambda: self.send_cmd("cancel"), "warning-outline").grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        themed_button("Pause / Resume", self.toggle_pause, "secondary").grid(row=2, column=1, sticky="ew", padx=8, pady=4)

        # row 3
        themed_button("Restart Bot", self.restart_bot, "primary").grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        themed_button("Quit", self.quit_bot, "secondary-outline").grid(row=3, column=1, sticky="ew", padx=8, pady=4)

        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)

    def _build_chart(self, parent):
        chart_frame = ttk.LabelFrame(parent, text="Price Chart")
        chart_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)

        # Slightly narrower figure so the left control panel can be wider
        self.fig = Figure(figsize=(5.5, 4.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(SYMBOL)
        self.ax.grid(True, alpha=0.2)

        self.line_close, = self.ax.plot([], [], linewidth=1.5)
        self.line_buy = self.ax.axhline(y=0, linewidth=1.0, linestyle="--", alpha=0.6)
        self.line_sell = self.ax.axhline(y=0, linewidth=1.0, linestyle="--", alpha=0.6)
        self.line_entry = self.ax.axhline(y=0, linewidth=1.0, linestyle=":", alpha=0.6)

        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.ax.tick_params(axis="x", rotation=0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

    
    def _build_info_panels(self, parent):
        """Right-side info row directly under the chart.

        Layout: [Position / Triggers] [Symbol] [Trade coin from wallet]
        """
        info = ttk.Frame(parent)
        info.grid(row=1, column=0, sticky="ew", padx=0, pady=(10, 0))
        info.columnconfigure(0, weight=1)
        info.columnconfigure(1, weight=1)
        info.columnconfigure(2, weight=1)

        # --- Position / Triggers (small box) ---
        self.position_frame = ttk.LabelFrame(info, text="Position / Triggers")
        self.position_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.position_text_var = tk.StringVar(value="No data yet")
        ttk.Label(
            self.position_frame,
            textvariable=self.position_text_var,
            justify="left",
        ).pack(fill="x", padx=8, pady=8)

        # --- Symbol box ---
        symbol_frame = ttk.LabelFrame(info, text="Symbol")
        symbol_frame.grid(row=0, column=1, sticky="nsew", padx=3)
        self.symbol_var = tk.StringVar(value=SYMBOL)
        ttk.Label(symbol_frame, text="Symbol (e.g. ENA/USDT)").pack(anchor="w", padx=8, pady=(6, 2))
        self.symbol_entry = ttk.Entry(symbol_frame, textvariable=self.symbol_var)
        self.symbol_entry.pack(fill="x", padx=8, pady=(0, 4))
        self.set_symbol_btn = ttk.Button(symbol_frame, text="Set Symbol", command=self.on_set_symbol)
        self.set_symbol_btn.pack(
            fill="x", padx=8, pady=(0, 6)
        )

        # --- Trade coin from wallet box ---
        wallet_sel_frame = ttk.LabelFrame(info, text="Trade coin from wallet")
        wallet_sel_frame.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        ttk.Label(wallet_sel_frame, text="Coin").pack(anchor="w", padx=8, pady=(6, 2))
        self.wallet_coin_combo = ttk.Combobox(wallet_sel_frame, values=[], state="readonly", width=10)
        self.wallet_coin_combo.pack(fill="x", padx=8, pady=(0, 4))
        self.wallet_coin_btn = ttk.Button(wallet_sel_frame, text="Use Wallet Coin", command=self.on_use_wallet_coin)
        self.wallet_coin_btn.pack(
            fill="x", padx=8, pady=(0, 6)
        )
    # ---------- callbacks ----------
    def send_cmd(self, cmd: str):
        self.cmd_queue.put(cmd)

    def on_apply_api(self):
        key = self.api_key_var.get().strip()
        secret = self.api_secret_var.get().strip()
        if not key or not secret:
            log_event(self.state, "API key/secret not provided; keeping current.")
            return
        self.send_cmd(f"set_api:{key}|{secret}")
        log_event(self.state, "API update requested (will apply in trading loop).")

    def _toggle_dry_run(self):
        # Push to trading thread; command handler enforces safety (flat/no open order).
        want = bool(self.dry_run_var.get())
        self.cmd_queue.put(f"dry_run:{1 if want else 0}")

    def _on_reset_dry_run(self):
        self.cmd_queue.put("reset_dry_run")

    def _on_toggle_tp1_trailing(self):
        want = bool(self.tp1_toggle_var.get())
        with self.state.lock:
            self.state.use_tp1_trailing = want
        save_settings(self.state)
        log_event(self.state, f"TP1 + trailing exits {'ENABLED' if want else 'DISABLED'}")

    def _on_toggle_soft_stop(self):
        want = bool(self.soft_stop_toggle_var.get())
        with self.state.lock:
            self.state.use_soft_stop = want
            self.state.soft_stop_counter = 0
        save_settings(self.state)
        log_event(self.state, f"Soft stop {'ENABLED' if want else 'DISABLED'} (hard stop always active when SL set)")

    def _on_toggle_dip_rebound(self):
        want = bool(self.dip_rebound_toggle_var.get())
        with self.state.lock:
            self.state.use_dip_rebound = want
            self.state.dip_wait_start_ts = 0.0
            self.state.dip_low = 0.0
            self.state.dip_higher_close_count = 0
        save_settings(self.state)
        log_event(self.state, f"Dip rebound confirmation {'ENABLED' if want else 'DISABLED'}")

    def _on_rebound_confirm_pct(self, event=None):
        raw = self.rebound_confirm_var.get().strip()
        try:
            v = max(0.0, float(raw))
        except Exception:
            return
        with self.state.lock:
            self.state.rebound_confirm_pct = v
        self.rebound_confirm_var.set(f"{v:.3f}")
        save_settings(self.state)
        log_event(self.state, f"Rebound confirm set to {v:.3f}%")

    def _on_rebound_timeout(self, event=None):
        raw = self.rebound_timeout_var.get().strip()
        try:
            v = max(5.0, float(raw))
        except Exception:
            return
        with self.state.lock:
            self.state.dip_rebound_timeout_sec = v
        self.rebound_timeout_var.set(f"{v:.0f}")
        save_settings(self.state)
        log_event(self.state, f"Rebound timeout set to {v:.0f}s")

    def _on_rebound_confirms(self, event=None):
        raw = self.rebound_confirms_var.get().strip()
        try:
            v = max(1, int(float(raw)))
        except Exception:
            return
        with self.state.lock:
            self.state.dip_confirm_closes = v
        self.rebound_confirms_var.set(str(v))
        save_settings(self.state)
        log_event(self.state, f"Dip confirm closes set to {v}")

    def _on_toggle_edge_aware(self):
        want = bool(self.edge_toggle_var.get())
        with self.state.lock:
            self.state.use_edge_aware_thresholds = want
        save_settings(self.state)
        log_event(self.state, f"Edge-aware thresholds {'ENABLED' if want else 'DISABLED'}")

    def _on_toggle_ioc_cap(self):
        want = bool(self.ioc_toggle_var.get())
        with self.state.lock:
            self.state.use_ioc_slippage_cap = want
        save_settings(self.state)
        log_event(self.state, f"IOC slippage cap {'ENABLED' if want else 'DISABLED'}")

    def _on_auto_coin_toggle(self):
        want = bool(self.auto_coin_enabled_var.get())
        with self.state.lock:
            self.state.auto_coin_enabled = want
            if want:
                self.state.auto_coin_last_scan_ts = 0.0
                self.state.auto_coin_last_winner = ""
                self.state.auto_coin_winner_streak = 0
                self.state.auto_coin_top_candidates = []
        save_settings(self.state)
        log_event(self.state, f"Auto-coin scanner {'ENABLED' if want else 'DISABLED'}")

    def _on_auto_coin_policy(self, event=None):
        label = str(self.auto_coin_policy_var.get())
        policy = "force" if label.lower().startswith("force") else "flat"
        with self.state.lock:
            self.state.auto_coin_policy = policy
        save_settings(self.state)
        log_event(self.state, f"Auto-coin policy set to {policy}.")

    def _on_auto_coin_scan_interval(self, event=None):
        raw = self.auto_coin_scan_var.get().strip()
        try:
            v = max(10.0, float(raw))
        except Exception:
            return
        with self.state.lock:
            self.state.auto_coin_scan_interval_sec = v
        self.auto_coin_scan_var.set(f"{v:.0f}")
        save_settings(self.state)
        log_event(self.state, f"Auto-coin scan interval set to {v:.0f}s.")

    def _on_auto_coin_dwell(self, event=None):
        raw = self.auto_coin_dwell_var.get().strip()
        try:
            v = max(0.0, float(raw))
        except Exception:
            return
        with self.state.lock:
            self.state.auto_coin_dwell_min = v
        self.auto_coin_dwell_var.set(f"{v:.0f}")
        save_settings(self.state)
        log_event(self.state, f"Auto-coin min dwell set to {v:.0f}m.")

    def _on_auto_coin_hysteresis(self, event=None):
        raw = self.auto_coin_hyst_var.get().strip()
        try:
            v = max(0.0, float(raw))
        except Exception:
            return
        with self.state.lock:
            self.state.auto_coin_hysteresis_pct = v
        self.auto_coin_hyst_var.set(f"{v:.1f}")
        save_settings(self.state)
        log_event(self.state, f"Auto-coin hysteresis set to {v:.1f}%.")

    def _on_auto_coin_candidates(self, event=None):
        raw = self.auto_coin_candidates_var.get().strip()
        try:
            v = max(5, int(float(raw)))
        except Exception:
            return
        with self.state.lock:
            self.state.auto_coin_candidates_n = v
        self.auto_coin_candidates_var.set(str(v))
        save_settings(self.state)
        log_event(self.state, f"Auto-coin candidates N set to {v}.")

    def _on_auto_coin_scan(self):
        self.send_cmd("auto_coin_scan")
        log_event(self.state, "Auto-coin scan requested.")

    def _on_auto_coin_force(self):
        self.send_cmd("auto_coin_force")
        log_event(self.state, "Auto-coin force switch requested.")

    def _on_auto_coin_table_select(self, event=None):
        try:
            if not hasattr(self, "auto_coin_table") or not hasattr(self, "auto_coin_select_var"):
                return
            sel = self.auto_coin_table.selection()
            if not sel:
                return
            values = self.auto_coin_table.item(sel[0], "values") or []
            sym = str(values[0]).strip() if values else ""
            if sym:
                self.auto_coin_select_var.set(sym)
        except Exception:
            pass

    def _on_auto_coin_manual_switch(self):
        sym = ""
        if hasattr(self, "auto_coin_select_var"):
            sym = self.auto_coin_select_var.get().strip()
        if sym:
            self.send_cmd(f"symbol:{sym}")

    def _on_autopilot_mode(self, key: str, var: tk.StringVar):
        mode = var.get().strip().lower()
        if mode not in ("manual", "auto", "hybrid"):
            return
        with self.state.lock:
            ap = _get_ap(self.state, key, 0.0)
            ap.mode = mode
            self.state.autopilot[key] = ap
        save_settings(self.state)
        log_event(self.state, f"Autopilot {key} mode -> {mode}")
        # Gray out manual entry when in auto
        try:
            ctrl = self.ap_controls.get(key)
            if ctrl and "entry" in ctrl:
                ctrl["entry"].configure(state="disabled" if mode == "auto" else "normal")
        except Exception:
            pass

    def _on_autopilot_manual(self, key: str, var: tk.StringVar):
        raw = var.get().strip()
        try:
            v = float(raw)
        except Exception:
            return
        now_ts = time.time()
        with self.state.lock:
            ap = _get_ap(self.state, key, v)
            ap.manual = v
            ap.last_manual_ts = now_ts
            ap.effective = v if ap.mode == "manual" else ap.effective
            self.state.autopilot[key] = ap
            if key == "trail_pct":
                self.state.effective_trail_pct = v
            elif key == "tp1_frac":
                self.state.effective_tp1_frac = v
            elif key == "soft_stop_confirms":
                self.state.soft_stop_confirms = int(round(v))
            elif key == "hard_stop_mult":
                self.state.effective_hard_stop_mult = v
            elif key == "stop_pct":
                self.state.stop_loss_effective_pct = v
                globals().__setitem__("STOP_LOSS_PCT", v)
            elif key == "auto_buy_pct":
                self.state.auto_buy_pct = max(0.0, min(v / 100.0, 1.0))
                self.state.auto_buy_pct_effective = self.state.auto_buy_pct
                if hasattr(self, "auto_buy_pct_var"):
                    self.auto_buy_pct_var.set(f"{self.state.auto_buy_pct * 100.0:.1f}")
            elif key == "max_slip_pct":
                self.state.max_slip_pct = v
        var.set(f"{v:.3f}")
        save_settings(self.state)
        log_event(self.state, f"Autopilot {key} manual -> {v}")


    def on_buy_threshold_change(self, value: str):
        try:
            v = float(value)
        except ValueError:
            return
        v = max(0.1, min(v, 10.0))
        self.state.buy_threshold_pct = v

    def on_sell_threshold_change(self, value: str):
        try:
            v = float(value)
        except ValueError:
            return
        v = max(0.1, min(v, 10.0))
        self.state.sell_threshold_pct = v

    def on_buy_threshold_entry(self, event=None):
        raw = self.buy_threshold_var.get().strip()
        try:
            v = float(raw)
        except ValueError:
            return
        v = max(0.01, min(v, 20.0))
        self.state.buy_threshold_pct = v
        self.buy_threshold_var.set(f"{v:.3f}")
        save_settings(self.state)

    def on_sell_threshold_entry(self, event=None):
        raw = self.sell_threshold_var.get().strip()
        try:
            v = float(raw)
        except ValueError:
            return
        v = max(0.01, min(v, 20.0))
        self.state.sell_threshold_pct = v
        self.sell_threshold_var.set(f"{v:.3f}")
        save_settings(self.state)

    def on_auto_buy_entry(self, event=None):
        raw = self.auto_buy_pct_var.get().strip()
        try:
            v = float(raw)
        except ValueError:
            return
        v = max(0.0, min(v, 100.0))
        self.state.auto_buy_pct = v / 100.0
        self.auto_buy_pct_var.set(f"{v:.1f}")
        try:
            ap = _get_ap(self.state, "auto_buy_pct", v)
            ap.manual = v
            ap.last_manual_ts = time.time()
            if ap.mode == "manual":
                ap.effective = v
            self.state.autopilot["auto_buy_pct"] = ap
        except Exception:
            pass
        save_settings(self.state)

    def on_manual_buy_entry(self, event=None):
        raw = self.manual_buy_pct_var.get().strip()
        try:
            v = float(raw)
        except ValueError:
            return
        v = max(0.0, min(v, 100.0))
        self.state.manual_buy_pct = v / 100.0
        self.manual_buy_pct_var.set(f"{v:.1f}")
        save_settings(self.state)

    def on_manual_sell_entry(self, event=None):
        raw = self.manual_sell_pct_var.get().strip()
        try:
            v = float(raw)
        except ValueError:
            return
        v = max(0.0, min(v, 100.0))
        self.state.manual_sell_pct = v / 100.0
        self.manual_sell_pct_var.set(f"{v:.1f}")
        save_settings(self.state)


    def on_set_symbol(self):
        sym = self.symbol_var.get().strip()
        if sym:
            self.send_cmd(f"symbol:{sym}")

    def on_use_wallet_coin(self):
        asset = self.wallet_coin_combo.get().strip()
        if asset:
            self.send_cmd(f"wallet_coin:{asset}")
            self.symbol_var.set(f"{asset}/USDT")

    def toggle_pause(self):
        if self.state.trading_paused_until and time.time() < self.state.trading_paused_until:
            self.send_cmd("resume")
        else:
            self.send_cmd("pause")

    def on_apply_config(self):
        global STOP_LOSS_PCT, DAILY_MAX_LOSS_USDT, DAILY_MAX_PROFIT_USDT
        global ANCHOR_MAX_AGE_SEC, ANCHOR_DRIFT_REARM_PCT
        global POLL_INTERVAL_SECONDS, MIN_NOTIONAL_USDT
        global MAX_STOPS_PER_HOUR, CIRCUIT_STOP_COOLDOWN_SEC

        sl = self.stop_loss_var.get().strip()
        if sl == "":
            STOP_LOSS_PCT = None
        else:
            try:
                STOP_LOSS_PCT = max(0.0, float(sl))
            except ValueError:
                pass
        try:
            self.state.stop_loss_effective_pct = STOP_LOSS_PCT
            ap = _get_ap(self.state, "stop_pct", float(STOP_LOSS_PCT or 0.0))
            ap.manual = float(STOP_LOSS_PCT or 0.0)
            ap.last_manual_ts = time.time()
            if ap.mode == "manual":
                ap.effective = ap.manual
            self.state.autopilot["stop_pct"] = ap
        except Exception:
            pass

        for (var, setter) in [
            (self.daily_loss_var, lambda x: globals().__setitem__("DAILY_MAX_LOSS_USDT", float(x))),
            (self.daily_profit_var, lambda x: globals().__setitem__("DAILY_MAX_PROFIT_USDT", float(x))),
            (self.anchor_age_var, lambda x: globals().__setitem__("ANCHOR_MAX_AGE_SEC", max(1.0, float(x)))),
            (self.anchor_drift_var, lambda x: globals().__setitem__("ANCHOR_DRIFT_REARM_PCT", max(0.0, float(x)))),
            (self.poll_interval_var, lambda x: globals().__setitem__("POLL_INTERVAL_SECONDS", max(0.5, float(x)))),
            (self.min_notional_var, lambda x: globals().__setitem__("MIN_NOTIONAL_USDT", max(0.0, float(x)))),
            (self.max_stops_var, lambda x: globals().__setitem__("MAX_STOPS_PER_HOUR", max(0, int(float(x))))),
            (self.circuit_cooldown_var, lambda x: globals().__setitem__("CIRCUIT_STOP_COOLDOWN_SEC", max(0.0, float(x)))),
        ]:
            try:
                setter(var.get().strip())
            except Exception:
                pass

        log_event(
            self.state,
            "Advanced updated: "
            f"SL={STOP_LOSS_PCT}, "
            f"daily_loss={DAILY_MAX_LOSS_USDT}, daily_profit={DAILY_MAX_PROFIT_USDT}, "
            f"anchor_max_age={ANCHOR_MAX_AGE_SEC}s, anchor_drift={ANCHOR_DRIFT_REARM_PCT}%, "
            f"poll_interval={POLL_INTERVAL_SECONDS}s, min_notional={MIN_NOTIONAL_USDT}, "            f"max_stops_per_hr={MAX_STOPS_PER_HOUR}, cooldown={CIRCUIT_STOP_COOLDOWN_SEC}s"
        )
        save_settings(self.state)

    def restart_bot(self):
        save_settings(self.state)
        log_event(self.state, "Restarting bot process...")
        self.stop_event.set()
        self.root.after(400, self._do_restart)

    def _do_restart(self):
        self.root.destroy()
        os.execl(sys.executable, sys.executable, *sys.argv)

    def quit_bot(self):
        save_settings(self.state)
        self.stop_event.set()
        self.root.after(250, self.root.destroy)

    # ---------- UI update ----------

    
    def _update_chart(self, times, close, buy_y, sell_y, entry_y):
        if not times or not close:
            return

        # Zoom in on the most recent candles so the chart is easier to read
        window = 60  # number of recent candles to show (smaller than HISTORY_LIMIT)
        if len(times) > window:
            times_plot = times[-window:]
            close_plot = close[-window:]
        else:
            times_plot = times
            close_plot = close

        # Update close line
        self.line_close.set_data(times_plot, close_plot)

        # Update trigger lines (same price levels regardless of zoom window)
        if buy_y is not None and buy_y > 0:
            self.line_buy.set_ydata([buy_y, buy_y])
            self.line_buy.set_visible(True)
        else:
            self.line_buy.set_visible(False)

        if sell_y is not None and sell_y > 0:
            self.line_sell.set_ydata([sell_y, sell_y])
            self.line_sell.set_visible(True)
        else:
            self.line_sell.set_visible(False)

        if entry_y is not None and entry_y > 0:
            self.line_entry.set_ydata([entry_y, entry_y])
            self.line_entry.set_visible(True)
        else:
            self.line_entry.set_visible(False)

        # Compute a tight y-range around the visible prices and trigger lines
        ys = list(close_plot)
        for y in (buy_y, sell_y, entry_y):
            if y is not None and y > 0:
                ys.append(y)

        if ys:
            ymin = min(ys)
            ymax = max(ys)
            if ymin == ymax:
                center = ymin
                margin = max(abs(center) * 0.01, 0.0001)
                ymin = center - margin
                ymax = center + margin
            else:
                margin = (ymax - ymin) * 0.03
                ymin -= margin
                ymax += margin
            self.ax.set_ylim(ymin, ymax)

        self.ax.set_title(SYMBOL)
        self.ax.set_xlim(times_plot[0], times_plot[-1])

        self.canvas.draw_idle()
    def update_ui(self):
        s = self.state

        # snapshot under lock
        with s.lock:
            last_price = s.last_price
            equity = s.wallet_equity
            pnl = s.daily_realized_pnl
            pnl_total = float(getattr(s, 'total_realized_pnl', 0.0))
            total_runtime_sec = float(getattr(s, 'total_runtime_sec', 0.0))
            session_start_ts = float(getattr(s, 'session_start_ts', time.time()))
            slip_bps = float(getattr(s, 'last_slippage_bps', 0.0))
            slip_side = str(getattr(s, 'last_slippage_side', ''))
            slip_best = float(getattr(s, 'last_slippage_best', 0.0))
            slip_fill = float(getattr(s, 'last_slippage_fill', 0.0))
            slip_levels = int(getattr(s, 'last_slippage_levels', 0) or 0)
            slip_ts = float(getattr(s, 'last_slippage_ts', 0.0))
            order_mode = s.order_mode
            status = s.status
            dry_run = s.dry_run
            wallet_rows = list(s.wallet_rows)
            position = s.position
            last_trade_price = s.last_trade_price
            data_driven = s.data_driven
            eff_buy = s.effective_buy_threshold_pct
            eff_sell = s.effective_sell_threshold_pct
            avg_high = s.avg_high
            avg_low = s.avg_low
            avg_range = s.avg_range_pct
            trend = s.data_trend
            data_tf = s.data_timeframe
            profile = getattr(s, 'adaptive_profile', ADAPTIVE_PROFILE_DEFAULT)
            tfs = list(getattr(s, 'adaptive_timeframes', []) or [])
            use_tp1_trailing = bool(getattr(s, "use_tp1_trailing", True))
            use_dip_rebound = bool(getattr(s, "use_dip_rebound", True))
            use_edge = bool(getattr(s, "use_edge_aware_thresholds", USE_EDGE_AWARE_THRESHOLDS_DEFAULT))
            use_ioc_cap = bool(getattr(s, "use_ioc_slippage_cap", USE_IOC_SLIPPAGE_CAP_DEFAULT))
            buy_thr = eff_buy if data_driven else s.buy_threshold_pct
            sell_thr = eff_sell if data_driven else s.sell_threshold_pct
            pending_buy_price = s.pending_buy_price
            manual_buy_pct = s.manual_buy_pct
            auto_buy_pct = s.auto_buy_pct
            sim_balance_usdt = s.sim_balance_usdt
            chart_times = list(s.chart_times)
            chart_close = list(s.chart_close)
            auto_coin_enabled = bool(getattr(s, "auto_coin_enabled", AUTO_COIN_ENABLED_DEFAULT))
            auto_coin_last_scan_ts = float(getattr(s, "auto_coin_last_scan_ts", 0.0))
            auto_coin_top_candidates = list(getattr(s, "auto_coin_top_candidates", []) or [])
            auto_coin_policy = str(getattr(s, "auto_coin_policy", AUTO_COIN_POLICY_DEFAULT))
            auto_coin_last_scan_info = str(getattr(s, "auto_coin_last_scan_info", "") or "")
            anchor_ts = s.anchor_timestamp
            dip_wait_start_ts = s.dip_wait_start_ts
            dip_timeout_sec = float(getattr(s, "dip_rebound_timeout_sec", DIP_REBOUND_TIMEOUT_SEC_DEFAULT))
            paused_until = s.trading_paused_until
            paused_reason = s.paused_reason
            next_trade_time = s.next_trade_time
            last_candle_ts = s.last_candle_ts
            day_start_date = s.day_start_date
            data_refresh_sec = float(getattr(s, "data_refresh_sec", DATA_REFRESH_SEC_DEFAULT))
            last_data_refresh_ts = float(getattr(s, "_last_data_refresh_ts", 0.0))
            auto_coin_scan_interval_sec = float(getattr(s, "auto_coin_scan_interval_sec", AUTO_COIN_SCAN_INTERVAL_SEC_DEFAULT))
            auto_coin_last_switch_ts = float(getattr(s, "auto_coin_last_switch_ts", 0.0))
            auto_coin_dwell_min = float(getattr(s, "auto_coin_dwell_min", AUTO_COIN_DWELL_MIN_DEFAULT))

        self.root.title(f"{SYMBOL} · Auto Scalper")
        self.status_var.set(status)
        self.order_mode_var.set(order_mode.upper())
        try:
            self.dry_run_var.set(bool(dry_run))
        except Exception:
            pass
        self.price_var.set(f"{last_price:.8f}" if last_price else "-")
        self.equity_var.set(f"{equity:.2f}")
        self.pnl_var.set(f"{pnl:.2f}")
        try:
            self.pnl_total_var.set(f"{pnl_total:.2f}")
            uptime_total = max(0.0, total_runtime_sec + (time.time() - session_start_ts))
            hh = int(uptime_total // 3600)
            mm = int((uptime_total % 3600) // 60)
            ss = int(uptime_total % 60)
            self.uptime_var.set(f"{hh:02d}:{mm:02d}:{ss:02d}")
            if slip_side:
                if slip_best > 0 and slip_fill > 0:
                    self.slippage_var.set(f"{slip_side} {slip_bps:.1f} bps (best {slip_best:.8f} → fill {slip_fill:.8f})")
                else:
                    self.slippage_var.set(f"{slip_side} {slip_bps:.1f} bps")
            else:
                self.slippage_var.set("-")
            if hasattr(self, "tp1_toggle_var"):
                try:
                    self.tp1_toggle_var.set(bool(use_tp1_trailing))
                except Exception:
                    pass
            if hasattr(self, "soft_stop_toggle_var"):
                try:
                    self.soft_stop_toggle_var.set(bool(getattr(s, "use_soft_stop", True)))
                except Exception:
                    pass
            if hasattr(self, "dip_rebound_toggle_var"):
                try:
                    self.dip_rebound_toggle_var.set(bool(use_dip_rebound))
                except Exception:
                    pass
            if hasattr(self, "rebound_confirm_var"):
                try:
                    self.rebound_confirm_var.set(f"{float(getattr(s, 'rebound_confirm_pct', REBOUND_CONFIRM_PCT_DEFAULT)):.3f}")
                except Exception:
                    pass
            if hasattr(self, "rebound_timeout_var"):
                try:
                    self.rebound_timeout_var.set(f"{float(getattr(s, 'dip_rebound_timeout_sec', DIP_REBOUND_TIMEOUT_SEC_DEFAULT)):.0f}")
                except Exception:
                    pass
            if hasattr(self, "rebound_confirms_var"):
                try:
                    self.rebound_confirms_var.set(str(int(getattr(s, "dip_confirm_closes", DIP_CONFIRM_CLOSES_DEFAULT))))
                except Exception:
                    pass
            if hasattr(self, "edge_toggle_var"):
                try:
                    self.edge_toggle_var.set(bool(use_edge))
                except Exception:
                    pass
            if hasattr(self, "ioc_toggle_var"):
                try:
                    self.ioc_toggle_var.set(bool(use_ioc_cap))
                except Exception:
                    pass
            if hasattr(self, "ap_controls"):
                defaults_map = {
                    "trail_pct": getattr(self.state, "effective_trail_pct", TRAIL_STOP_PCT),
                    "tp1_frac": getattr(self.state, "effective_tp1_frac", TP1_SELL_FRACTION),
                    "soft_stop_confirms": float(getattr(self.state, "soft_stop_confirms", SOFT_STOP_CONFIRMS_DEFAULT)),
                    "hard_stop_mult": getattr(self.state, "effective_hard_stop_mult", HARD_STOP_MULT_DEFAULT),
                    "max_slip_pct": float(getattr(self.state, "max_slip_pct", MAX_SLIP_PCT_DEFAULT)),
                    "stop_pct": float(getattr(self.state, "stop_loss_effective_pct", STOP_LOSS_PCT if STOP_LOSS_PCT is not None else 0.0)),
                    "auto_buy_pct": float(getattr(self.state, "auto_buy_pct_effective", s.auto_buy_pct * 100.0)),
                }
                for key, vars_map in self.ap_controls.items():
                    try:
                        ap = _get_ap(self.state, key, defaults_map.get(key, 0.0))
                        vars_map["mode"].set(ap.mode)
                        vars_map["manual"].set(f"{ap.manual:.3f}")
                        vars_map["auto"].set(f"{ap.auto:.3f}")
                        vars_map["eff"].set(f"{ap.effective:.3f}")
                        if "entry" in vars_map:
                            vars_map["entry"].configure(state="disabled" if ap.mode == "auto" else "normal")
                    except Exception:
                        continue
        except Exception:
            pass

        # Enable dry-run reset button only when in DRY_RUN
        try:
            if hasattr(self, "reset_dry_run_btn"):
                self.reset_dry_run_btn.configure(state="normal" if dry_run else "disabled")
        except Exception:
            pass

        # Auto coin UI + symbol controls
        try:
            if hasattr(self, "auto_coin_enabled_var"):
                self.auto_coin_enabled_var.set(bool(auto_coin_enabled))
            if hasattr(self, "auto_coin_policy_var"):
                policy_label = "Force close then switch" if auto_coin_policy == "force" else "Switch only when flat"
                self.auto_coin_policy_var.set(policy_label)
            if hasattr(self, "auto_coin_symbol_var"):
                badge = " (Auto)" if auto_coin_enabled else ""
                self.auto_coin_symbol_var.set(f"{SYMBOL}{badge}")
            if hasattr(self, "auto_coin_last_scan_var"):
                if auto_coin_last_scan_ts > 0:
                    ts = dt.datetime.fromtimestamp(auto_coin_last_scan_ts)
                    self.auto_coin_last_scan_var.set(ts.strftime("%H:%M:%S"))
                else:
                    self.auto_coin_last_scan_var.set("-")
            if hasattr(self, "auto_coin_summary_var"):
                self.auto_coin_summary_var.set(auto_coin_last_scan_info if auto_coin_last_scan_info else "-")
            if hasattr(self, "auto_coin_table"):
                for row in self.auto_coin_table.get_children():
                    self.auto_coin_table.delete(row)
                for c in auto_coin_top_candidates[:5]:
                    sym = _market_to_symbol(str(c.get("market") or ""))
                    atr = float(c.get("atr_pct") or 0.0)
                    spread = float(c.get("spread_bps") or 0.0)
                    slip = float(c.get("slip_bps_buy_100") or 0.0)
                    value = float(c.get("value") or 0.0)
                    score = float(c.get("score") or 0.0)
                    self.auto_coin_table.insert(
                        "",
                        "end",
                        values=(sym, f"{atr:.2f}", f"{spread:.0f}", f"{slip:.0f}", _format_usdt_value(value), f"{score:.2f}"),
                    )
            if hasattr(self, "auto_coin_select_combo") and hasattr(self, "auto_coin_select_var"):
                symbols: List[str] = []
                for c in auto_coin_top_candidates[:10]:
                    sym = _market_to_symbol(str(c.get("market") or ""))
                    if sym and sym not in symbols:
                        symbols.append(sym)
                if SYMBOL and SYMBOL not in symbols:
                    symbols.insert(0, SYMBOL)
                self.auto_coin_select_combo.configure(values=symbols)
                current = self.auto_coin_select_var.get().strip()
                if current not in symbols:
                    if symbols:
                        self.auto_coin_select_var.set(symbols[0])
                    else:
                        self.auto_coin_select_var.set("")
                select_state = "disabled" if auto_coin_enabled else "readonly"
                try:
                    self.auto_coin_select_combo.configure(state=select_state)
                except Exception:
                    pass
                if hasattr(self, "auto_coin_switch_btn"):
                    btn_state = "disabled" if auto_coin_enabled or not symbols else "normal"
                    self.auto_coin_switch_btn.configure(state=btn_state)
            if hasattr(self, "symbol_entry"):
                st = "disabled" if auto_coin_enabled else "normal"
                self.symbol_entry.configure(state=st)
                if hasattr(self, "set_symbol_btn"):
                    self.set_symbol_btn.configure(state=st)
                if hasattr(self, "wallet_coin_combo"):
                    self.wallet_coin_combo.configure(state="disabled" if auto_coin_enabled else "readonly")
                if hasattr(self, "wallet_coin_btn"):
                    self.wallet_coin_btn.configure(state=st)
                try:
                    focused = self.root.focus_get() == self.symbol_entry
                except Exception:
                    focused = False
                if auto_coin_enabled or not focused:
                    self.symbol_var.set(SYMBOL)
        except Exception:
            pass

        # Time / timers
        try:
            now_ts = time.time()
            self.time_now_var.set(dt.datetime.now().strftime("%H:%M:%S"))

            if last_candle_ts:
                candle_age = max(0.0, now_ts - (float(last_candle_ts) / 1000.0))
                self.time_candle_var.set(f"{_fmt_duration(candle_age)} ({TIMEFRAME})")
            else:
                self.time_candle_var.set("-")

            if anchor_ts:
                age = max(0.0, now_ts - float(anchor_ts))
                self.time_anchor_var.set(f"{_fmt_duration(age)} / {_fmt_duration(ANCHOR_MAX_AGE_SEC)}")
            else:
                self.time_anchor_var.set("-")

            if use_dip_rebound and dip_wait_start_ts and dip_wait_start_ts > 0:
                elapsed = max(0.0, now_ts - float(dip_wait_start_ts))
                timeout = max(1.0, float(dip_timeout_sec))
                remaining = max(0.0, timeout - elapsed)
                self.time_dip_var.set(f"{_fmt_duration(elapsed)} / {_fmt_duration(timeout)} (rem {_fmt_duration(remaining)})")
            elif use_dip_rebound:
                self.time_dip_var.set("-")
            else:
                self.time_dip_var.set("disabled")

            cooldown_str = "-"
            if paused_until and now_ts < float(paused_until):
                remaining = max(0.0, float(paused_until) - now_ts)
                reason = str(paused_reason or "").strip()
                if reason:
                    cooldown_str = f"{_fmt_duration(remaining)} ({reason})"
                else:
                    cooldown_str = _fmt_duration(remaining)
            elif next_trade_time and now_ts < float(next_trade_time):
                remaining = max(0.0, float(next_trade_time) - now_ts)
                cooldown_str = f"{_fmt_duration(remaining)} (trade)"
            self.time_cooldown_var.set(cooldown_str)

            if auto_coin_enabled:
                if auto_coin_last_scan_ts > 0:
                    last_age = max(0.0, now_ts - auto_coin_last_scan_ts)
                    next_in = max(0.0, auto_coin_scan_interval_sec - last_age)
                    self.time_auto_scan_var.set(f"next {_fmt_duration(next_in)} (last {_fmt_duration(last_age)} ago)")
                else:
                    self.time_auto_scan_var.set(f"next {_fmt_duration(auto_coin_scan_interval_sec)}")

                if auto_coin_last_switch_ts > 0:
                    dwell_total = max(0.0, auto_coin_dwell_min * 60.0)
                    dwell_elapsed = max(0.0, now_ts - auto_coin_last_switch_ts)
                    dwell_remaining = max(0.0, dwell_total - dwell_elapsed)
                    if dwell_remaining > 0:
                        self.time_auto_dwell_var.set(_fmt_duration(dwell_remaining))
                    else:
                        self.time_auto_dwell_var.set("ready")
                else:
                    self.time_auto_dwell_var.set("ready")
            else:
                self.time_auto_scan_var.set("-")
                self.time_auto_dwell_var.set("-")

            if data_driven:
                refresh_sec = data_refresh_sec
                if profile in ADAPTIVE_PRESETS and profile != "Custom":
                    refresh_sec = float(ADAPTIVE_PRESETS[profile]["refresh"])
                if last_data_refresh_ts > 0:
                    age = max(0.0, now_ts - last_data_refresh_ts)
                    self.time_adaptive_var.set(f"{_fmt_duration(age)} ago / {_fmt_duration(refresh_sec)}")
                else:
                    self.time_adaptive_var.set(f"every {_fmt_duration(refresh_sec)}")
            else:
                self.time_adaptive_var.set("-")

            if isinstance(day_start_date, dt.date):
                now_utc = dt.datetime.now(dt.timezone.utc)
                next_day = dt.datetime.combine(day_start_date + dt.timedelta(days=1), dt.time(0, 0, tzinfo=dt.timezone.utc))
                remaining = max(0.0, (next_day - now_utc).total_seconds())
                self.time_day_reset_var.set(_fmt_duration(remaining))
            else:
                self.time_day_reset_var.set("-")
        except Exception:
            pass

        # Adaptive tab readout
        try:
            if hasattr(self, "adaptive_info_var"):
                if data_driven:
                    self.adaptive_info_var.set(
                        f"USING DATA ({profile}): BUY {buy_thr:.3f}% · SELL {sell_thr:.3f}%  |  tfs={','.join(tfs) if tfs else data_tf}  "
                        f"(trend@{data_tf}={trend})  vol≈{avg_range:.3f}%/min  avgH={avg_high:.6f} avgL={avg_low:.6f}"
                    )
                else:
                    self.adaptive_info_var.set(
                        f"USING MANUAL: BUY {buy_thr:.3f}% · SELL {sell_thr:.3f}%"
                    )
        except Exception:
            pass

        # Gray out live-parameter entries when their autopilot mode is auto
        try:
            ap_auto_buy = _ap_mode(self.state, "auto_buy_pct")
            if hasattr(self, "auto_buy_entry"):
                self.auto_buy_entry.configure(state="disabled" if ap_auto_buy == "auto" else "normal")
            ap_stop = _ap_mode(self.state, "stop_pct")
            if hasattr(self, "stop_loss_entry"):
                self.stop_loss_entry.configure(state="disabled" if ap_stop == "auto" else "normal")
        except Exception:
            pass


        # Manual buy display
        if SHOW_REAL_WALLET and not dry_run:
            available_usdt = 0.0
            for row in wallet_rows:
                if row.get("asset") == "USDT":
                    available_usdt = float(row.get("amount") or 0.0)
                    break
        else:
            available_usdt = sim_balance_usdt

        auto_pct = auto_buy_pct * 100.0
        auto_spend = available_usdt * auto_buy_pct
        self.auto_buy_display_var.set(f"{auto_pct:.1f}% (~{auto_spend:.2f} USDT)")

        pct = manual_buy_pct * 100.0
        spend = available_usdt * manual_buy_pct
        self.manual_buy_display_var.set(f"{pct:.1f}% (~{spend:.2f} USDT)")

        # Distance to triggers
        buy_dist_str = "-"
        sell_dist_str = "-"
        buy_y = None
        sell_y = None
        entry_y = None

        if last_trade_price and last_price:
            buy_y = last_trade_price * (1 - buy_thr / 100.0)
            sell_y = last_trade_price * (1 + sell_thr / 100.0)

            if buy_y and buy_y > 0:
                buy_diff = (last_price - buy_y) / buy_y * 100.0
                buy_dist_str = f"{buy_diff:+.2f}% vs BUY"
            if sell_y and sell_y > 0:
                sell_diff = (sell_y - last_price) / sell_y * 100.0
                sell_dist_str = f"{sell_diff:+.2f}% vs SELL"

        self.buy_dist_var.set(buy_dist_str)
        self.sell_dist_var.set(sell_dist_str)

        # Wallet coin dropdown
        coins = [row["asset"] for row in wallet_rows if row.get("asset") and row["asset"] != "USDT"]
        self.wallet_coin_combo["values"] = coins

        # Position panel
        if position:
            base = SYMBOL.split("/")[0]
            unreal = (last_price - position.entry_price) * position.qty if last_price else 0.0
            age = dt.datetime.now(dt.timezone.utc) - position.entry_time
            entry_y = position.entry_price
            lines = [
                f"IN POSITION · {SYMBOL}",
                f"Entry: {position.entry_price:.8f}",
                f"Size:  {position.qty:.6f} {base}",
                f"Unrealized PnL: {unreal:.2f} USDT",
                f"Realized (trade): {position.realized_pnl:.2f} USDT",
                f"Age: {str(age).split('.')[0]}",
            ]
            self.position_text_var.set("\n".join(lines))
        else:
            lines = [f"NO POSITION · {SYMBOL}"]
            if pending_buy_price:
                lines.append(f"Next BUY trigger: {pending_buy_price:.8f}")
            self.position_text_var.set("\n".join(lines))

        # Chart update
        if chart_times and chart_close and len(chart_times) == len(chart_close):
            self._update_chart(chart_times, chart_close, buy_y, sell_y, entry_y)

        # Events log
        with s.lock:
            events = list(s.event_log)

        self.events_text.config(state="normal")
        self.events_text.delete("1.0", tk.END)
        for line in events:
            self.events_text.insert(tk.END, line + "\n")
        self.events_text.config(state="disabled")
        self.events_text.see(tk.END)

        if not self.stop_event.is_set():
            self.root.after(900, self.update_ui)


# ============================================================
# ============================ MAIN ==========================
# ============================================================
def run_bot():
    # Load persisted settings (symbol, thresholds, adaptive stats, advanced config).
    global SYMBOL, DRY_RUN
    global STOP_LOSS_PCT, DAILY_MAX_LOSS_USDT, DAILY_MAX_PROFIT_USDT
    global ANCHOR_MAX_AGE_SEC, ANCHOR_DRIFT_REARM_PCT
    global POLL_INTERVAL_SECONDS, MIN_NOTIONAL_USDT
    global MAX_STOPS_PER_HOUR, CIRCUIT_STOP_COOLDOWN_SEC

    settings = load_settings() or {}

    try:
        if settings.get("symbol"):
            SYMBOL = str(settings["symbol"]).upper().strip()
            config.SYMBOL = SYMBOL
    except Exception:
        pass

    try:
        if "dry_run" in settings:
            DRY_RUN = bool(settings["dry_run"])
    except Exception:
        pass

    # Apply advanced config if present
    try:
        if "STOP_LOSS_PCT" in settings:
            STOP_LOSS_PCT = settings["STOP_LOSS_PCT"]
            state.stop_loss_effective_pct = STOP_LOSS_PCT
        if "DAILY_MAX_LOSS_USDT" in settings:
            DAILY_MAX_LOSS_USDT = float(settings["DAILY_MAX_LOSS_USDT"])
        if "DAILY_MAX_PROFIT_USDT" in settings:
            DAILY_MAX_PROFIT_USDT = float(settings["DAILY_MAX_PROFIT_USDT"])
        if "ANCHOR_MAX_AGE_SEC" in settings:
            ANCHOR_MAX_AGE_SEC = float(settings["ANCHOR_MAX_AGE_SEC"])
        if "ANCHOR_DRIFT_REARM_PCT" in settings:
            ANCHOR_DRIFT_REARM_PCT = float(settings["ANCHOR_DRIFT_REARM_PCT"])
        if "POLL_INTERVAL_SECONDS" in settings:
            POLL_INTERVAL_SECONDS = float(settings["POLL_INTERVAL_SECONDS"])
        if "MIN_NOTIONAL_USDT" in settings:
            MIN_NOTIONAL_USDT = float(settings["MIN_NOTIONAL_USDT"])
        if "MAX_STOPS_PER_HOUR" in settings:
            MAX_STOPS_PER_HOUR = int(settings["MAX_STOPS_PER_HOUR"])
        if "CIRCUIT_STOP_COOLDOWN_SEC" in settings:
            CIRCUIT_STOP_COOLDOWN_SEC = float(settings["CIRCUIT_STOP_COOLDOWN_SEC"])
    except Exception:
        pass

    exchange = init_exchange()
    state = BotState()
    # Apply persisted live params + adaptive settings
    try:
        with state.lock:
            if "buy_threshold_pct" in settings:
                state.buy_threshold_pct = float(settings["buy_threshold_pct"])
            if "sell_threshold_pct" in settings:
                state.sell_threshold_pct = float(settings["sell_threshold_pct"])
            if "auto_buy_pct" in settings:
                state.auto_buy_pct = float(settings["auto_buy_pct"])
            if "manual_buy_pct" in settings:
                state.manual_buy_pct = float(settings["manual_buy_pct"])
            if "manual_sell_pct" in settings:
                state.manual_sell_pct = float(settings["manual_sell_pct"])
            elif "MANUAL_SELL_PCT" in settings:
                # backward-compat from older settings files
                state.manual_sell_pct = float(settings["MANUAL_SELL_PCT"])
            if "use_tp1_trailing" in settings:
                state.use_tp1_trailing = bool(settings["use_tp1_trailing"])
            if "use_soft_stop" in settings:
                state.use_soft_stop = bool(settings["use_soft_stop"])
            if "soft_stop_confirms" in settings:
                try:
                    state.soft_stop_confirms = max(1, int(settings["soft_stop_confirms"]))
                except Exception:
                    state.soft_stop_confirms = config.SOFT_STOP_CONFIRMS_DEFAULT
            if "use_dip_rebound" in settings:
                state.use_dip_rebound = bool(settings["use_dip_rebound"])
            if "rebound_confirm_pct" in settings:
                try:
                    state.rebound_confirm_pct = max(0.0, float(settings["rebound_confirm_pct"]))
                except Exception:
                    state.rebound_confirm_pct = config.REBOUND_CONFIRM_PCT_DEFAULT
            if "dip_rebound_timeout_sec" in settings:
                try:
                    state.dip_rebound_timeout_sec = max(5.0, float(settings["dip_rebound_timeout_sec"]))
                except Exception:
                    state.dip_rebound_timeout_sec = config.DIP_REBOUND_TIMEOUT_SEC_DEFAULT
            if "use_edge_aware_thresholds" in settings:
                state.use_edge_aware_thresholds = bool(settings["use_edge_aware_thresholds"])
            if "use_ioc_slippage_cap" in settings:
                state.use_ioc_slippage_cap = bool(settings["use_ioc_slippage_cap"])
            if "max_slip_pct" in settings:
                try:
                    state.max_slip_pct = max(0.01, float(settings["max_slip_pct"]))
                except Exception:
                    state.max_slip_pct = config.MAX_SLIP_PCT_DEFAULT
            if "dip_confirm_closes" in settings:
                try:
                    state.dip_confirm_closes = max(1, int(settings["dip_confirm_closes"]))
                except Exception:
                    state.dip_confirm_closes = config.DIP_CONFIRM_CLOSES_DEFAULT
            if "auto_coin_enabled" in settings:
                state.auto_coin_enabled = bool(settings["auto_coin_enabled"])
            if "auto_coin_policy" in settings:
                policy = str(settings["auto_coin_policy"] or AUTO_COIN_POLICY_DEFAULT).lower()
                state.auto_coin_policy = policy if policy in ("flat", "force") else AUTO_COIN_POLICY_DEFAULT
            if "auto_coin_scan_interval_sec" in settings:
                try:
                    state.auto_coin_scan_interval_sec = max(10.0, float(settings["auto_coin_scan_interval_sec"]))
                except Exception:
                    state.auto_coin_scan_interval_sec = AUTO_COIN_SCAN_INTERVAL_SEC_DEFAULT
            if "auto_coin_dwell_min" in settings:
                try:
                    state.auto_coin_dwell_min = max(0.0, float(settings["auto_coin_dwell_min"]))
                except Exception:
                    state.auto_coin_dwell_min = AUTO_COIN_DWELL_MIN_DEFAULT
            if "auto_coin_hysteresis_pct" in settings:
                try:
                    state.auto_coin_hysteresis_pct = max(0.0, float(settings["auto_coin_hysteresis_pct"]))
                except Exception:
                    state.auto_coin_hysteresis_pct = AUTO_COIN_HYSTERESIS_PCT_DEFAULT
            if "auto_coin_candidates_n" in settings:
                try:
                    state.auto_coin_candidates_n = max(5, int(float(settings["auto_coin_candidates_n"])))
                except Exception:
                    state.auto_coin_candidates_n = AUTO_COIN_CANDIDATES_N_DEFAULT
            if "autopilot" in settings and isinstance(settings["autopilot"], dict):
                for k, v in settings["autopilot"].items():
                    try:
                        ap = _get_ap(state, k, float(v.get("manual", 0.0) or 0.0))
                        ap.manual = float(v.get("manual", ap.manual))
                        ap.mode = str(v.get("mode", ap.mode) or ap.mode)
                        ap.effective = ap.manual
                        state.autopilot[k] = ap
                    except Exception:
                        continue

            if "total_realized_pnl" in settings:
                state.total_realized_pnl = float(settings["total_realized_pnl"])
            if "total_runtime_sec" in settings:
                state.total_runtime_sec = float(settings["total_runtime_sec"])

            state.dry_run = bool(DRY_RUN)

            # adaptive
            if "data_driven" in settings:
                state.data_driven = bool(settings["data_driven"])
            if "data_timeframe" in settings:
                state.data_timeframe = str(settings["data_timeframe"])
            if "data_lookback" in settings:
                state.data_lookback = int(settings["data_lookback"])
            if "data_refresh_sec" in settings:
                state.data_refresh_sec = int(settings["data_refresh_sec"])

            # adaptive profile (multi-horizon)
            if "adaptive_profile" in settings:
                state.adaptive_profile = str(settings["adaptive_profile"])
            if "adaptive_timeframes" in settings and isinstance(settings["adaptive_timeframes"], list):
                state.adaptive_timeframes = [str(x) for x in settings["adaptive_timeframes"] if str(x)]
            if "adaptive_weights" in settings and isinstance(settings["adaptive_weights"], list):
                try:
                    state.adaptive_weights = [float(x) for x in settings["adaptive_weights"]]
                except Exception:
                    state.adaptive_weights = []
            if "adaptive_k_buy" in settings:
                state.adaptive_k_buy = float(settings["adaptive_k_buy"])
            if "adaptive_k_sell" in settings:
                state.adaptive_k_sell = float(settings["adaptive_k_sell"])
            if "adaptive_alpha" in settings:
                state.adaptive_alpha = float(settings["adaptive_alpha"])
            state.blended_atr_per_min_pct = float(settings.get("blended_atr_per_min_pct", 0.0) or 0.0)

            state.avg_high = float(settings.get("avg_high", 0.0) or 0.0)
            state.avg_low = float(settings.get("avg_low", 0.0) or 0.0)
            state.avg_range_pct = float(settings.get("avg_range_pct", 0.0) or 0.0)
            state.data_trend = str(settings.get("data_trend", "?") or "?")

            state.effective_buy_threshold_pct = float(settings.get("effective_buy_threshold_pct", state.buy_threshold_pct))
            state.effective_sell_threshold_pct = float(settings.get("effective_sell_threshold_pct", state.sell_threshold_pct))

            # Seed autopilot defaults
            _get_ap(state, "trail_pct", float(getattr(state, "effective_trail_pct", config.TRAIL_STOP_PCT)))
            _get_ap(state, "tp1_frac", float(getattr(state, "effective_tp1_frac", config.TP1_SELL_FRACTION)))
            _get_ap(state, "soft_stop_confirms", float(getattr(state, "soft_stop_confirms", config.SOFT_STOP_CONFIRMS_DEFAULT)))
            _get_ap(state, "hard_stop_mult", float(getattr(state, "effective_hard_stop_mult", config.HARD_STOP_MULT_DEFAULT)))
            _get_ap(state, "max_slip_pct", float(getattr(state, "max_slip_pct", config.MAX_SLIP_PCT_DEFAULT)))
            _get_ap(state, "stop_pct", float(getattr(state, "stop_loss_effective_pct", STOP_LOSS_PCT if STOP_LOSS_PCT is not None else 0.0)))
            _get_ap(state, "auto_buy_pct", float(getattr(state, "auto_buy_pct_effective", state.auto_buy_pct * 100.0)))
    except Exception:
        pass

    cmd_queue: "queue.Queue[str]" = queue.Queue()
    stop_event = threading.Event()

    t = threading.Thread(target=trading_loop, args=(exchange, state, cmd_queue, stop_event), daemon=True)
    t.start()

    if tb is not None:
        root = tb.Window(themename="darkly")
    else:
        root = tk.Tk()

    BotGUI(root, state, cmd_queue, stop_event)
    log_event(state, "GUI started. Fields + chart update live. Use DRY_RUN first if you're not sure.")
    root.mainloop()


if __name__ == "__main__":
    run_bot()
