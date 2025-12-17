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
from typing import Optional, List, Tuple, Dict, Any
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

from bot_app.config import *
from bot_app.exchange_utils import (
    fetch_best_book_price,
    fetch_ohlcv_df,
    init_exchange,
    simulate_dry_run_market_fill,
)
from bot_app.settings_io import load_settings, save_settings
from bot_app.state import BotState, OpenOrder, Position, log_event



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


def place_market_buy(exchange: ccxt.Exchange, state: BotState, qty: float, price_estimate: float,
                     now: dt.datetime, entry_index: int) -> Optional[Position]:
    if qty <= 0:
        return None
    symbol = SYMBOL

    if DRY_RUN:
        if SIMULATE_SLIPPAGE_DRY_RUN:
            filled_qty, fill_price, best, worst, used = simulate_dry_run_market_fill(
                exchange, symbol, "buy", qty, price_estimate
            )
            # 'best' is best ask encountered; slippage is relative to that
            slip_bps = 0.0
            if best and best > 0:
                slip_bps = (fill_price / best - 1.0) * 10000.0
            # store for UI (and persistence via save_settings)
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
            log_event(
                state,
                f"MARKET BUY (DRY_RUN) {filled_qty:.6f} {symbol.split('/')[0]} @ {fill_price:.8f} "
                f"(book best={best:.8f}, worst={worst:.8f}, slip={slip_bps:.1f} bps, lvls={used})"
            )
        else:
            fill_price = price_estimate
            filled_qty = qty
            try:
                with state.lock:
                    state.last_slippage_bps = 0.0
                    state.last_slippage_side = "BUY"
                    state.last_slippage_best = float(price_estimate or 0.0)
                    state.last_slippage_fill = float(fill_price)
                    state.last_slippage_levels = 0
                    state.last_slippage_ts = time.time()
            except Exception:
                pass
            log_event(state, f"MARKET BUY (DRY_RUN) {filled_qty:.6f} {symbol.split('/')[0]} @ {fill_price:.8f}")
    else:
        best_pre = fetch_best_book_price(exchange, symbol, "buy", price_estimate)
        try:
            order = exchange.create_order(symbol, "market", "buy", qty, price_estimate)
            filled_qty = float(order.get("filled") or order.get("amount") or qty)
            fill_price = float(order.get("average") or order.get("price") or price_estimate)
            slip_bps = 0.0
            if best_pre and best_pre > 0:
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
            log_event(state, f"MARKET BUY filled {filled_qty:.6f} {symbol.split('/')[0]} @ {fill_price:.8f} (best={best_pre:.8f}, slip={slip_bps:.1f} bps)")
        except Exception as e:
            log_event(state, f"[ERROR] Market BUY error: {e}")
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
        entry_time=now,
        entry_index=entry_index,
        entry_fee=entry_fee,
    )


def place_market_sell(exchange: ccxt.Exchange, state: BotState, position: Position, qty_exit: float,
                      price_estimate: float) -> Tuple[float, float]:
    if qty_exit <= 0:
        return 0.0, price_estimate

    symbol = position.symbol

    if DRY_RUN:
        if SIMULATE_SLIPPAGE_DRY_RUN:
            filled_qty, fill_price, best, worst, used = simulate_dry_run_market_fill(
                exchange, symbol, "sell", qty_exit, price_estimate
            )
            slip_bps = 0.0
            if best and best > 0:
                # for sells, worse fill is LOWER than best bid => positive bps
                slip_bps = (1.0 - fill_price / best) * 10000.0
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
            log_event(
                state,
                f"MARKET SELL (DRY_RUN) {filled_qty:.6f} {symbol.split('/')[0]} @ {fill_price:.8f} "
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
            log_event(state, f"MARKET SELL (DRY_RUN) {filled_qty:.6f} {symbol.split('/')[0]} @ {fill_price:.8f}")
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
    if qty_to_sell <= 0:
        log_event(state, "[WARN] Market SELL skipped: quantity <= 0 after safety margin.")
        return 0.0, price_estimate

    best_pre = fetch_best_book_price(exchange, symbol, "sell", price_estimate)
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
        log_event(state, f"MARKET SELL filled {filled_qty:.6f} {symbol.split('/')[0]} @ {fill_price:.8f} (best={best_pre:.8f}, slip={slip_bps:.1f} bps)")
        return filled_qty, fill_price
    except Exception as e:
        log_event(state, f"[ERROR] Market SELL error: {e}")
        return 0.0, price_estimate


def place_limit_buy(exchange: ccxt.Exchange, state: BotState, price: float, usdt_to_spend: float,
                    now: dt.datetime, entry_index: int) -> Optional[OpenOrder]:
    if usdt_to_spend <= 0 or price <= 0:
        return None
    qty = usdt_to_spend / price
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

    qty_ratio = qty_exit / position.qty
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
                    entry_time=dt.datetime.now(dt.timezone.utc),
                    entry_index=len(df) - 1,
                    entry_fee=fee,
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
                entry_time=dt.datetime.now(dt.timezone.utc),
                entry_index=len(df) - 1,
                entry_fee=filled_qty * avg_price * TAKER_FEE,
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
        log_event(state, f"Re-anchoring BUY reference {old_anchor:.8f} -> {last_price:.8f} (age={0 if age is None else age:.0f}s).")

    buy_trigger_price = state.last_trade_price * (1 - buy_thr / 100.0)

    if state.pending_buy_price is None or abs(buy_trigger_price - state.pending_buy_price) > 1e-12:
        state.pending_buy_price = buy_trigger_price
        # keep log quieter; comment back in if you want it noisy:
        # log_event(state, f"New BUY trigger set at {buy_trigger_price:.8f} (waiting for dip)")

    if last_price > buy_trigger_price:
        state.status = "WAIT_DIP"
        return

    # size
    if DRY_RUN:
        available_usdt = state.sim_balance_usdt
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
        qty = spend_usdt / last_price
        now = dt.datetime.now(dt.timezone.utc)
        pos = place_market_buy(exchange, state, qty, last_price, now, last_idx)
        if not pos:
            return

        state.position = pos
        state.status = "IN_POSITION"
        state.next_trade_time = time.time() + POLL_INTERVAL_SECONDS

        state.last_trade_price = pos.entry_price
        state.last_trade_side = "BUY"
        state.anchor_timestamp = time.time()
        state.pending_buy_price = None
        log_event(state, f"Entered position (market) @ {pos.entry_price:.8f}, qty={pos.qty:.6f}")
    else:
        if state.open_order and state.open_order.status == "open" and state.open_order.side == "buy":
            return

        now = dt.datetime.now(dt.timezone.utc)
        oo = place_limit_buy(exchange, state, last_price, spend_usdt, now, last_idx)
        if oo:
            state.open_order = oo
            state.next_trade_time = time.time() + POLL_INTERVAL_SECONDS


def manage_position(exchange: ccxt.Exchange, state: BotState, df: pd.DataFrame):
    position = state.position
    if position is None:
        return

    last_row = df.iloc[-1]
    last_price = float(last_row["close"])
    sell_thr = (state.effective_sell_threshold_pct if state.data_driven else state.sell_threshold_pct)

    sell_trigger = state.last_trade_price * (1 + sell_thr / 100.0)

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

    # market exits
    full_exit = False
    reason = None

    if last_price >= sell_trigger:
        full_exit = True
        reason = "take_profit_threshold"

    if (not full_exit) and (STOP_LOSS_PCT is not None):
        stop_price = position.entry_price * (1 - STOP_LOSS_PCT / 100.0)
        if last_price <= stop_price:
            full_exit = True
            reason = "stop_loss"

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
        pos = Position(symbol=SYMBOL, entry_price=last_price, qty=qty_wallet, entry_time=now, entry_index=-1, entry_fee=0.0)
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
    state.last_trade_price = None
    state.last_trade_side = None
    state.anchor_timestamp = None
    state.pending_buy_price = None
    state.pending_sell_price = None
    state.last_candle_ts = None
    state.status = "INIT"
    state.sim_base_qty = 0.0

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
        else:
            try:
                balance = exchange.fetch_balance()
                available_usdt = float(balance.get("free", {}).get("USDT") or 0.0)
            except Exception as e:
                log_event(state, f"[ERROR] Manual BUY: balance error: {e}")
                return

        spend = available_usdt * max(0.0, min(state.manual_buy_pct, 1.0))
        if spend < MIN_NOTIONAL_USDT:
            log_event(state, f"Manual BUY skipped: not enough USDT (have {available_usdt:.2f}).")
            return

        qty = spend / last_price
        now = dt.datetime.now(dt.timezone.utc)
        pos = place_market_buy(exchange, state, qty, last_price, now, -1)
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

            new_candle = state.last_candle_ts is None or last_ts > state.last_candle_ts
            if new_candle:
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
        self.tabs.add(self.tab_trading, text="Trading")
        self.tabs.add(self.tab_adaptive, text="Adaptive")

        self._build_status_card(self.tab_trading)
        self._build_api_card(self.tab_trading)
        self._build_params_card(self.tab_trading)
        self._build_buttons_card(self.tab_trading)

        self._build_adaptive_tab(self.tab_adaptive)


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

        self.auto_buy_display_var = tk.StringVar(value="")
        ttk.Label(card, textvariable=self.auto_buy_display_var).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 8)
        )

        ttk.Label(card, text="Manual BUY size (% of USDT)").grid(row=4, column=0, sticky="w", padx=8, pady=(6, 2))
        e_manual = ttk.Entry(card, textvariable=self.manual_buy_pct_var, width=10)
        e_manual.grid(row=4, column=1, sticky="e", padx=8, pady=(6, 2))
        e_manual.bind("<Return>", self.on_manual_buy_entry)
        e_manual.bind("<FocusOut>", self.on_manual_buy_entry)

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
            ttk.Entry(card, textvariable=var, width=12).grid(row=r, column=1, sticky="e", padx=8, pady=3)

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
        sym_entry = ttk.Entry(symbol_frame, textvariable=self.symbol_var)
        sym_entry.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Button(symbol_frame, text="Set Symbol", command=self.on_set_symbol).pack(
            fill="x", padx=8, pady=(0, 6)
        )

        # --- Trade coin from wallet box ---
        wallet_sel_frame = ttk.LabelFrame(info, text="Trade coin from wallet")
        wallet_sel_frame.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        ttk.Label(wallet_sel_frame, text="Coin").pack(anchor="w", padx=8, pady=(6, 2))
        self.wallet_coin_combo = ttk.Combobox(wallet_sel_frame, values=[], state="readonly", width=10)
        self.wallet_coin_combo.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Button(wallet_sel_frame, text="Use Wallet Coin", command=self.on_use_wallet_coin).pack(
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
            buy_thr = eff_buy if data_driven else s.buy_threshold_pct
            sell_thr = eff_sell if data_driven else s.sell_threshold_pct
            pending_buy_price = s.pending_buy_price
            manual_buy_pct = s.manual_buy_pct
            auto_buy_pct = s.auto_buy_pct
            sim_balance_usdt = s.sim_balance_usdt
            chart_times = list(s.chart_times)
            chart_close = list(s.chart_close)

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