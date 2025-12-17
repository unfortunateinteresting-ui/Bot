import datetime as dt
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from bot_app import config


@dataclass
class Position:
    symbol: str
    entry_price: float
    qty: float
    initial_qty: float
    entry_time: dt.datetime
    entry_index: int
    realized_pnl: float = 0.0
    entry_fee: float = 0.0
    last_exit_reason: Optional[str] = None
    tp1_done: bool = False
    peak_price: float = 0.0


@dataclass
class OpenOrder:
    id: Optional[str]
    side: str       # "buy" or "sell"
    type: str       # "limit"
    price: float
    qty: float
    status: str     # "open", "filled", "canceled"
    created_time: dt.datetime
    simulated: bool = False


@dataclass
class BotState:
    # Thread safety: trading loop writes, UI reads.
    lock: threading.Lock = field(default_factory=threading.Lock)
    dry_run: bool = config.DRY_RUN  # True = paper trading; False = live

    position: Optional[Position] = None
    sim_balance_usdt: float = config.SIMULATED_BALANCE_USDT
    sim_base_qty: float = 0.0
    use_tp1_trailing: bool = config.USE_TP1_TRAILING_DEFAULT
    use_soft_stop: bool = config.USE_SOFT_STOP_DEFAULT
    soft_stop_confirms: int = config.SOFT_STOP_CONFIRMS_DEFAULT
    soft_stop_counter: int = 0
    use_dip_rebound: bool = config.USE_DIP_REBOUND_DEFAULT
    rebound_confirm_pct: float = config.REBOUND_CONFIRM_PCT_DEFAULT
    dip_rebound_timeout_sec: float = config.DIP_REBOUND_TIMEOUT_SEC_DEFAULT
    dip_wait_start_ts: float = 0.0
    dip_low: float = 0.0
    dip_higher_close_count: int = 0
    use_edge_aware_thresholds: bool = config.USE_EDGE_AWARE_THRESHOLDS_DEFAULT
    use_ioc_slippage_cap: bool = config.USE_IOC_SLIPPAGE_CAP_DEFAULT
    max_slip_pct: float = config.MAX_SLIP_PCT_DEFAULT

    daily_realized_pnl: float = 0.0
    day_start_date: dt.date = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).date())
    stop_loss_timestamps: deque = field(default_factory=deque)
    trading_paused_until: Optional[float] = None
    paused_reason: Optional[str] = None

    next_trade_time: float = 0.0
    last_candle_ts: Optional[int] = None

    status: str = "INIT"
    last_error: Optional[str] = None

    last_trade_price: Optional[float] = None
    last_trade_side: Optional[str] = None  # "BUY" or "SELL"
    anchor_timestamp: Optional[float] = None

    pending_buy_price: Optional[float] = None
    pending_sell_price: Optional[float] = None

    event_log: deque = field(default_factory=lambda: deque(maxlen=config.MAX_EVENT_LOG))

    order_mode: str = "market"  # "market" or "limit"
    open_order: Optional[OpenOrder] = None

    # Live-tunable params
    buy_threshold_pct: float = config.BUY_THRESHOLD_PCT_DEFAULT
    sell_threshold_pct: float = config.SELL_THRESHOLD_PCT_DEFAULT
    manual_buy_pct: float = config.MANUAL_BUY_PCT_DEFAULT  # 0..1 fraction of USDT
    auto_buy_pct: float = config.AUTO_BUY_PCT_DEFAULT      # 0..1 fraction of USDT

    manual_sell_pct: float = config.MANUAL_SELL_PCT_DEFAULT  # 0..1 fraction of position

    # Lifetime stats (persisted)
    total_realized_pnl: float = 0.0
    total_runtime_sec: float = 0.0
    session_start_ts: float = field(default_factory=time.time)

    # Last slippage stats (market orders; displayed in UI)
    last_slippage_bps: float = 0.0
    last_slippage_side: str = ""
    last_slippage_best: float = 0.0
    last_slippage_fill: float = 0.0
    last_slippage_levels: int = 0
    last_slippage_ts: float = 0.0

    # --- Adaptive (data-driven) thresholds ---
    data_driven: bool = config.DATA_DRIVEN_DEFAULT
    data_timeframe: str = config.DATA_TIMEFRAME_DEFAULT
    data_lookback: int = config.DATA_LOOKBACK_CANDLES_DEFAULT
    data_refresh_sec: int = config.DATA_REFRESH_SEC_DEFAULT

    avg_high: float = 0.0
    avg_low: float = 0.0
    avg_range_pct: float = 0.0
    data_trend: str = "?"
    effective_buy_threshold_pct: float = config.BUY_THRESHOLD_PCT_DEFAULT
    effective_sell_threshold_pct: float = config.SELL_THRESHOLD_PCT_DEFAULT
    _last_data_refresh_ts: float = 0.0

    # --- Adaptive profiles (multi-horizon) ---
    adaptive_profile: str = config.ADAPTIVE_PROFILE_DEFAULT
    adaptive_timeframes: List[str] = field(default_factory=lambda: list(config.ADAPTIVE_PRESETS[config.ADAPTIVE_PROFILE_DEFAULT]["timeframes"]))
    adaptive_weights: List[float] = field(default_factory=list)  # used in Custom (optional)
    adaptive_k_buy: float = float(config.ADAPTIVE_PRESETS[config.ADAPTIVE_PROFILE_DEFAULT]["k_buy"])
    adaptive_k_sell: float = float(config.ADAPTIVE_PRESETS[config.ADAPTIVE_PROFILE_DEFAULT]["k_sell"])
    adaptive_alpha: float = float(config.ADAPTIVE_PRESETS[config.ADAPTIVE_PROFILE_DEFAULT]["alpha"])
    blended_atr_per_min_pct: float = 0.0

    # Snapshots for UI
    wallet_rows: List[Dict[str, Any]] = field(default_factory=list)
    wallet_equity: float = 0.0
    last_price: float = 0.0

    # Chart data (copied from latest OHLCV)
    chart_times: List[dt.datetime] = field(default_factory=list)
    chart_close: List[float] = field(default_factory=list)
    chart_high: List[float] = field(default_factory=list)
    chart_low: List[float] = field(default_factory=list)


def log_event(state: BotState, msg: str):
    ts = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with state.lock:
        state.event_log.append(line)
    print(line, flush=True)
