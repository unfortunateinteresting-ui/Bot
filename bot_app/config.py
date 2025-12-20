import os

API_KEY = os.getenv("API_KEY", "")
SECRET_KEY = os.getenv("SECRET_KEY", "")

EXCHANGE_NAME = "coinex"
SYMBOL = "BOT/USDT"  # default; can be changed from GUI or wallet selector

TIMEFRAME = "1m"
HISTORY_LIMIT = 300  # candles shown on chart + used for logic

DRY_RUN = False  # LIVE if you use real keys with trading perms
SIMULATED_BALANCE_USDT = 100.0
TAKER_FEE = 0.0015  # 0.15% per side (adjust to your exchange)

# Feature toggles
USE_TP1_TRAILING_DEFAULT = True   # enable partial TP + trailing exit logic
USE_SOFT_STOP_DEFAULT = True      # confirm stops with multiple closes unless hard-stop trips
USE_DIP_REBOUND_DEFAULT = True    # require rebound confirmation before buying the dip
USE_EDGE_AWARE_THRESHOLDS_DEFAULT = True  # clamp thresholds to cover fees + spread + slippage
USE_IOC_SLIPPAGE_CAP_DEFAULT = True       # switch market orders to IOC limits with slippage cap

# Soft/Hard stop settings
SOFT_STOP_CONFIRMS_DEFAULT = 1    # closes below stop before exiting (soft stop)
HARD_STOP_MULT_DEFAULT = 1.5      # hard-stop triggers at this multiple of stop distance (e.g., 1.5x stop depth)

# Dip-to-bounce entry settings
REBOUND_CONFIRM_PCT_DEFAULT = 0.08      # % rebound off dip low to confirm entry (light)
DIP_REBOUND_TIMEOUT_SEC_DEFAULT = 90.0  # max seconds to wait for rebound confirmation
DIP_CONFIRM_CLOSES_DEFAULT = 1          # number of higher closes to confirm bounce
DIP_NO_NEW_LOW_BUFFER_PCT = 0.02        # 0.02% buffer for "no new lows" timeout fallback

# Edge-aware thresholds
EDGE_BUFFER_PCT_DEFAULT = 0.05         # extra cushion on top of costs (in %)
EDGE_SPREAD_FALLBACK_PCT = 0.02        # fallback spread % if ticker spread unavailable
EDGE_SLIPPAGE_FALLBACK_PCT = 0.05      # fallback slippage % if last slippage unknown

# IOC slippage cap for execution (percent)
MAX_SLIP_PCT_DEFAULT = 0.60            # 0.60% default cap over best bid/ask for IOC limits

# Trailing defaults tuned for scalping
TRAIL_STOP_PCT = 0.30                  # base trail giveback %

# ---- DRY_RUN slippage simulator (order book based) ----
# If enabled, DRY_RUN market orders will fill using the live order book depth,
# which makes PnL closer to reality (spread + depth slippage).
SIMULATE_SLIPPAGE_DRY_RUN = True
SLIPPAGE_ORDERBOOK_LEVELS = 50          # how many bid/ask levels to read
SLIPPAGE_FALLBACK_BPS = 25              # used if the book has insufficient depth (25 bps = 0.25%)

BUY_THRESHOLD_PCT_DEFAULT = 1.0   # % dip below anchor to trigger BUY
SELL_THRESHOLD_PCT_DEFAULT = 1.0  # % pump above anchor to trigger SELL
STOP_LOSS_PCT = None              # e.g. 5.0 for 5% SL; None = disabled

# Partial take-profit + trailing stop
TP1_SELL_FRACTION = 0.50          # fraction of position to sell on first TP (0.30 - 0.60 recommended)
TRAIL_STOP_PCT = 0.80             # % giveback from peak (post-TP1) that triggers exit of remainder
TRAIL_ATR_MULT = 1.5              # optional: widen trail using blended ATR% per minute * this multiplier

ANCHOR_MAX_AGE_SEC = 3600         # after 1h, re-anchor to latest price
ANCHOR_DRIFT_REARM_PCT = 2.0      # or if price drifts >2% above anchor

DAILY_MAX_LOSS_USDT = -100.0
DAILY_MAX_PROFIT_USDT = 200.0
MAX_STOPS_PER_HOUR = 3
CIRCUIT_STOP_COOLDOWN_SEC = 1800

POLL_INTERVAL_SECONDS = 3
MIN_NOTIONAL_USDT = 5.0

SHOW_REAL_WALLET = True
MAX_EVENT_LOG = 80

MANUAL_BUY_PCT_DEFAULT = 1.0  # % of USDT used by the Manual BUY button (1.0 = 100%)
AUTO_BUY_PCT_DEFAULT = 1.0    # % of USDT used by automatic entries (1.0 = 100%)
MANUAL_SELL_PCT_DEFAULT = 1.0  # % of position sold by the Manual SELL button (1.0 = 100%)


# ---- Persistence ----
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot1_settings.json")

# ---- Data-driven / adaptive thresholds ----
DATA_DRIVEN_DEFAULT = False
DATA_TIMEFRAME_DEFAULT = "5m"
DATA_LOOKBACK_CANDLES_DEFAULT = 60
DATA_REFRESH_SEC_DEFAULT = 30

DATA_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]

# Clamp computed thresholds (percent)
DATA_BUY_MIN_PCT, DATA_BUY_MAX_PCT = 0.20, 3.00
DATA_SELL_MIN_PCT, DATA_SELL_MAX_PCT = 0.20, 3.00

# Multipliers applied to average candle range % based on trend
DATA_BUY_MULT_UP,   DATA_SELL_MULT_UP   = 0.70, 1.10
DATA_BUY_MULT_DOWN, DATA_SELL_MULT_DOWN = 1.20, 0.80
DATA_BUY_MULT_SIDE, DATA_SELL_MULT_SIDE = 1.00, 1.00


# ---- Adaptive profiles (multi-horizon ATR% blending) ----
# These presets control which OHLCV timeframes are used to estimate volatility/trend,
# and how aggressively the bot adjusts the effective BUY/SELL thresholds.
ADAPTIVE_PROFILE_DEFAULT = "Scalping"
ADAPTIVE_PROFILES = ["Scalping", "Short", "Medium", "Long", "Custom"]

# For each profile:
# - timeframes: list of OHLCV timeframes to blend (multi-horizon)
# - weights:    blending weights (must match timeframes length)
# - lookback:   candles fetched per timeframe
# - refresh:    seconds between refreshes
# - alpha:      smoothing factor for thresholds (EMA); higher = more reactive
# - k_buy/k_sell: multipliers on blended volatility (ATR% per minute)
ADAPTIVE_PRESETS = {
    "Scalping": {"timeframes": ["1m", "3m", "5m"], "weights": [0.60, 0.30, 0.10], "lookback": 140, "refresh": 10,  "alpha": 0.35, "k_buy": 2.2, "k_sell": 2.6},
    "Short":    {"timeframes": ["3m", "5m", "15m"],"weights": [0.40, 0.40, 0.20], "lookback": 120, "refresh": 20,  "alpha": 0.25, "k_buy": 2.8, "k_sell": 3.2},
    "Medium":   {"timeframes": ["15m", "1h"],      "weights": [0.40, 0.60],       "lookback": 96,  "refresh": 60,  "alpha": 0.15, "k_buy": 3.8, "k_sell": 4.5},
    "Long":     {"timeframes": ["1h", "4h"],       "weights": [0.35, 0.65],       "lookback": 90,  "refresh": 180, "alpha": 0.10, "k_buy": 5.2, "k_sell": 6.0},
    "Custom":   {"timeframes": [DATA_TIMEFRAME_DEFAULT], "weights": [],           "lookback": DATA_LOOKBACK_CANDLES_DEFAULT, "refresh": DATA_REFRESH_SEC_DEFAULT, "alpha": 0.20, "k_buy": 3.0, "k_sell": 3.5},
}
