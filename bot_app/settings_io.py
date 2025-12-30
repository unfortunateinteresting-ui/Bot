import json
import os
import time
from typing import Any

from bot_app import config
from bot_app.state import BotState


def _atomic_write_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_settings() -> dict:
    try:
        if not os.path.exists(config.CONFIG_PATH):
            return {}
        with open(config.CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def save_settings(state: BotState) -> None:
    """Persist both live parameters and last computed adaptive stats."""
    try:
        # Copy under lock (so UI + trading loop don't collide mid-write)
        with state.lock:
            payload: dict[str, Any] = {
                "symbol": config.SYMBOL,
                "dry_run": bool(getattr(state, "dry_run", config.DRY_RUN)),

                # manual thresholds + sizing
                "buy_threshold_pct": float(state.buy_threshold_pct),
                "sell_threshold_pct": float(state.sell_threshold_pct),
                "auto_buy_pct": float(state.auto_buy_pct),
                "manual_buy_pct": float(state.manual_buy_pct),
                "manual_sell_pct": float(getattr(state, "manual_sell_pct", config.MANUAL_SELL_PCT_DEFAULT)),
                "use_tp1_trailing": bool(getattr(state, "use_tp1_trailing", config.USE_TP1_TRAILING_DEFAULT)),
                "use_soft_stop": bool(getattr(state, "use_soft_stop", config.USE_SOFT_STOP_DEFAULT)),
                "soft_stop_confirms": int(getattr(state, "soft_stop_confirms", config.SOFT_STOP_CONFIRMS_DEFAULT)),
                "use_dip_rebound": bool(getattr(state, "use_dip_rebound", config.USE_DIP_REBOUND_DEFAULT)),
                "rebound_confirm_pct": float(getattr(state, "rebound_confirm_pct", config.REBOUND_CONFIRM_PCT_DEFAULT)),
                "dip_rebound_timeout_sec": float(getattr(state, "dip_rebound_timeout_sec", config.DIP_REBOUND_TIMEOUT_SEC_DEFAULT)),
                "use_edge_aware_thresholds": bool(getattr(state, "use_edge_aware_thresholds", config.USE_EDGE_AWARE_THRESHOLDS_DEFAULT)),
                "use_ioc_slippage_cap": bool(getattr(state, "use_ioc_slippage_cap", config.USE_IOC_SLIPPAGE_CAP_DEFAULT)),
                "max_slip_pct": float(getattr(state, "max_slip_pct", config.MAX_SLIP_PCT_DEFAULT)),
                "dip_confirm_closes": int(getattr(state, "dip_confirm_closes", config.DIP_CONFIRM_CLOSES_DEFAULT)),
                "autopilot": {k: {"manual": float(v.manual), "mode": str(v.mode)} for k, v in (getattr(state, "autopilot", {}) or {}).items()},

                # auto coin (trending scanner)
                "auto_coin_enabled": bool(getattr(state, "auto_coin_enabled", config.AUTO_COIN_ENABLED_DEFAULT)),
                "auto_coin_policy": str(getattr(state, "auto_coin_policy", config.AUTO_COIN_POLICY_DEFAULT)),
                "auto_coin_scan_interval_sec": float(getattr(state, "auto_coin_scan_interval_sec", config.AUTO_COIN_SCAN_INTERVAL_SEC_DEFAULT)),
                "auto_coin_dwell_min": float(getattr(state, "auto_coin_dwell_min", config.AUTO_COIN_DWELL_MIN_DEFAULT)),
                "auto_coin_hysteresis_pct": float(getattr(state, "auto_coin_hysteresis_pct", config.AUTO_COIN_HYSTERESIS_PCT_DEFAULT)),
                "auto_coin_candidates_n": int(getattr(state, "auto_coin_candidates_n", config.AUTO_COIN_CANDIDATES_N_DEFAULT)),

                # Freqtrade-inspired logic
                "ft_enabled": bool(getattr(state, "ft_enabled", config.FT_LOGIC_ENABLED_DEFAULT)),
                "ft_signal_only": bool(getattr(state, "ft_signal_only", config.FT_SIGNAL_ONLY_DEFAULT)),
                "ft_entry_ema_filter": bool(getattr(state, "ft_entry_ema_filter", config.FT_ENTRY_EMA_FILTER_DEFAULT)),
                "ft_entry_rsi_filter": bool(getattr(state, "ft_entry_rsi_filter", config.FT_ENTRY_RSI_FILTER_DEFAULT)),
                "ft_entry_vol_filter": bool(getattr(state, "ft_entry_vol_filter", config.FT_ENTRY_VOL_FILTER_DEFAULT)),
                "ft_exit_signal": bool(getattr(state, "ft_exit_signal", config.FT_EXIT_SIGNAL_ENABLED_DEFAULT)),
                "ft_exit_rsi_signal": bool(getattr(state, "ft_exit_rsi_signal", config.FT_EXIT_RSI_SIGNAL_DEFAULT)),
                "ft_roi_enabled": bool(getattr(state, "ft_roi_enabled", config.FT_ROI_ENABLED_DEFAULT)),
                "ft_ema_fast": int(getattr(state, "ft_ema_fast", config.FT_EMA_FAST_DEFAULT)),
                "ft_ema_slow": int(getattr(state, "ft_ema_slow", config.FT_EMA_SLOW_DEFAULT)),
                "ft_rsi_period": int(getattr(state, "ft_rsi_period", config.FT_RSI_PERIOD_DEFAULT)),
                "ft_rsi_entry_max": float(getattr(state, "ft_rsi_entry_max", config.FT_RSI_ENTRY_MAX_DEFAULT)),
                "ft_rsi_exit_min": float(getattr(state, "ft_rsi_exit_min", config.FT_RSI_EXIT_MIN_DEFAULT)),
                "ft_vol_period": int(getattr(state, "ft_vol_period", config.FT_VOL_PERIOD_DEFAULT)),
                "ft_vol_mult": float(getattr(state, "ft_vol_mult", config.FT_VOL_MULT_DEFAULT)),
                "ft_exit_min_profit_pct": float(getattr(state, "ft_exit_min_profit_pct", config.FT_EXIT_MIN_PROFIT_PCT_DEFAULT)),
                "ft_roi_table": str(getattr(state, "ft_roi_table", config.FT_ROI_TABLE_DEFAULT)),

                # lifetime stats
                "total_realized_pnl": float(getattr(state, "total_realized_pnl", 0.0)),
                "total_runtime_sec": float(getattr(state, "total_runtime_sec", 0.0)) + max(0.0, (time.time() - float(getattr(state, "session_start_ts", time.time())))),

                # adaptive/data-driven settings + last stats
                "data_driven": bool(getattr(state, "data_driven", False)),
                "data_timeframe": str(getattr(state, "data_timeframe", config.DATA_TIMEFRAME_DEFAULT)),
                "data_lookback": int(getattr(state, "data_lookback", config.DATA_LOOKBACK_CANDLES_DEFAULT)),
                "data_refresh_sec": int(getattr(state, "data_refresh_sec", config.DATA_REFRESH_SEC_DEFAULT)),
                "adaptive_profile": str(getattr(state, "adaptive_profile", config.ADAPTIVE_PROFILE_DEFAULT)),
                "adaptive_timeframes": list(getattr(state, "adaptive_timeframes", [])),
                "adaptive_weights": list(getattr(state, "adaptive_weights", [])),
                "adaptive_k_buy": float(getattr(state, "adaptive_k_buy", config.ADAPTIVE_PRESETS[config.ADAPTIVE_PROFILE_DEFAULT]["k_buy"])),
                "adaptive_k_sell": float(getattr(state, "adaptive_k_sell", config.ADAPTIVE_PRESETS[config.ADAPTIVE_PROFILE_DEFAULT]["k_sell"])),
                "adaptive_alpha": float(getattr(state, "adaptive_alpha", config.ADAPTIVE_PRESETS[config.ADAPTIVE_PROFILE_DEFAULT]["alpha"])),
                "blended_atr_per_min_pct": float(getattr(state, "blended_atr_per_min_pct", 0.0)),

                "avg_high": float(getattr(state, "avg_high", 0.0)),
                "avg_low": float(getattr(state, "avg_low", 0.0)),
                "avg_range_pct": float(getattr(state, "avg_range_pct", 0.0)),
                "data_trend": str(getattr(state, "data_trend", "?") or "?"),
                "effective_buy_threshold_pct": float(getattr(state, "effective_buy_threshold_pct", state.buy_threshold_pct)),
                "effective_sell_threshold_pct": float(getattr(state, "effective_sell_threshold_pct", state.sell_threshold_pct)),

                # advanced globals
                "STOP_LOSS_PCT": config.STOP_LOSS_PCT,
                "DAILY_MAX_LOSS_USDT": config.DAILY_MAX_LOSS_USDT,
                "DAILY_MAX_PROFIT_USDT": config.DAILY_MAX_PROFIT_USDT,
                "ANCHOR_MAX_AGE_SEC": config.ANCHOR_MAX_AGE_SEC,
                "ANCHOR_DRIFT_REARM_PCT": config.ANCHOR_DRIFT_REARM_PCT,
                "POLL_INTERVAL_SECONDS": config.POLL_INTERVAL_SECONDS,
                "MIN_NOTIONAL_USDT": config.MIN_NOTIONAL_USDT,
                "MAX_STOPS_PER_HOUR": config.MAX_STOPS_PER_HOUR,
                "CIRCUIT_STOP_COOLDOWN_SEC": config.CIRCUIT_STOP_COOLDOWN_SEC,
            }
        _atomic_write_json(config.CONFIG_PATH, payload)
    except Exception:
        # Never crash the bot because saving failed.
        pass
