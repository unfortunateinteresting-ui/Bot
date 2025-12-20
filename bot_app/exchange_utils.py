import pandas as pd
import ccxt

from bot_app import config


def init_exchange() -> ccxt.Exchange:
    exchange_class = getattr(ccxt, config.EXCHANGE_NAME)
    exchange = exchange_class({
        "apiKey": config.API_KEY,
        "secret": config.SECRET_KEY,
        "enableRateLimit": True,
    })
    if hasattr(exchange, "options"):
        exchange.options["createMarketBuyOrderRequiresPrice"] = False
    exchange.load_markets()
    return exchange


def fetch_ohlcv_df(exchange: ccxt.Exchange) -> pd.DataFrame | None:
    try:
        ohlcv = exchange.fetch_ohlcv(config.SYMBOL, timeframe=config.TIMEFRAME, limit=config.HISTORY_LIMIT)
    except Exception as e:
        print(f"Error fetching OHLCV: {e}", flush=True)
        return None

    if not ohlcv:
        return None

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    return df


def simulate_dry_run_market_fill(exchange: ccxt.Exchange, symbol: str, side: str, qty_base: float,
                                 price_fallback: float):
    """Simulate a market fill from the current order book."""
    if qty_base <= 0:
        return 0.0, price_fallback, price_fallback, price_fallback, 0

    try:
        ob = exchange.fetch_order_book(symbol, limit=config.SLIPPAGE_ORDERBOOK_LEVELS)
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

        if filled <= 0:
            return qty_base, price_fallback, price_fallback, price_fallback, 0

        if remaining > 0:
            bps = config.SLIPPAGE_FALLBACK_BPS / 10000.0
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
        return qty_base, price_fallback, price_fallback, price_fallback, 0


def fetch_best_book_price(exchange: ccxt.Exchange, symbol: str, side: str, price_fallback: float) -> float:
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
        ticker = exchange.fetch_ticker(symbol) or {}
        if side == "buy":
            value = ticker.get("ask") or ticker.get("last") or price_fallback
        else:
            value = ticker.get("bid") or ticker.get("last") or price_fallback
        return float(value or price_fallback)
    except Exception:
        return float(price_fallback)
