"""Tahmin degerlendirme + islem simulasyonu icin paylasilan, ag erisimi olmayan
saf fonksiyonlar. Hem scripts/backtest.py (gecmise donuk simulasyon) hem de
scripts/evaluate_archive.py (docs/history/ arsivindeki gercek tahminleri
gerceklesen fiyatlarla kiyaslama) bunu kullanir - mantik iki yerde ayri ayri
yazilmaz.
"""
import random
from dataclasses import dataclass, field

# Varsayilan komisyon: islem basi (giris + cikis) %0.05 + %0.05, OKX taker'a yakin.
DEFAULT_FEE_PCT = 0.05
DEFAULT_LEVERAGE = 1.0
# Ayni mumda hem TP hem SL/likidasyon degilirse, kayip sayilir (temkinli varsayim).
_LOSS_REASONS = {"SL", "LIQUIDATION"}


# ---------------------------------------------------------------------------
# Yon dogrulugu / tahmin hatasi
# ---------------------------------------------------------------------------

def direction_correct(side: str, entry_price: float, actual_price) -> bool:
    """actual_price None ise None doner (henuz gerceklesmemis)."""
    if actual_price is None or entry_price is None:
        return None
    if side == "BUY":
        return actual_price > entry_price
    if side == "SELL":
        return actual_price < entry_price
    return None


def forecast_error(predicted_value, actual_value) -> tuple:
    """(mutlak hata, yuzde hata) - actual_value None/0 ise (None, None)."""
    if predicted_value is None or actual_value is None or actual_value == 0:
        return None, None
    abs_err = abs(predicted_value - actual_value)
    pct_err = abs_err / abs(actual_value) * 100
    return abs_err, pct_err


def in_confidence_band(actual_value, lower, upper) -> bool:
    if actual_value is None or lower is None or upper is None:
        return None
    return lower <= actual_value <= upper


# ---------------------------------------------------------------------------
# Likidasyon fiyati (basitlestirilmis - bakim marjini/funding ihmal edilir;
# crypto-trader'in kendi backtest'indeki gibi "yaklasik likidasyon kontrolu")
# ---------------------------------------------------------------------------

def liquidation_price(entry_price: float, side: str, leverage: float):
    if leverage is None or leverage <= 1.0:
        return None
    if side == "BUY":
        return entry_price * (1 - 1 / leverage)
    return entry_price * (1 + 1 / leverage)


# ---------------------------------------------------------------------------
# Islem simulasyonu
# ---------------------------------------------------------------------------

@dataclass
class TradeResult:
    exit_reason: str  # "TP" | "SL" | "LIQUIDATION" | "EXPIRED" | "INSUFFICIENT_DATA"
    exit_ts: str = None
    exit_price: float = None
    hours_held: int = None
    pnl_pct: float = None  # kaldiracli, komisyon dusulmus, marjin uzerinden yuzde


def simulate_trade(entry_price: float, side: str, take_profit: float, stop_loss: float,
                    future_candles: list, leverage: float = DEFAULT_LEVERAGE,
                    fee_pct: float = DEFAULT_FEE_PCT, horizon_hours: int = 48) -> TradeResult:
    """future_candles: entry'den SONRAKI saatlik mumlar, kronolojik sira, her biri
    en az {ts, high, low, close} icerir. horizon_hours kadar (ya da elde ne varsa)
    ileri bakip TP/SL/likidasyon/sure dolumunu tespit eder.

    Ayni mum icinde hem TP hem SL(/likidasyon) tetiklenirse temkinli davranilir:
    kayip (SL ya da likidasyon, hangisi daha yakinsa) sayilir.
    """
    if not future_candles:
        return TradeResult(exit_reason="INSUFFICIENT_DATA")

    liq = liquidation_price(entry_price, side, leverage)
    if side == "BUY":
        # Fiyat dusunce once hangisine (SL mi likidasyon mu) daha once carpar -
        # ikisinden entry'ye daha YAKIN (daha yuksek) olan.
        effective_stop = max(stop_loss, liq) if liq is not None else stop_loss
        stop_is_liq = liq is not None and effective_stop == liq
    else:
        effective_stop = min(stop_loss, liq) if liq is not None else stop_loss
        stop_is_liq = liq is not None and effective_stop == liq

    candles = future_candles[:horizon_hours]
    for i, candle in enumerate(candles):
        high, low = candle["high"], candle["low"]
        if side == "BUY":
            tp_hit = high >= take_profit
            stop_hit = low <= effective_stop
        else:
            tp_hit = low <= take_profit
            stop_hit = high >= effective_stop

        if tp_hit and stop_hit:
            exit_reason = "LIQUIDATION" if stop_is_liq else "SL"
            exit_price = effective_stop
        elif stop_hit:
            exit_reason = "LIQUIDATION" if stop_is_liq else "SL"
            exit_price = effective_stop
        elif tp_hit:
            exit_reason = "TP"
            exit_price = take_profit
        else:
            continue

        pnl_pct = _pnl_pct(entry_price, exit_price, side, leverage, fee_pct, is_liquidation=stop_is_liq and exit_reason == "LIQUIDATION")
        return TradeResult(
            exit_reason=exit_reason, exit_ts=candle["ts"], exit_price=exit_price,
            hours_held=i + 1, pnl_pct=pnl_pct,
        )

    # Ne TP ne SL/likidasyon tetiklendi -> ufkun sonunda (ya da elimizdeki son
    # mumda, veri kisaysa) kapaniyor.
    last = candles[-1]
    pnl_pct = _pnl_pct(entry_price, last["close"], side, leverage, fee_pct, is_liquidation=False)
    exit_reason = "EXPIRED" if len(candles) >= horizon_hours else "INSUFFICIENT_DATA"
    return TradeResult(
        exit_reason=exit_reason, exit_ts=last["ts"], exit_price=last["close"],
        hours_held=len(candles), pnl_pct=pnl_pct,
    )


def _pnl_pct(entry_price, exit_price, side, leverage, fee_pct, is_liquidation) -> float:
    if is_liquidation:
        # Likidasyonda marjinin (hemen hemen) tamami kaybedilir - basitlestirilmis.
        return -100.0
    price_move_pct = (exit_price - entry_price) / entry_price * (1 if side == "BUY" else -1) * 100
    leveraged_pct = price_move_pct * leverage
    fee_cost_pct = 2 * fee_pct * leverage  # giris + cikis, notional = marjin*kaldirac
    return leveraged_pct - fee_cost_pct


def price_at_horizon(future_candles: list, hours: int):
    """entry'den `hours` saat sonraki KAPANIS fiyati (varsa)."""
    if not future_candles or hours < 1 or hours > len(future_candles):
        return None
    return future_candles[hours - 1]["close"]


# ---------------------------------------------------------------------------
# Karsilastirma stratejileri - ayni entry/ATR/TP-SL cercevesini kullanir, sadece
# yon (side) secimi farkli.
# ---------------------------------------------------------------------------

def always_buy_side(**kwargs) -> str:
    return "BUY"


def momentum_side(context_closes, **kwargs) -> str:
    """Baglam serisinin son 48 saatteki (ya da elde ne varsa) yonunu takip eder."""
    if context_closes is None or len(context_closes) < 2:
        return "BUY"
    window = context_closes[-48:] if len(context_closes) >= 48 else context_closes
    return "BUY" if window[-1] >= window[0] else "SELL"


def random_side(rng: random.Random = None, **kwargs) -> str:
    rng = rng or random
    return rng.choice(["BUY", "SELL"])


COMPARISON_STRATEGIES = {
    "always_buy": always_buy_side,
    "momentum": momentum_side,
    "random": random_side,
}


# ---------------------------------------------------------------------------
# Toplulastirma
# ---------------------------------------------------------------------------

def aggregate_trades(trades: list) -> dict:
    """trades: dict listesi, her biri en az {exit_reason, pnl_pct, entry_ts} icerir.
    pnl_pct'ler sabit pozisyon buyuklugu varsayimiyla TOPLANIR (bilesik faiz degil) -
    bu bir basitlestirme, bkz. panel/README aciklamasi."""
    n = len(trades)
    if n == 0:
        return {
            "count": 0, "win_rate_pct": None, "direction_accuracy_pct": None,
            "exit_distribution": {}, "avg_win_pct": None, "avg_loss_pct": None,
            "total_return_pct": None, "max_drawdown_pct": None, "profit_factor": None,
        }

    wins = [t for t in trades if (t.get("pnl_pct") or 0) > 0]
    losses = [t for t in trades if (t.get("pnl_pct") or 0) <= 0]
    exit_dist = {}
    for t in trades:
        exit_dist[t["exit_reason"]] = exit_dist.get(t["exit_reason"], 0) + 1

    ordered = sorted(trades, key=lambda t: t.get("entry_ts") or "")
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in ordered:
        cum += t.get("pnl_pct") or 0.0
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)

    gross_win = sum(t["pnl_pct"] for t in wins)
    gross_loss = abs(sum(t["pnl_pct"] for t in losses))
    direction_flags = [t["direction_correct"] for t in trades if t.get("direction_correct") is not None]

    return {
        "count": n,
        "win_rate_pct": round(100 * len(wins) / n, 2),
        "direction_accuracy_pct": round(100 * sum(direction_flags) / len(direction_flags), 2) if direction_flags else None,
        "exit_distribution": exit_dist,
        "avg_win_pct": round(sum(t["pnl_pct"] for t in wins) / len(wins), 4) if wins else None,
        "avg_loss_pct": round(sum(t["pnl_pct"] for t in losses) / len(losses), 4) if losses else None,
        "total_return_pct": round(cum, 4),
        "max_drawdown_pct": round(max_dd, 4),
        # gross_loss == 0 iken sonsuz olur - JSON'da gecersiz oldugu icin None birakilir
        # (frontend "kayipsiz" olarak yorumlar; ayirt etmek icin avg_loss_pct zaten None olur).
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else None,
    }
