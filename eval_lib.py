"""Tahmin degerlendirme + islem simulasyonu icin paylasilan, ag erisimi olmayan
saf fonksiyonlar. Hem scripts/backtest.py (gecmise donuk simulasyon) hem de
scripts/evaluate_archive.py (docs/history/ arsivindeki gercek tahminleri
gerceklesen fiyatlarla kiyaslama) bunu kullanir - mantik iki yerde ayri ayri
yazilmaz.
"""
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

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

def compound_equity_curve(trades: list, start_equity: float = 1.0) -> list:
    """trades'i entry_ts sirasina gore BILESIK olarak birlestirir (her islem sermayenin
    TAMAMINI kullanir varsayimiyla - ORTUSEN (independent mod) islemlerde bu gercekci
    DEGILDIR, sadece coin basina TEK POZISYON modunda anlamlidir, bkz. aggregate_trades).
    Doner: [(entry_ts, equity), ...] - equity 1.0'dan baslar (yuzde degil, carpan)."""
    ordered = sorted(trades, key=lambda t: t.get("entry_ts") or "")
    equity = start_equity
    curve = [(None, equity)]
    for t in ordered:
        pnl = t.get("pnl_pct")
        if pnl is None:
            continue
        # Tek bir islemde -%100'den fazla kayip (kaldiracli likidasyon disinda) olmasin
        # diye tabanli (equity negatife dusmesin).
        equity = max(equity * (1 + pnl / 100), 0.0)
        curve.append((t.get("entry_ts"), equity))
    return curve


def real_max_drawdown_pct(equity_curve: list) -> float:
    """Standart tanim: tepe noktadan yuzde kac dustugu (bileşik equity egrisinden).
    Sonuc (-100, 0] araliginda - additive/toplamsal versiyondaki gibi -%1773 gibi
    'imkansiz' degerler CIKAMAZ (equity negatife dusemedigi icin taban -%100)."""
    if not equity_curve:
        return None
    peak = equity_curve[0][1]
    max_dd = 0.0
    for _, eq in equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            dd = (eq / peak - 1) * 100
            max_dd = min(max_dd, dd)
    return round(max_dd, 4)


def compound_return_pct(trades: list) -> float:
    """Bilesik toplam getiri (%) - compound_equity_curve'un son degeri."""
    curve = compound_equity_curve(trades)
    if len(curve) < 2:
        return None
    return round((curve[-1][1] - 1) * 100, 4)


def aggregate_trades(trades: list) -> dict:
    """trades: dict listesi, her biri en az {exit_reason, pnl_pct, entry_ts} icerir.

    IKI FARKLI GETIRI METRIGI raporlanir, KARISTIRILMAMALI:
      - total_return_pct: TOPLAMSAL (additive) - her islem sabit/kucuk bir pozisyon
        buyuklugu ile ACILIP sermayeye geri EKLENMEDEN yan yana konsa ne olurdu
        (orn. her islem icin ayni sabit teminati her seferinde yeniden kullanma).
        ORTUSEN (independent mod) islemler icin bu daha anlamlidir (ayni anda birden
        cok pozisyon acik olabilir, "sermayenin tamami" kavrami yoktur).
      - compound_return_pct: BILESIK (compound) - sermayenin TAMAMININ her islemden
        SONRAKI islem icin yeniden kullanildigi varsayimiyla. Bu SADECE islemler
        ORTUSMUYORSA (coin basina TEK POZISYON modu) gercekci bir yorumdur; ortusen
        islemlerde (independent mod) 'ayni anda birden fazla islemde ayni sermaye'
        anlamina gelecegi icin YANILTICI olur - yine de hesaplanir ama boyle
        etiketlenmelidir (bkz. caller).
    max_drawdown_pct alanı artik BILESIK equity egrisinden (gercek, sinirli) hesaplanir;
    eski toplamsal versiyon max_drawdown_pct_additive olarak ayrica tutulur."""
    n = len(trades)
    if n == 0:
        return {
            "count": 0, "win_rate_pct": None, "direction_accuracy_pct": None,
            "exit_distribution": {}, "avg_win_pct": None, "avg_loss_pct": None,
            "total_return_pct": None, "compound_return_pct": None,
            "max_drawdown_pct": None, "max_drawdown_pct_additive": None,
            "profit_factor": None,
        }

    wins = [t for t in trades if (t.get("pnl_pct") or 0) > 0]
    losses = [t for t in trades if (t.get("pnl_pct") or 0) <= 0]
    exit_dist = {}
    for t in trades:
        exit_dist[t["exit_reason"]] = exit_dist.get(t["exit_reason"], 0) + 1

    ordered = sorted(trades, key=lambda t: t.get("entry_ts") or "")
    cum = 0.0
    peak = 0.0
    max_dd_additive = 0.0
    for t in ordered:
        cum += t.get("pnl_pct") or 0.0
        peak = max(peak, cum)
        max_dd_additive = min(max_dd_additive, cum - peak)

    equity_curve = compound_equity_curve(trades)
    compound_total = round((equity_curve[-1][1] - 1) * 100, 4) if len(equity_curve) > 1 else None
    real_dd = real_max_drawdown_pct(equity_curve)

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
        "compound_return_pct": compound_total,
        "max_drawdown_pct": real_dd,
        "max_drawdown_pct_additive": round(max_dd_additive, 4),
        # gross_loss == 0 iken sonsuz olur - JSON'da gecersiz oldugu icin None birakilir
        # (frontend "kayipsiz" olarak yorumlar; ayirt etmek icin avg_loss_pct zaten None olur).
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else None,
    }


def build_portfolio_equity_curve(trades_by_coin: dict, start_equity: float = 1.0,
                                  full_date_range: tuple = None) -> list:
    """Esit agirlikli, GUNLUK yeniden dengelenen portfoy equity egrisi kurar.

    trades_by_coin: {coin: [trade, ...]} - HER COIN'IN KENDI (tercihen coin basina
    TEK POZISYON modundaki, ORTUSMEYEN) islem listesi. Yontem: her islemin pnl'i
    KAPANIS (exit_ts) gununde gerceklesmis sayilir; o gun herhangi bir coin'de kapanan
    islem varsa, o coin'in getirisi 1/N agirlikla portfoy gunluk getirisine katilir
    (o gun kapanan islemi olmayan coinler o gun icin %0/nakit sayilir). Portfoy
    getirisi bu gunluk degerlerin BILESIGIDIR.

    full_date_range: (start_date, end_date) "YYYY-MM-DD" verilirse, ARADAKI TUM
    gunler (islem kapanmayanlar dahil, %0 getiriyle) egriye dahil edilir - farkli
    stratejilerin gunluk getiri serilerini TARIHE GORE HIZALI (eslesmis cift) hale
    getirmek icin gerekir (orn. iki stratejinin anlamlilik testinde kiyaslanmasi).

    Bu, 'N coin'in getirisini duz toplama' yerine standart bir esit-agirlikli,
    yeniden dengelenen portfoy yaklasimidir - MAX DUSUS artik gercek ve sinirlidir."""
    coins = list(trades_by_coin.keys())
    n_coins = len(coins)
    if n_coins == 0:
        return []

    # gun -> {coin: o gun kapanan islemlerin toplam pnl_pct'i (birden fazlaysa toplanir)}
    daily = {}
    for coin, trades in trades_by_coin.items():
        for t in trades:
            exit_ts = t.get("exit_ts")
            pnl = t.get("pnl_pct")
            if exit_ts is None or pnl is None:
                continue
            day = exit_ts[:10]  # "YYYY-MM-DD"
            daily.setdefault(day, {}).setdefault(coin, 0.0)
            daily[day][coin] += pnl

    if full_date_range is not None:
        start_day, end_day = full_date_range
        d = datetime.strptime(start_day, "%Y-%m-%d")
        end_d = datetime.strptime(end_day, "%Y-%m-%d")
        while d <= end_d:
            daily.setdefault(d.strftime("%Y-%m-%d"), {})
            d += timedelta(days=1)

    equity = start_equity
    curve = [(None, equity)]
    for day in sorted(daily.keys()):
        day_returns = daily[day]
        portfolio_return_pct = sum(day_returns.get(c, 0.0) for c in coins) / n_coins
        equity = max(equity * (1 + portfolio_return_pct / 100), 0.0)
        curve.append((day, equity))
    return curve


def daily_returns_from_equity_curve(curve: list) -> list:
    """[(gun, equity), ...] -> gunluk yuzde getiri listesi, kronolojik sirada."""
    returns = []
    for i in range(1, len(curve)):
        prev_eq = curve[i - 1][1]
        eq = curve[i][1]
        returns.append((eq / prev_eq - 1) * 100 if prev_eq > 0 else 0.0)
    return returns


# ---------------------------------------------------------------------------
# Perp funding maliyeti
# ---------------------------------------------------------------------------

FUNDING_HOURS = (0, 8, 16)  # UTC, OKX'in standart funding saatleri
DEFAULT_FUNDING_RATE_PCT = 0.01  # gercek veri yoksa varsayilan (%0.01 / 8s - tipik/ortalama bir deger)


def funding_events_between(entry_ts, exit_ts) -> list:
    """entry (dahil degil) ile exit (dahil) arasinda gerceklesen funding anlari (00/08/16 UTC)."""
    if exit_ts is None or entry_ts is None or exit_ts <= entry_ts:
        return []
    t = entry_ts.replace(minute=0, second=0, microsecond=0)
    if t <= entry_ts:
        t += timedelta(hours=1)
    while t.hour not in FUNDING_HOURS:
        t += timedelta(hours=1)
    events = []
    while t <= exit_ts:
        events.append(t)
        t += timedelta(hours=8)
    return events


def funding_cost_pct(entry_ts, exit_ts, side: str, leverage: float, funding_lookup: dict = None,
                      default_rate_pct: float = DEFAULT_FUNDING_RATE_PCT) -> float:
    """Pozisyon acikken gecen her funding anindaki maliyeti toplar (yuzde, marjin
    uzerinden). funding_lookup: {datetime: oransal funding_rate (orn. 0.0001=%0.01)}
    ya da None. Bir an icin veri yoksa default_rate_pct (yuzde) kullanilir - long
    icin maliyet (rate>0 iken), short icin ters isaretli (rate>0 iken kazanc)."""
    events = funding_events_between(entry_ts, exit_ts)
    if not events:
        return 0.0
    sign = 1 if side == "BUY" else -1
    total = 0.0
    for ts in events:
        rate = funding_lookup.get(ts) if funding_lookup is not None else None
        rate_pct = rate * 100 if rate is not None else default_rate_pct
        total += sign * rate_pct * leverage
    return round(total, 6)


# ---------------------------------------------------------------------------
# Olasilik filtresi (Varyant D) - kantillerden P(fiyat > giris) enterpolasyonu
# ---------------------------------------------------------------------------

def prob_above_entry(entry_price: float, quantile_row) -> float:
    """quantile_row: en az 10 deger [ozel_kolon, q10, q20, ..., q90] (fix_quantile_crossing
    sayesinde artan sirada). P(gercek fiyat > entry_price) tahminini, bilinen 9 kantil
    noktasi (0.1..0.9) arasinda parcali dogrusal CDF enterpolasyonuyla dondurur.
    Bandin disina tasan durumlarda kaba bir sinir (0.95/0.05) doner."""
    if quantile_row is None or len(quantile_row) < 10 or entry_price is None:
        return None
    qs = np.asarray(quantile_row, dtype=float)[1:]
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    if entry_price <= qs[0]:
        return 0.95
    if entry_price >= qs[-1]:
        return 0.05
    cdf_at_entry = float(np.interp(entry_price, qs, probs))
    return round(1 - cdf_at_entry, 4)


# ---------------------------------------------------------------------------
# Blok bootstrap guven araligi (zaman serisi bagimliligina karsi basit iid
# bootstrap yerine kullanilir)
# ---------------------------------------------------------------------------

def block_bootstrap_ci(values, block_size: int = 20, n_boot: int = 2000, ci: float = 0.95,
                        seed: int = 42, statistic=np.mean) -> dict:
    """Kronolojik sirali degerler (orn. islem pnl_pct'leri) icin blok bootstrap
    %ci guven araligi. Komsu degerler blok halinde orneklenir, boylece kisa vadeli
    bagimlilik (ust uste binen pencereler, ardisik piyasa rejimi) bir olcude
    korunur. `effective_n` = n // block_size, kaba bir "bagimsiz ornek sayisi"
    tahminidir (blok icindeki degerler tek bir bagimsiz gozlem gibi sayilir)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0, "effective_n": 0,
                "block_size": block_size, "n_boot": n_boot}
    block_size = max(1, min(block_size, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([values[s:s + block_size] for s in starts])[:n]
        boot_stats[i] = statistic(sample)
    boot_stats.sort()
    lo_i = int((1 - ci) / 2 * n_boot)
    hi_i = min(n_boot - 1, int((1 + ci) / 2 * n_boot))
    effective_n = max(1, n // block_size)
    return {
        "mean": round(float(statistic(values)), 4),
        "ci_low": round(float(boot_stats[lo_i]), 4),
        "ci_high": round(float(boot_stats[hi_i]), 4),
        "n": n,
        "effective_n": effective_n,
        "block_size": block_size,
        "n_boot": n_boot,
    }


def block_bootstrap_compound_return(pnls_in_order, block_size: int = 20, n_boot: int = 2000,
                                     ci: float = 0.95, seed: int = 42) -> dict:
    """block_bootstrap_ci'nin BILESIK GETIRI versiyonu: statistic=np.mean yerine her
    bootstrap orneginde BILESIK TOPLAM GETIRIYI ((prod(1+r/100)-1)*100) hesaplar, boylece
    donen CI, 'compound_return_pct' nokta tahminiyle AYNI OLCEKTEDIR (eski hata: CI
    islem-basina-ORTALAMA olcegindeydi, nokta tahmin ise TOPLAM/bilesik olcekteydi).
    pnls_in_order: ZATEN kronolojik sirali pnl_pct listesi (entry_ts'e gore)."""
    values = np.asarray(pnls_in_order, dtype=float)
    n = len(values)
    if n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0, "effective_n": 0,
                "block_size": block_size, "n_boot": n_boot}
    block_size = max(1, min(block_size, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))

    def compound_stat(sample):
        return (np.prod(1 + sample / 100) - 1) * 100

    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([values[s:s + block_size] for s in starts])[:n]
        boot_stats[i] = compound_stat(sample)
    boot_stats.sort()
    lo_i = int((1 - ci) / 2 * n_boot)
    hi_i = min(n_boot - 1, int((1 + ci) / 2 * n_boot))
    effective_n = max(1, n // block_size)
    return {
        "mean": round(float(compound_stat(values)), 4),
        "ci_low": round(float(boot_stats[lo_i]), 4),
        "ci_high": round(float(boot_stats[hi_i]), 4),
        "n": n,
        "effective_n": effective_n,
        "block_size": block_size,
        "n_boot": n_boot,
    }


def paired_diff_significance(a_values, b_values, block_size: int = 20, n_boot: int = 2000,
                              ci: float = 0.95, seed: int = 42) -> dict:
    """a-b farkinin (eslesmis - ayni sinyal/kesim noktasindan gelen ciftler) blok
    bootstrap guven araligi sifiri kapsiyor mu. Kapsiyorsa `significant=False`
    ("anlamli fark yok"); a ve b farkli uzunluktaysa kisa olana kirpilir."""
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return {"significant": None, "diff_ci": block_bootstrap_ci([], block_size, n_boot, ci, seed)}
    diff = a[:n] - b[:n]
    res = block_bootstrap_ci(diff, block_size=block_size, n_boot=n_boot, ci=ci, seed=seed)
    significant = res["ci_low"] is not None and not (res["ci_low"] <= 0 <= res["ci_high"])
    return {"significant": significant, "diff_ci": res}
