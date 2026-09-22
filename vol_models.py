"""Oynaklik (log-RV) tahmin yontemleri - 6 tanesi, hepsi AYNI veri/kesim noktalariyla
calisir ki kiyas adil olsun. Ucu (naive, EWMA) parametre gerektirmez; HAR-RV ve
GARCH(1,1) SADECE gelistirme (dev) doneminde FIT edilir, sonra hem dev'de (walk-
forward, ayni sabit katsayilarla) hem kilitli testte (donuk katsayilarla, YENIDEN
FIT EDILMEDEN) uygulanir - "yontem secimi sadece gelistirme doneminde" kuralina
uyar. TimesFM-direct ve TimesFM-hibrit hicbir fit gerektirmez (zero-shot / mevcut
tahmin onbelleginden turetilir).

Hepsi log(RV) DONDURUR (RV degil) - dogrudan karsilastirilabilir, log-olcekli hedef.
"""
import numpy as np
import pandas as pd

from realized_vol import (
    RV_EPSILON,
    build_hourly_log_rv_series,
    log_rv,
    rv_at_cutpoint,
)

EWMA_LAMBDA = 0.94  # RiskMetrics standardi
HAR_COMPONENT_HOURS = {"daily": 24, "weekly": 24 * 5, "monthly": 24 * 22}


# ---------------------------------------------------------------------------
# 1) Naif: son H saatin RV'si -> sonraki H saat icin tahmin
# ---------------------------------------------------------------------------

def forecast_naive(candles: list, idx_by_ts: dict, cutpoint_ts: str, horizon_hours: int, method: str = "cc"):
    rv = rv_at_cutpoint(candles, idx_by_ts, cutpoint_ts, horizon_hours, method=method, lookback=True)
    return log_rv(rv)


# ---------------------------------------------------------------------------
# 2) EWMA / RiskMetrics (lambda=0.94)
# ---------------------------------------------------------------------------

def build_ewma_variance_series(candles: list, lam: float = EWMA_LAMBDA) -> dict:
    """Her saat icin, o ana kadarki TUM gecmisi kullanan EWMA (RiskMetrics) SAATLIK
    varyans tahminini hesaplar: sigma2_t = lam*sigma2_{t-1} + (1-lam)*r_t^2.
    Doner: {ts: sigma2} (o saatin KAPANISINDAKI tahmin - bir sonraki saate iliskin)."""
    closes = np.array([c["close"] for c in candles], dtype=float)
    r = np.diff(np.log(closes))
    r2 = pd.Series(r ** 2)
    ewma_var = r2.ewm(alpha=(1 - lam), adjust=False).mean().to_numpy()
    # ewma_var[i], candles[i+1]'in saatlik getirisinden (r[i]) hesaplandi - yani
    # candles[i+1] zamaninda "bilinen" bir tahmindir.
    out = {}
    for i, val in enumerate(ewma_var):
        out[candles[i + 1]["ts"]] = float(val)
    return out


def forecast_ewma(ewma_var_by_ts: dict, cutpoint_ts: str, horizon_hours: int):
    """H saatlik RV tahmini = saatlik EWMA varyans * H (iid varsayimiyla dogrusal olceklenir)."""
    sigma2 = ewma_var_by_ts.get(cutpoint_ts)
    if sigma2 is None:
        return None
    return log_rv(sigma2 * horizon_hours)


# ---------------------------------------------------------------------------
# 3) HAR-RV (Corsi 2009) - gunluk/haftalik/aylik RV bilesenleriyle OLS regresyon
# ---------------------------------------------------------------------------

def build_har_features(candles: list) -> dict:
    """Her saat icin (RVd, RVw, RVm) uclusunu hesaplar - trailing gunluk/haftalik/
    aylik RV (RV, log DEGIL - regresyon icin log'u ayri alinir). Doner: {ts: (RVd,RVw,RVm)}."""
    daily = build_hourly_log_rv_series(candles, window_hours=HAR_COMPONENT_HOURS["daily"], method="cc")
    weekly = build_hourly_log_rv_series(candles, window_hours=HAR_COMPONENT_HOURS["weekly"], method="cc")
    monthly = build_hourly_log_rv_series(candles, window_hours=HAR_COMPONENT_HOURS["monthly"], method="cc")
    out = {}
    for i, c in enumerate(candles):
        if daily[i] is None or weekly[i] is None or monthly[i] is None:
            continue
        # build_hourly_log_rv_series LOG(RV) donuyor; HAR ozellikleri icin RV'ye geri donduruyoruz
        # (regresyonda log'u tekrar aliniyor - build_har_features sadece dogrusal/carpimsal
        # olcekte oldugu icin exp(log_rv)-epsilon ~ RV, hafif yuvarlama farkiyla).
        rvd = np.exp(daily[i]) - RV_EPSILON
        rvw = np.exp(weekly[i]) - RV_EPSILON
        rvm = np.exp(monthly[i]) - RV_EPSILON
        out[c["ts"]] = (max(rvd, 0.0), max(rvw, 0.0), max(rvm, 0.0))
    return out


def fit_har_rv(har_features_by_coin: dict, targets_by_coin: dict, horizon_hours: int) -> dict:
    """TUM coinleri BIRLESTIRIP (havuzlanmis) TEK bir OLS regresyonu fit eder:
    log(RV_target) = b0 + b1*log(RVd) + b2*log(RVw) + b3*log(RVm).
    har_features_by_coin: {coin: {ts: (RVd,RVw,RVm)}}, targets_by_coin: {coin: {ts: RV_target}}
    (RV, log DEGIL). SADECE gelistirme donemindeki (ts, target) ciftleriyle cagirilmali.
    Doner: {"coef": [b0,b1,b2,b3], "n_obs": int} - katsayilar donuk, kilitli testte
    YENIDEN FIT EDILMEDEN kullanilir."""
    import statsmodels.api as sm

    rows_x, rows_y = [], []
    for coin, feats in har_features_by_coin.items():
        targets = targets_by_coin.get(coin, {})
        for ts, (rvd, rvw, rvm) in feats.items():
            target = targets.get(ts)
            if target is None or rvd <= 0 or rvw <= 0 or rvm <= 0:
                continue
            rows_x.append([np.log(rvd), np.log(rvw), np.log(rvm)])
            rows_y.append(log_rv(target))

    if len(rows_y) < 30:
        return {"coef": None, "n_obs": len(rows_y)}

    X = sm.add_constant(np.array(rows_x))
    y = np.array(rows_y)
    model = sm.OLS(y, X).fit()
    return {"coef": model.params.tolist(), "n_obs": len(rows_y), "r2_insample": float(model.rsquared)}


def forecast_har_rv(har_coef: list, har_features_by_ts: dict, cutpoint_ts: str):
    if har_coef is None:
        return None
    feats = har_features_by_ts.get(cutpoint_ts)
    if feats is None:
        return None
    rvd, rvw, rvm = feats
    if rvd <= 0 or rvw <= 0 or rvm <= 0:
        return None
    b0, b1, b2, b3 = har_coef
    return float(b0 + b1 * np.log(rvd) + b2 * np.log(rvw) + b3 * np.log(rvm))


# ---------------------------------------------------------------------------
# 4) GARCH(1,1) - arch paketiyle, saatlik getiriler uzerinde
# ---------------------------------------------------------------------------

def fit_garch(hourly_returns_pct: np.ndarray) -> dict:
    """GARCH(1,1)'i SADECE gelistirme donemindeki saatlik getirilerle (yuzde olcekli,
    arch paketinin numerik kararliligi icin) fit eder. Doner: fit edilmis arch sonuc
    nesnesinin parametreleri (dict) - donuk, kilitli testte yeniden fit edilmez."""
    from arch import arch_model

    am = arch_model(hourly_returns_pct, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
    res = am.fit(disp="off")
    return {"omega": float(res.params["omega"]), "alpha": float(res.params["alpha[1]"]),
            "beta": float(res.params["beta[1]"]), "n_obs": len(hourly_returns_pct)}


def build_garch_variance_series(candles: list, garch_params: dict) -> dict:
    """Fit edilmis (donuk) GARCH(1,1) katsayilarini TUM seri (dev+kilitli, gecmisten
    bugune) uzerinde REKURSIF olarak ileri yururur (yeniden fit YOK, sadece filtre
    uygulama) - her saat icin bir sonraki saatin kosullu varyans tahminini verir.
    Getiriler YUZDE olceginde (fit ile tutarli)."""
    if garch_params.get("omega") is None:
        return {}
    closes = np.array([c["close"] for c in candles], dtype=float)
    r_pct = np.diff(np.log(closes)) * 100
    omega, alpha, beta = garch_params["omega"], garch_params["alpha"], garch_params["beta"]

    long_run_var = omega / max(1 - alpha - beta, 1e-6)
    sigma2 = long_run_var
    out = {}
    for i in range(len(r_pct)):
        # sigma2 su an r_pct[i-1]'e kadar bilgiyi yansitiyor -> candles[i]'nin (bir
        # onceki saatin sonu) tahmini, candles[i+1] zamaninda "bilinir" sayilir.
        out[candles[i + 1]["ts"]] = sigma2 / 10000  # yuzdeden geri donustur (varyans olcegi)
        sigma2 = omega + alpha * (r_pct[i] ** 2) + beta * sigma2
    return out


def forecast_garch(garch_var_by_ts: dict, garch_params: dict, cutpoint_ts: str, horizon_hours: int):
    """H saatlik RV tahmini: GARCH kosullu varyansinin H adim ileriye kapali-form
    toplami (arch'in coklu-adim varyans toplaminin ayni GARCH(1,1) rekursiyonundan
    analitik turevi): sum_{k=1}^{H} sigma2_{t+k}, sigma2_{t+k} = long_run + (alpha+beta)^(k-1) * (sigma2_{t+1} - long_run)."""
    sigma2_next = garch_var_by_ts.get(cutpoint_ts)
    if sigma2_next is None or garch_params.get("omega") is None:
        return None
    omega, alpha, beta = garch_params["omega"], garch_params["alpha"], garch_params["beta"]
    persistence = alpha + beta
    long_run_var = (omega / max(1 - persistence, 1e-6)) / 10000  # varyans olcegine (yuzdeden) geri donustur

    total = 0.0
    sigma2_k = sigma2_next
    for k in range(horizon_hours):
        total += sigma2_k
        sigma2_k = long_run_var + persistence * (sigma2_k - long_run_var)
    return log_rv(total)


# ---------------------------------------------------------------------------
# 5) TimesFM-direct: log(RV) serisini DOGRUDAN tahmin ettir (fiyat degil)
# ---------------------------------------------------------------------------
# bkz. scripts/vol_predict_cache.py - bu, build_hourly_log_rv_series'i TimesFM'e
# baglam olarak verip forecast_horizon kadar ileri tahmin alir. Onbellekten okuma
# burada, uretme predict_cache benzeri ayri bir script'te (agir islem, TimesFM
# cagirir).

def forecast_timesfm_direct(cached_pred: dict, horizon_hours: int):
    """cached_pred: scripts/vol_predict_cache.py'nin uretecegi {"points": [...]} -
    log(RV) serisinin kendisi tahmin edildigi icin points[horizon_hours-1] DOGRUDAN
    log(RV) tahminidir (ek donusum gerekmez)."""
    if cached_pred is None:
        return None
    points = cached_pred.get("points")
    if not points or len(points) < horizon_hours:
        return None
    return float(points[horizon_hours - 1])


# ---------------------------------------------------------------------------
# 6) TimesFM-hibrit: fiyat kantillerinden ima edilen oynaklik
# ---------------------------------------------------------------------------

Z_80 = 1.2815515655446004  # standart normalin %90 kantili (q90-q10 araligi ~%80 kapsar)


def forecast_timesfm_hybrid(price_pred: dict, entry_price: float, horizon_hours: int):
    """price_pred: mevcut YON calismasinin HAM tahmin onbellegi (scripts/predict_cache.py,
    context=300) - {"points":[...], "quantiles":[[ozel,q10..q90],...]}. q10/q90 fiyat
    kantillerinden ima edilen log-getiri std'sini cikarir (normal dagilim varsayimiyla),
    H saatlik ima edilen RV = std_h^2."""
    if price_pred is None or entry_price is None or entry_price <= 0:
        return None
    quantiles = price_pred.get("quantiles")
    if not quantiles or len(quantiles) < horizon_hours:
        return None
    row = quantiles[horizon_hours - 1]
    if row is None or len(row) < 10:
        return None
    q10, q90 = row[1], row[9]
    if q10 is None or q90 is None or q10 <= 0 or q90 <= 0:
        return None
    implied_std = (np.log(q90) - np.log(q10)) / (2 * Z_80)
    implied_var_h = implied_std ** 2  # zaten H-saatlik ufuktaki kantillerden, ek olcekleme gerekmez
    return log_rv(implied_var_h)
