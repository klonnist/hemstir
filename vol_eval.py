"""Oynaklik tahmin degerlendirme metrikleri - literaturun standartlari.

Tum fonksiyonlar RV (varyans) OLCEGINDE calisir (girdi olarak actual_rv/predicted_rv
alir, log DEGIL) - cagiran taraf log(RV) tahminini once exp() ile RV'ye cevirmeli
(QLIKE/MSE varyans olceginde tanimlidir, log-olcekte degil).
"""
import numpy as np
from scipy import stats


def qlike(actual_rv, predicted_rv) -> float:
    """QLIKE (Quasi-Likelihood) kayip fonksiyonu - oynaklik tahmini literaturunde
    STANDART ana metrik (Patton 2011): asimetrik cezalandirmadan ari, tahmin
    olcegine gore normalize (MSE'den farkli olarak buyuk RV donemlerinde tek bir
    aykiri deger tum metrigi domine etmez). Daha DUSUK = daha iyi.
    L(actual, pred) = actual/pred - log(actual/pred) - 1."""
    actual = np.asarray(actual_rv, dtype=float)
    pred = np.asarray(predicted_rv, dtype=float)
    mask = (actual > 0) & (pred > 0) & np.isfinite(actual) & np.isfinite(pred)
    if not mask.any():
        return None
    ratio = actual[mask] / pred[mask]
    losses = ratio - np.log(ratio) - 1
    return float(np.mean(losses))


def qlike_series(actual_rv, predicted_rv) -> np.ndarray:
    """qlike ile ayni ama ORTALAMA almadan, HER GOZLEM icin kayip degeri doner -
    Diebold-Mariano testi bu seriye ihtiyac duyar."""
    actual = np.asarray(actual_rv, dtype=float)
    pred = np.asarray(predicted_rv, dtype=float)
    out = np.full(len(actual), np.nan)
    mask = (actual > 0) & (pred > 0) & np.isfinite(actual) & np.isfinite(pred)
    ratio = actual[mask] / pred[mask]
    out[mask] = ratio - np.log(ratio) - 1
    return out


def mse(actual_rv, predicted_rv) -> float:
    """Ortalama kare hata (varyans olceginde) - QLIKE'a ek, daha bilindik metrik."""
    actual = np.asarray(actual_rv, dtype=float)
    pred = np.asarray(predicted_rv, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not mask.any():
        return None
    return float(np.mean((actual[mask] - pred[mask]) ** 2))


def r_squared_log(actual_log_rv, predicted_log_rv) -> float:
    """log(RV) olceginde R^2 (aciklama gucu) - klasik korelasyon-karesi tanimi."""
    a = np.asarray(actual_log_rv, dtype=float)
    p = np.asarray(predicted_log_rv, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    if mask.sum() < 2:
        return None
    a, p = a[mask], p[mask]
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    if ss_tot <= 0:
        return None
    return float(1 - ss_res / ss_tot)


def mincer_zarnowitz(actual_log_rv, predicted_log_rv) -> dict:
    """Mincer-Zarnowitz regresyonu: gercek = a + b*tahmin. Iyi kalibre bir tahminci
    icin a~0, b~1 beklenir (F-testiyle H0: a=0,b=1 reddedilmez). scipy ile basit
    OLS (numpy polyfit) + a=0,b=1 ortak hipotezi icin Wald-tipi bir test uygular."""
    import statsmodels.api as sm

    a = np.asarray(actual_log_rv, dtype=float)
    p = np.asarray(predicted_log_rv, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    if len(a) < 10:
        return {"intercept": None, "slope": None, "r2": None, "n": len(a),
                 "joint_test_pvalue": None, "well_calibrated": None}

    X = sm.add_constant(p)
    model = sm.OLS(a, X).fit()
    intercept, slope = model.params[0], model.params[1]
    # H0: intercept=0 VE slope=1 - Wald F-testi (R*beta = q kisitlamasi).
    # NOT: statsmodels'in wald_test'i sadece R*beta=0 sifir-kisitlamasini
    # destekler; R*beta=q icin F-istatistigini elle hesaplamak gerekir.
    r_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    q = np.array([0.0, 1.0])
    try:
        params = np.asarray(model.params)
        cov = np.asarray(model.cov_params())
        diff = r_matrix @ params - q
        r_cov_rt = r_matrix @ cov @ r_matrix.T
        wald_stat = float(diff @ np.linalg.inv(r_cov_rt) @ diff)
        df_num = r_matrix.shape[0]
        df_denom = model.df_resid
        f_stat = wald_stat / df_num
        pvalue = float(1 - stats.f.cdf(f_stat, df_num, df_denom))
    except Exception:
        pvalue = None

    return {
        "intercept": round(float(intercept), 4),
        "slope": round(float(slope), 4),
        "r2": round(float(model.rsquared), 4),
        "n": len(a),
        "joint_test_pvalue": round(pvalue, 4) if pvalue is not None else None,
        # pvalue > 0.05 -> H0 (a=0,b=1) REDDEDILEMEZ -> "iyi kalibre" (sapma yok)
        "well_calibrated": (pvalue > 0.05) if pvalue is not None else None,
    }


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> dict:
    """Diebold-Mariano testi: yontem A'nin kayip serisi (orn. qlike_series), yontem
    B'ninkinden (referans - genelde HAR-RV) ISTATISTIKSEL OLARAK FARKLI mi. d_t =
    loss_a[t]-loss_b[t] serisinin ortalamasi sifirdan farkli mi (Newey-West HAC
    standart hatasiyla, otokorelasyona karsi robust)."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return {"dm_stat": None, "pvalue": None, "n": n, "better": None}

    mean_d = np.mean(d)
    # Newey-West HAC varyans (lag = n^(1/3) kurali)
    max_lag = max(1, int(n ** (1 / 3)))
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, max_lag + 1):
        cov = np.cov(d[lag:], d[:-lag])[0, 1] if n > lag else 0.0
        weight = 1 - lag / (max_lag + 1)
        var_d += 2 * weight * cov
    var_d = max(var_d, 1e-12) / n

    dm_stat = mean_d / np.sqrt(var_d)
    pvalue = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))
    return {
        "dm_stat": round(float(dm_stat), 4),
        "pvalue": round(pvalue, 4),
        "n": n,
        "significant": pvalue < 0.05,
        # negatif dm_stat -> A'nin kaybi B'den DUSUK (A DAHA IYI)
        "a_better": bool(dm_stat < 0) if pvalue < 0.05 else None,
    }


def pinball_loss(actual, quantile_pred, tau: float) -> float:
    """Pinball (quantile) kaybi - tek bir kantil tahmininin kalitesini olcer.
    tau: hedeflenen kantil (orn. 0.1, 0.5, 0.9)."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(quantile_pred, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not mask.any():
        return None
    diff = actual[mask] - pred[mask]
    loss = np.where(diff >= 0, tau * diff, (tau - 1) * diff)
    return float(np.mean(loss))


def interval_coverage(actual, lower, upper) -> float:
    """Gercek deger [lower, upper] araliginda ne siklikta kaldi (%) - kalibrasyon
    icin. %50/%80/%95 seviyelerinde cagrilir."""
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
    if not mask.any():
        return None
    inside = (actual[mask] >= lower[mask]) & (actual[mask] <= upper[mask])
    return round(100 * float(np.mean(inside)), 2)
