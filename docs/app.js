let forecastData = null;
let evaluationData = null;
let backtestData = null; // null = henuz cekilmedi (buyuk dosya, sadece secilince cekilir)
let backtestAvailable = null; // null = bilinmiyor, true/false HEAD kontrolunden sonra
let activeCoin = null;
let chart = null;
let historyChart = null;
let equityChart = null;
const historyState = { source: "archive", mode: "single_position", n: "20" };

function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("tr-TR", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function fmtPrice(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const digits = n >= 100 ? 2 : n >= 1 ? 4 : 6;
  return Number(n).toLocaleString("tr-TR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

// ---- Tema ----
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const btn = document.getElementById("theme-toggle");
  if (btn) btn.textContent = theme === "light" ? "☀️" : "🌙";
  try { localStorage.setItem("theme", theme); } catch (e) { /* gizli sekme vb. olabilir */ }
}

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch (e) { /* yoksay */ }
  const preferred = saved || (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  applyTheme(preferred);
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    applyTheme(current === "light" ? "dark" : "light");
    if (chart) renderChart(activeCoin); // eksen/legend renkleri temaya gore yeniden cizilsin
  });
}

// ---- Veri yukleme ----
async function loadForecasts() {
  const res = await fetch("forecasts.json?_=" + Date.now());
  if (!res.ok) throw new Error(`forecasts.json: ${res.status}`);
  return res.json();
}

async function loadJsonOptional(path) {
  try {
    const res = await fetch(path + "?_=" + Date.now());
    if (!res.ok) return null;
    return await res.json();
  } catch (e) {
    return null;
  }
}

async function checkFileExists(path) {
  try {
    const res = await fetch(path, { method: "HEAD" });
    return res.ok;
  } catch (e) {
    return false;
  }
}

function buildTabs(coins, retiredSymbols) {
  const tabs = document.getElementById("tabs");
  const active = Object.keys(coins).map(symbol =>
    `<button class="tab-btn" data-coin="${symbol}">${symbol}</button>`
  );
  const retired = (retiredSymbols || []).map(symbol =>
    `<button class="tab-btn tab-btn-retired" data-coin="${symbol}" title="Artık izlenmiyor, sadece geçmiş veri">${symbol} 🕓</button>`
  );
  tabs.innerHTML = active.join("") + retired.join("");
  tabs.addEventListener("click", (e) => {
    const btn = e.target.closest(".tab-btn");
    if (btn && btn.dataset.coin) selectCoin(btn.dataset.coin);
  });
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function selectCoin(symbol) {
  activeCoin = symbol;
  document.querySelectorAll(".tab-btn[data-coin]").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.coin === symbol);
  });

  if (!forecastData.coins[symbol]) {
    // Artik takip edilmeyen (retired) coin: canli tahmin/sinyal/grafik yok,
    // sadece Gecmis Performans bolumu (arsiv/backtest) gosterilir.
    document.getElementById("signal-panel").innerHTML =
      `<div class="empty">${symbol} artık aktif olarak takip edilmiyor (kapsam 5 major coin'e daraltıldı). Aşağıda geçmiş arşiv/backtest verisi hâlâ mevcut.</div>`;
    if (chart) { chart.destroy(); chart = null; }
    document.getElementById("chart-title").textContent = `${symbol} — artık takip edilmiyor`;
    document.getElementById("meta").innerHTML = "";
    renderHistoryPanel(symbol);
    return;
  }

  renderSignal(symbol);
  renderChart(symbol);
  renderMeta(symbol);
  renderHistoryPanel(symbol);
}

const STRENGTH_LABELS = { guclu: "Güçlü", orta: "Orta", zayif: "Zayıf" };

function renderSignal(symbol) {
  const panel = document.getElementById("signal-panel");
  const sig = forecastData.coins[symbol].signal;
  if (!sig) {
    panel.innerHTML = `<div class="empty">${symbol} için sinyal hesaplanamadı (yetersiz geçmiş veri).</div>`;
    return;
  }
  const isBuy = sig.side === "BUY";
  const sideLabel = isBuy ? "AL · BUY" : "SAT · SELL";
  const moveSign = sig.expected_move_pct >= 0 ? "+" : "";

  panel.innerHTML = `
    <div class="signal-head">
      <span class="signal-badge ${isBuy ? "side-buy-badge" : "side-sell-badge"}">${sideLabel}</span>
      <span class="signal-tag">Sinyal gücü: ${STRENGTH_LABELS[sig.strength] || sig.strength}</span>
      <span class="signal-tag">Risk/Ödül 1:${sig.risk_reward}</span>
    </div>
    <div class="signal-grid">
      <div class="signal-item"><div class="label">Giriş</div><div class="value">${fmtPrice(sig.entry_price)}</div></div>
      <div class="signal-item"><div class="label">Take Profit</div><div class="value pos">${fmtPrice(sig.take_profit)}</div></div>
      <div class="signal-item"><div class="label">Stop Loss</div><div class="value neg">${fmtPrice(sig.stop_loss)}</div></div>
      <div class="signal-item"><div class="label">ATR (14)</div><div class="value">${fmtPrice(sig.atr)}</div></div>
      <div class="signal-item"><div class="label">Beklenen Hareket</div><div class="value ${sig.expected_move_pct >= 0 ? "pos" : "neg"}">${moveSign}${sig.expected_move_pct}%</div></div>
    </div>
    <p class="signal-disclaimer">
      Kural tabanlı, otomatik üretilir: yön TimesFM'in ${forecastData.horizon_hours} saat sonrası için beklediği
      fiyatın mevcut fiyata göre yukarı/aşağı olmasından; Take Profit / Stop Loss ise ATR(14)'ün sabit
      katlarından (SL = 1.5×ATR, TP = 2.5×ATR) hesaplanır — TimesFM'in güven aralığı risk yönetimi için
      kullanılmaz. Kanıtlanmış bir strateji ya da yatırım tavsiyesi değildir.
    </p>
  `;
}

function renderMeta(symbol) {
  const meta = document.getElementById("meta");
  const coin = forecastData.coins[symbol];
  const lastClose = coin.history.length ? coin.history[coin.history.length - 1].close : null;
  const firstForecast = coin.forecast.length ? coin.forecast[0] : null;
  const lastForecast = coin.forecast.length ? coin.forecast[coin.forecast.length - 1] : null;
  meta.innerHTML = `
    ${coin.inst_id} · son gerçek kapanış: <strong>${fmtPrice(lastClose)}</strong> USDT
    ${firstForecast ? ` · ${fmtTime(firstForecast.ts)} tahmini: <strong>${fmtPrice(firstForecast.value)}</strong>` : ""}
    ${lastForecast ? ` · ${fmtTime(lastForecast.ts)} tahmini: <strong>${fmtPrice(lastForecast.value)}</strong>` : ""}
    · model: ${forecastData.model} · bağlam: ${forecastData.context_hours} saat · ufuk: ${forecastData.horizon_hours} saat
  `;
}

function renderChart(symbol) {
  const coin = forecastData.coins[symbol];
  document.getElementById("chart-title").textContent = `${symbol} (${coin.inst_id}) — Saatlik Kapanış ve TimesFM Tahmini`;

  const historyLen = coin.history.length;
  const labels = [
    ...coin.history.map(p => fmtTime(p.ts)),
    ...coin.forecast.map(p => fmtTime(p.ts)),
  ];

  const historySeries = [...coin.history.map(p => p.close), ...coin.forecast.map(() => null)];

  const forecastSeries = new Array(historyLen).fill(null);
  if (historyLen > 0) forecastSeries[historyLen - 1] = coin.history[historyLen - 1].close; // koptugu yerde birlestir
  forecastSeries.push(...coin.forecast.map(p => p.value));

  const hasBand = coin.forecast.length > 0 && coin.forecast.every(p => p.lower !== null && p.upper !== null);
  const upperSeries = new Array(historyLen).fill(null).concat(hasBand ? coin.forecast.map(p => p.upper) : []);
  const lowerSeries = new Array(historyLen).fill(null).concat(hasBand ? coin.forecast.map(p => p.lower) : []);

  const accent = cssVar("--accent") || "#5b8cff";
  const green = cssVar("--green") || "#3ddc84";
  const band = cssVar("--band") || "rgba(91,140,255,0.18)";
  const muted = cssVar("--muted") || "#8b93a7";
  const gridColor = cssVar("--card-border") || "#232838";

  const datasets = [];
  if (hasBand) {
    datasets.push({
      label: "Üst sınır",
      data: upperSeries,
      borderWidth: 0,
      pointRadius: 0,
      fill: false,
      backgroundColor: band,
      spanGaps: false,
    });
    datasets.push({
      label: "Alt sınır",
      data: lowerSeries,
      borderWidth: 0,
      pointRadius: 0,
      fill: "-1",
      backgroundColor: band,
      spanGaps: false,
    });
  }
  datasets.push({
    label: "Gerçek",
    data: historySeries,
    borderColor: accent,
    backgroundColor: accent,
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.15,
    spanGaps: false,
  });
  datasets.push({
    label: "Tahmin",
    data: forecastSeries,
    borderColor: green,
    backgroundColor: green,
    borderWidth: 2,
    borderDash: [6, 4],
    pointRadius: 0,
    tension: 0.15,
    spanGaps: false,
  });

  const config = {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          filter: (item) => {
            const label = item.dataset.label;
            if (label === "Gerçek" || label === "Tahmin") return true;
            if (label === "Üst sınır" || label === "Alt sınır") {
              return item.parsed.y !== null && item.parsed.y !== undefined;
            }
            return false;
          },
          itemSort: (a, b) => {
            const order = { "Gerçek": 0, "Tahmin": 1, "Üst sınır": 2, "Alt sınır": 3 };
            return (order[a.dataset.label] ?? 9) - (order[b.dataset.label] ?? 9);
          },
          callbacks: { label: (item) => `${item.dataset.label}: ${fmtPrice(item.parsed.y)} USDT` },
        },
      },
      scales: {
        x: {
          ticks: { color: muted, maxTicksLimit: 10, autoSkip: true },
          grid: { color: gridColor },
        },
        y: {
          ticks: { color: muted, callback: (v) => fmtPrice(v) },
          grid: { color: gridColor },
        },
      },
    },
  };

  if (chart) chart.destroy();
  chart = new Chart(document.getElementById("chart").getContext("2d"), config);
}

// ---------------------------------------------------------------------------
// Gecmis Performans: arsivlenmis gercek tahminler (evaluationData) ve/veya
// walk-forward backtest (backtestData) sonuclarini gosterir. Ikisi de ayni
// sekle sahip JSON'lar (bkz. scripts/evaluate_archive.py, scripts/backtest.py).
// ---------------------------------------------------------------------------

const STRATEGY_LABELS = { model: "TimesFM (model)", always_buy: "Her zaman AL", momentum: "Momentum takip", random: "Rastgele yon" };
const STRATEGY_COLOR_VARS = { model: "--accent", always_buy: "--muted", momentum: "--green", random: "--red" };

function fmtPct(v, withSign) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const sign = withSign && v > 0 ? "+" : "";
  return sign + v.toFixed(2) + "%";
}

function getActiveDataset(symbol) {
  const src = historyState.source === "backtest" ? backtestData : evaluationData;
  if (!src || !src.coins || !src.coins[symbol]) return null;
  return src.coins[symbol];
}

function pickRecent(list, n) {
  if (!list) return [];
  if (n === "all") return list;
  const count = parseInt(n, 10) || 20;
  return list.slice(-count);
}

function renderHistoryPanel(symbol) {
  const panel = document.getElementById("history-panel");
  const hasArchive = !!(evaluationData && evaluationData.coins);
  const hasBacktest = backtestAvailable === true;

  if (!hasArchive && !hasBacktest) {
    panel.innerHTML = `
      <h2>Geçmiş Performans</h2>
      <div class="empty">Henüz değerlendirme verisi yok — arşiv birkaç çalıştırma biriktikten sonra burada görünecek.</div>
    `;
    document.getElementById("cross-coin-panel").innerHTML = "";
    return;
  }

  if (historyState.source === "backtest" && !hasBacktest) historyState.source = "archive";
  if (historyState.source === "archive" && !hasArchive) historyState.source = "backtest";

  panel.innerHTML = `
    <div class="history-head">
      <h2>Geçmiş Performans</h2>
      <div class="history-controls">
        <select id="history-source">
          <option value="archive" ${!hasArchive ? "disabled" : ""}>Canlı arşiv (gerçek tahminler)</option>
          <option value="backtest" ${!hasBacktest ? "disabled" : ""}>Backtest (geriye dönük simülasyon)</option>
        </select>
        <select id="history-mode">
          <option value="single_position">Coin başına tek pozisyon</option>
          <option value="independent">Her sinyal bağımsız işlem</option>
        </select>
        <select id="history-n">
          <option value="10">Son 10 tahmin</option>
          <option value="20">Son 20 tahmin</option>
          <option value="50">Son 50 tahmin</option>
          <option value="all">Tümü</option>
        </select>
      </div>
    </div>
    <div id="history-body"></div>
  `;

  const sourceSel = document.getElementById("history-source");
  const modeSel = document.getElementById("history-mode");
  const nSel = document.getElementById("history-n");
  sourceSel.value = historyState.source;
  modeSel.value = historyState.mode;
  nSel.value = historyState.n;

  sourceSel.addEventListener("change", () => { historyState.source = sourceSel.value; renderHistoryBody(symbol); });
  modeSel.addEventListener("change", () => { historyState.mode = modeSel.value; renderHistoryBody(symbol); });
  nSel.addEventListener("change", () => { historyState.n = nSel.value; renderHistoryBody(symbol); });

  renderHistoryBody(symbol);
}

async function renderHistoryBody(symbol) {
  const body = document.getElementById("history-body");
  const sourceLabel = historyState.source === "backtest" ? "backtest" : "canlı arşiv";

  if (historyState.source === "backtest" && !backtestData) {
    body.innerHTML = `<div class="empty">Backtest verisi yükleniyor (büyük dosya olabilir, biraz sürebilir)…</div>`;
    document.getElementById("cross-coin-panel").innerHTML = "";
    backtestData = await loadJsonOptional("backtest.json");
    if (activeCoin !== symbol || historyState.source !== "backtest") return; // bu sirada baska bir sekme/kaynak secildi
    if (!backtestData) {
      backtestAvailable = false;
      historyState.source = "archive";
      renderHistoryPanel(symbol);
      return;
    }
  }

  const dataset = getActiveDataset(symbol);
  if (!dataset) {
    body.innerHTML = `<div class="empty">${symbol} için ${sourceLabel} verisi yok.</div>`;
    document.getElementById("cross-coin-panel").innerHTML = "";
    return;
  }

  const sampleSize = dataset.sample_size || 0;
  const warning = sampleSize < 30
    ? `<div class="sample-warning">⚠️ Örnek sayısı az (${sampleSize} işlem) — istatistiksel olarak güvenilir değil, yorumlarken temkinli olun.</div>`
    : "";

  body.innerHTML = `
    ${warning}
    <div class="chart-wrap history-chart-wrap"><canvas id="history-chart"></canvas></div>
    <div class="chart-wrap equity-chart-wrap"><canvas id="equity-chart"></canvas></div>
    <div id="history-summary"></div>
  `;

  renderHistoryChart(dataset);
  renderEquityChart(dataset);
  renderHistorySummary(dataset);
  renderCrossCoinTable();
}

function checkpointsToPoints(entryTs, entryPrice, checkpoints) {
  const pts = [{ x: Date.parse(entryTs), y: entryPrice }];
  ["6", "12", "24", "36", "48"].forEach(h => {
    const cp = checkpoints && checkpoints[h];
    if (cp && cp.value !== null && cp.value !== undefined) pts.push({ x: Date.parse(cp.ts), y: cp.value });
  });
  return pts;
}

function renderHistoryChart(dataset) {
  const canvas = document.getElementById("history-chart");
  if (historyChart) { historyChart.destroy(); historyChart = null; }
  const preds = pickRecent(dataset.predictions || [], historyState.n);
  if (!preds.length) {
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    return;
  }

  const accent = cssVar("--accent") || "#5b8cff";
  const green = cssVar("--green") || "#3ddc84";
  const red = cssVar("--red") || "#ff5c5c";
  const muted = cssVar("--muted") || "#8b93a7";
  const gridColor = cssVar("--card-border") || "#232838";

  const baseline = preds
    .map(p => ({ x: Date.parse(p.entry_ts), y: p.entry_price }))
    .sort((a, b) => a.x - b.x);

  const datasets = [{
    label: "Gerçek fiyat (tahmin anları)",
    data: baseline,
    borderColor: accent,
    backgroundColor: accent,
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.1,
    order: 10,
    meta: { kind: "baseline" },
  }];

  preds.forEach(p => {
    const color = p.direction_correct === true ? green : p.direction_correct === false ? red : muted;
    datasets.push({
      label: "Geçmiş tahmin",
      data: checkpointsToPoints(p.entry_ts, p.entry_price, p.checkpoints),
      borderColor: color,
      backgroundColor: color,
      borderWidth: 1.5,
      borderDash: [4, 3],
      pointRadius: 0,
      tension: 0,
      order: 5,
      meta: { kind: "prediction", side: p.side, strength: p.strength, directionCorrect: p.direction_correct },
    });
    datasets.push({
      label: "Giriş",
      data: [{ x: Date.parse(p.entry_ts), y: p.entry_price }],
      showLine: false,
      pointStyle: "triangle",
      rotation: p.side === "SELL" ? 180 : 0,
      pointRadius: 5,
      pointHoverRadius: 7,
      pointBackgroundColor: p.side === "SELL" ? red : green,
      pointBorderColor: p.side === "SELL" ? red : green,
      order: 1,
      meta: { kind: "entry", side: p.side, price: p.entry_price },
    });
  });

  const modelEntry = dataset.strategies && dataset.strategies.model && dataset.strategies.model[historyState.mode];
  const modelTrades = (modelEntry && modelEntry.trades) || [];
  const shownEntryTs = new Set(preds.map(p => p.entry_ts));
  modelTrades.filter(t => shownEntryTs.has(t.entry_ts) && t.exit_ts && t.exit_price !== null).forEach(t => {
    const isWin = t.exit_reason === "TP";
    const isLoss = t.exit_reason === "SL" || t.exit_reason === "LIQUIDATION";
    const color = isWin ? green : isLoss ? red : muted;
    datasets.push({
      label: "Çıkış",
      data: [{ x: Date.parse(t.exit_ts), y: t.exit_price }],
      showLine: false,
      pointStyle: isWin ? "circle" : isLoss ? "crossRot" : "rect",
      pointRadius: 5,
      pointHoverRadius: 7,
      pointBackgroundColor: color,
      pointBorderColor: color,
      order: 1,
      meta: { kind: "exit", exitReason: t.exit_reason, pnlPct: t.pnl_pct },
    });
  });

  historyChart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "point", intersect: true },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => items.length ? fmtTime(new Date(items[0].parsed.x).toISOString()) : "",
            label: (item) => {
              const m = item.dataset.meta || {};
              if (m.kind === "entry") return `${m.side === "SELL" ? "SAT" : "AL"} girişi: ${fmtPrice(m.price)}`;
              if (m.kind === "exit") {
                const pnl = m.pnlPct !== null && m.pnlPct !== undefined ? fmtPct(m.pnlPct, true) : "—";
                return `Çıkış (${m.exitReason}): ${fmtPrice(item.parsed.y)} · K/Z: ${pnl}`;
              }
              if (m.kind === "prediction") {
                return `Tahmin (${m.side || "?"}, ${STRENGTH_LABELS[m.strength] || m.strength || "—"}): ${fmtPrice(item.parsed.y)}`;
              }
              return `${fmtPrice(item.parsed.y)} USDT`;
            },
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          ticks: { color: muted, maxTicksLimit: 8, callback: (v) => fmtTime(new Date(v).toISOString()) },
          grid: { color: gridColor },
        },
        y: {
          ticks: { color: muted, callback: (v) => fmtPrice(v) },
          grid: { color: gridColor },
        },
      },
    },
  });
}

function buildEquityCurve(trades) {
  const sorted = [...trades].sort((a, b) => Date.parse(a.entry_ts) - Date.parse(b.entry_ts));
  let cum = 0;
  return sorted.map(t => {
    cum += (t.pnl_pct || 0);
    return { x: Date.parse(t.entry_ts), y: Math.round(cum * 10000) / 10000 };
  });
}

function renderEquityChart(dataset) {
  const canvas = document.getElementById("equity-chart");
  if (equityChart) { equityChart.destroy(); equityChart = null; }

  const muted = cssVar("--muted") || "#8b93a7";
  const gridColor = cssVar("--card-border") || "#232838";
  const datasets = [];
  Object.keys(dataset.strategies || {}).forEach(strat => {
    const entry = dataset.strategies[strat][historyState.mode];
    if (!entry || !entry.trades || !entry.trades.length) return;
    const color = cssVar(STRATEGY_COLOR_VARS[strat] || "--muted") || "#8b93a7";
    datasets.push({
      label: STRATEGY_LABELS[strat] || strat,
      data: buildEquityCurve(entry.trades),
      borderColor: color,
      backgroundColor: color,
      borderWidth: strat === "model" ? 2.5 : 1.5,
      pointRadius: 0,
      tension: 0.1,
    });
  });

  if (!datasets.length) {
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    return;
  }

  equityChart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "nearest", intersect: false },
      plugins: {
        legend: { display: true, position: "top", labels: { color: muted, boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            title: (items) => items.length ? fmtTime(new Date(items[0].parsed.x).toISOString()) : "",
            label: (item) => `${item.dataset.label}: ${fmtPct(item.parsed.y, true)}`,
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          ticks: { color: muted, maxTicksLimit: 8, callback: (v) => fmtTime(new Date(v).toISOString()) },
          grid: { color: gridColor },
        },
        y: {
          ticks: { color: muted, callback: (v) => Number(v).toFixed(2) + "%" },
          grid: { color: gridColor },
        },
      },
    },
  });
}

function exitDistLabel(dist) {
  if (!dist || !Object.keys(dist).length) return "—";
  return Object.entries(dist).map(([k, v]) => `${k}: ${v}`).join(" · ");
}

function statItem(label, value, cls) {
  return `<div class="signal-item"><div class="label">${label}</div><div class="value ${cls || ""}">${value}</div></div>`;
}

function renderAggCards(agg) {
  if (!agg || !agg.count) {
    return `<div class="empty">Yeterli işlem yok.</div>`;
  }
  return `<div class="signal-grid">
    ${statItem("İşlem Sayısı", agg.count)}
    ${statItem("Yön Doğruluğu", agg.direction_accuracy_pct != null ? agg.direction_accuracy_pct.toFixed(1) + "%" : "—")}
    ${statItem("Kazanma Oranı", agg.win_rate_pct != null ? agg.win_rate_pct.toFixed(1) + "%" : "—")}
    ${statItem("Dağılım (TP·SL·Süre)", exitDistLabel(agg.exit_distribution))}
    ${statItem("Ort. Kazanç", fmtPct(agg.avg_win_pct, true), "pos")}
    ${statItem("Ort. Kayıp", fmtPct(agg.avg_loss_pct, true), "neg")}
    ${statItem("Toplam Getiri", fmtPct(agg.total_return_pct, true), (agg.total_return_pct || 0) >= 0 ? "pos" : "neg")}
    ${statItem("Maks. Düşüş", fmtPct(agg.max_drawdown_pct), "neg")}
    ${statItem("Profit Factor", agg.profit_factor != null ? agg.profit_factor.toFixed(2) : "—")}
  </div>`;
}

function renderHistorySummary(dataset) {
  const container = document.getElementById("history-summary");
  const modelEntry = dataset.strategies && dataset.strategies.model && dataset.strategies.model[historyState.mode];

  let html = `<h3 class="history-subhead">TimesFM Sinyali — Genel</h3>${renderAggCards(modelEntry)}`;

  if (modelEntry && modelEntry.by_strength) {
    html += `<h3 class="history-subhead">Sinyal Gücüne Göre</h3><div class="breakdown-grid">`;
    ["guclu", "orta", "zayif"].forEach(s => {
      html += `<div><h4>${STRENGTH_LABELS[s] || s}</h4>${renderAggCards(modelEntry.by_strength[s])}</div>`;
    });
    html += `</div>`;
  }
  if (modelEntry && modelEntry.by_side) {
    html += `<h3 class="history-subhead">Yöne Göre</h3><div class="breakdown-grid">`;
    ["BUY", "SELL"].forEach(s => {
      html += `<div><h4>${s === "BUY" ? "AL" : "SAT"}</h4>${renderAggCards(modelEntry.by_side[s])}</div>`;
    });
    html += `</div>`;
  }

  const compStrategies = Object.keys(dataset.strategies || {}).filter(s => s !== "model");
  if (compStrategies.length) {
    html += `<h3 class="history-subhead">Karşılaştırma Stratejileri</h3><div class="breakdown-grid">`;
    compStrategies.forEach(strat => {
      const entry = dataset.strategies[strat][historyState.mode];
      html += `<div><h4>${STRATEGY_LABELS[strat] || strat}</h4>${renderAggCards(entry)}</div>`;
    });
    html += `</div>`;
  }

  container.innerHTML = html;
}

function renderCrossCoinTable() {
  const container = document.getElementById("cross-coin-panel");
  const src = historyState.source === "backtest" ? backtestData : evaluationData;
  if (!src || !src.coins || !Object.keys(src.coins).length) {
    container.innerHTML = "";
    return;
  }

  const rows = Object.entries(src.coins).map(([symbol, c]) => {
    const modelEntry = c.strategies && c.strategies.model && c.strategies.model[historyState.mode];
    const dirAcc48 = c.direction_accuracy_pct && c.direction_accuracy_pct["48"];
    return { symbol, agg: modelEntry, sample: c.sample_size || 0, dirAcc48 };
  }).sort((a, b) => {
    const av = a.agg && a.agg.total_return_pct !== null && a.agg.total_return_pct !== undefined ? a.agg.total_return_pct : -Infinity;
    const bv = b.agg && b.agg.total_return_pct !== null && b.agg.total_return_pct !== undefined ? b.agg.total_return_pct : -Infinity;
    return bv - av;
  });

  const sourceLabel = historyState.source === "backtest" ? "Backtest" : "Canlı Arşiv";
  const modeLabel = historyState.mode === "independent" ? "her sinyal bağımsız işlem" : "coin başına tek pozisyon";
  const lowSample = rows.some(r => r.sample > 0 && r.sample < 30);

  container.innerHTML = `
    <h2>Tüm Coinler — ${sourceLabel} (${modeLabel})</h2>
    <div style="overflow-x:auto;">
      <table>
        <thead><tr>
          <th>Coin</th><th>İşlem</th><th>Yön Doğ. (48s)</th><th>Kazanma %</th>
          <th>Toplam Getiri</th><th>Maks. Düşüş</th><th>Profit Factor</th>
        </tr></thead>
        <tbody>
          ${rows.map(r => `<tr>
            <td>${r.symbol}</td>
            <td>${r.sample}</td>
            <td>${r.dirAcc48 != null ? r.dirAcc48.toFixed(1) + "%" : "—"}</td>
            <td>${r.agg && r.agg.win_rate_pct != null ? r.agg.win_rate_pct.toFixed(1) + "%" : "—"}</td>
            <td class="${r.agg && (r.agg.total_return_pct || 0) >= 0 ? "pos" : "neg"}">${r.agg ? fmtPct(r.agg.total_return_pct, true) : "—"}</td>
            <td class="neg">${r.agg ? fmtPct(r.agg.max_drawdown_pct) : "—"}</td>
            <td>${r.agg && r.agg.profit_factor != null ? r.agg.profit_factor.toFixed(2) : "—"}</td>
          </tr>`).join("")}
        </tbody>
      </table>
    </div>
    ${lowSample ? `<p class="sample-warning">⚠️ Bazı coinlerde örnek sayısı 30'un altında — istatistiksel olarak güvenilir değil.</p>` : ""}
  `;
}

async function init() {
  initTheme();
  try {
    forecastData = await loadForecasts();
  } catch (e) {
    document.getElementById("tabs").innerHTML = "";
    document.getElementById("meta").innerHTML =
      `<div class="empty">Henüz tahmin verisi yok — GitHub Actions'ın ilk çalışması bekleniyor.</div>`;
    return;
  }

  const coins = forecastData.coins || {};
  const symbols = Object.keys(coins);
  if (!symbols.length) {
    document.getElementById("meta").innerHTML = `<div class="empty">Tahmin verisi boş.</div>`;
    return;
  }

  document.getElementById("updated").textContent = "Son güncelleme: " + fmtTime(forecastData.generated_at);

  // evaluation.json kucuk, hemen cekilir; backtest.json onlarca MB olabilir - sadece
  // varligini (HEAD) kontrol ederiz, kullanici "Backtest"i secince tembel indirilir.
  const [evalRes, backtestExists] = await Promise.all([
    loadJsonOptional("evaluation.json"),
    checkFileExists("backtest.json"),
  ]);
  evaluationData = evalRes;
  backtestAvailable = backtestExists;
  if (!evaluationData && backtestAvailable) historyState.source = "backtest";

  const retiredSymbols = evaluationData && evaluationData.coins
    ? Object.keys(evaluationData.coins).filter(s => !coins[s])
    : [];
  buildTabs(coins, retiredSymbols);

  selectCoin(symbols[0]);
}

init();
