let forecastData = null;
let activeCoin = null;
let chart = null;

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

function buildTabs(coins) {
  const tabs = document.getElementById("tabs");
  tabs.innerHTML = Object.keys(coins).map(symbol =>
    `<button class="tab-btn" data-coin="${symbol}">${symbol}</button>`
  ).join("");
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
  renderSignal(symbol);
  renderChart(symbol);
  renderMeta(symbol);
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
  buildTabs(coins);
  selectCoin(symbols[0]);
}

init();
