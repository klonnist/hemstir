# OKX TimesFM Tahmin Paneli

OKX'in genel (public, API anahtarı gerekmez) piyasa verisiyle 14 kripto para için Google'ın
**TimesFM** modelini zero-shot modda (yeniden eğitim yok, doğrudan inference) kullanarak
24-48 saatlik saatlik kapanış fiyatı tahmini üreten, buna ATR bazlı bir **AL/SAT + TP/SL**
sinyali ekleyen, her tahmini **arşivleyip** gerçekleşen fiyatlarla değerlendiren ve isteğe
bağlı bir **walk-forward backtest** ile geçmişte nasıl performans göstereceğini simüle eden,
sonucu GitHub Pages üzerinde statik bir panelde gösteren proje.

**Bu bir yatırım tavsiyesi değildir.** TimesFM yalnızca geçmiş sayısal fiyat serisine bakar;
haber, duyarlılık (sentiment) ya da temel analiz bilgisi yoktur. AL/SAT + TP/SL de kanıtlanmış
bir strateji değil, basit ve kural tabanlı bir hesaplamadır (aşağıya bakın). Gerçek emir
göndermez, sadece istatistiksel bir tahmin ve buna bağlı bir seviye önerisi gösterir.

## Dosyalar

- `okx_client.py` — OKX `/api/v5/market/candles` (+ `/market/history-candles` sayfalama)
  için ince istemci
- `indicators.py` — ATR (Average True Range) hesaplama
- `eval_lib.py` — tahmin degerlendirme + islem simulasyonu (TP/SL/likidasyon tespiti,
  toplulastirma) icin paylasilan, ag erisimi olmayan fonksiyonlar; birim testleri var
  (`tests/test_eval_lib.py`)
- `generate_forecasts.py` — veri çekme + TimesFM inference + sinyal hesaplama +
  `docs/forecasts.json` üretimi + her çalıştırmayı `docs/history/<COIN>.jsonl`'a arşivler
- `scripts/backfill_from_git.py` — **tek seferlik**: `docs/forecasts.json`'ın git geçmişindeki
  eski sürümlerini arşive geri kazanır (bu repoda zaten çalıştırıldı, tekrar çalıştırmanıza
  gerek yok)
- `scripts/evaluate_archive.py` — arşivdeki gerçek geçmiş tahminleri gerçekleşen OKX
  fiyatlarıyla kıyaslar, `docs/evaluation.json` üretir (generate_forecasts.yml'nin bir parçası)
- `scripts/backtest.py` — walk-forward backtest: geçmişte her 6 saatte bir TimesFM'i SADECE
  o ana kadarki veriyle çalıştırıp aynı sinyal mantığını uygular, `docs/backtest.json` üretir
  (ayrı, manuel tetiklenen workflow — bkz. aşağı)
- `docs/` — GitHub Pages ile yayınlanan statik panel (`index.html`, `style.css`, `app.js`)
  ve Actions'ın periyodik olarak commit'lediği `forecasts.json` / `history/` / `evaluation.json`
  / `backtest.json`
- `.github/workflows/generate_forecasts.yml` — otomasyon (her 6 saatte bir + manuel tetikleme)
- `.github/workflows/backtest.yml` — walk-forward backtest'i manuel tetikler (uzun sürebilir)
- `requirements.txt` — Python bağımlılıkları

## AL/SAT + TP/SL Sinyali Nasıl Hesaplanır

Bu proje `crypto-trader` (aynı geliştiricinin başka bir reposu, EMA/RSI/ATR bazlı bir OKX
botu) ile aynı TP/SL yaklaşımını kullanır:

1. **Yön (BUY/SELL)**: TimesFM'in ufuk sonu (varsayılan 48. saat) için tahmin ettiği fiyat,
   mevcut (son gerçek) kapanıştan yüksekse **BUY**, düşükse **SELL**.
2. **TP/SL**: `ATR(14)`'ün (saatlik mumlardan Wilder yöntemiyle hesaplanır) sabit katları —
   `SL = giriş ∓ 1.5×ATR`, `TP = giriş ± 2.5×ATR` (yön BUY/SELL'e göre işaret değişir),
   yani risk/ödül oranı sabit ~1:1.67.
3. **Sinyal gücü**: beklenen hareketin ATR'ye oranı 1'in altındaysa "zayıf", 1-2 arasında
   "orta", 2 ve üzerinde "güçlü" olarak etiketlenir — yönü değiştirmez, sadece bilgi amaçlıdır.

TimesFM'in ürettiği güven aralığı (quantile band) TP/SL için **kullanılmaz** — o saf
istatistiksel yayılımdır, risk yönetimi için tasarlanmamıştır; grafikte sadece görsel olarak
gösterilir. `forecasts.json`'daki her coin'in `signal` alanında `side`, `strength`,
`entry_price`, `atr`, `expected_move_pct`, `take_profit`, `stop_loss`, `risk_reward`
bulunur (yeterli geçmiş veri yoksa `null` olabilir).

## Coinler

Hepsi **perp / vadeli** (`<COIN>-USDT-SWAP`), saatlik (`1H`) mum verisiyle — kullanıcı
kaldıraçlı işlem açtığı için tüm coinler spot yerine perp fiyat/volatilite verisinden
hesaplanıyor. Liste `generate_forecasts.py`'deki `COINS` sözlüğünde tanımlı:

BTC, ETH, SOL, XRP, ADA, AVAX, DOGE, DOT, LINK, LTC, ETHFI, CRV, NEAR, BNB —
[Hasanwavebot](https://klonnist.github.io/Hasanwavebot/) botunun izlediği coin listesinden
esinlenildi.

## Yerel Çalıştırma

```bash
pip install -r requirements.txt
python generate_forecasts.py
```

Bu komut OKX'ten her coin için son ~300 saatlik (≈12.5 gün) kapanış verisini çeker, TimesFM
ağırlıklarını Hugging Face'ten indirir (ilk çalıştırmada birkaç yüz MB — bağlantı hızınıza
göre birkaç dakika sürebilir), 48 saatlik tahmini üretir ve `docs/forecasts.json`'a yazar.
Ardından `docs/index.html`'i tarayıcıda açarak sonucu yerelde görebilirsiniz (bir HTTP sunucusu
üzerinden açmanız gerekir, örn. `python -m http.server` — `fetch()` `file://` ile çalışmaz).

Ortam değişkenleriyle ayarlanabilir: `CONTEXT_HOURS` (varsayılan 300), `HORIZON_HOURS`
(varsayılan 48), `TIMESFM_CHECKPOINT`.

## TimesFM Sürümü ve Lisansı (önemli)

Script **TimesFM 2.5** (`google/timesfm-2.5-200m-pytorch`, **Apache-2.0** lisanslı, ticari
kullanıma açık) checkpoint'ini, `timesfm.TimesFM_2p5_200M_torch` sınıfıyla yükler.

`requirements.txt`'de `timesfm[torch]==2.0.2` olarak **bilerek sabitlendi**: `pip install
timesfm[torch]` (sürüm belirtmeden) 2026 itibarıyla PyPI'den **timesfm 3.0.x**'i çözüyor.

> ⚠️ TimesFM **v3.0** ağırlıkları Google tarafından **ticari kullanıma kapalı** bir lisansla
> dağıtılıyor. Bu proje kasıtlı olarak v3.0'ı **kullanmaz** — `timesfm` paketi 2.0.2'ye
> sabitlenmiştir (bu paket sürümü, 2.5 model mimarisini/checkpoint'ini Apache-2.0 API'siyle
> yükleyen `TimesFM_2p5_200M_torch` sınıfını içerir). `requirements.txt`'i kendiniz
> güncellerseniz hangi sürümün hangi lisansla geldiğini kontrol edin.

## Tahmin Arşivi ve Geçmiş Performans

Her `generate_forecasts.py` çalıştırması, o anki tahmini/sinyali `docs/history/<COIN>.jsonl`'a
da ekler (dosya başına en fazla ~800 kayıt — 6 saatlik aralıkla ~200 gün — eskisi otomatik
budanır). Bu arşiv sayesinde panel "geçmişte model ne tahmin etti, tuttu mu, sinyallere göre
işlem açsaydım ne olurdu" sorularına gerçek verilerle cevap verebiliyor:

- `scripts/evaluate_archive.py`, `generate_forecasts.yml`'nin bir parçası olarak her
  çalıştırmada arşivdeki tahminleri OKX'ten çekilen **gerçekleşmiş** fiyatlarla kıyaslayıp
  `docs/evaluation.json`'ı günceller — bu "Canlı arşiv" verisidir, panelde öyle etiketlenir.
- Bu repo ilk kurulduğunda `docs/forecasts.json`'ın git geçmişindeki eski sürümleri
  `scripts/backfill_from_git.py` ile tek seferlik geri kazanıldı (bkz. yukarıdaki dosya
  listesi) — o yüzden arşiv, projenin en başından beri var gibi davranıyor.

## Backtest (Geriye Dönük Simülasyon)

`scripts/backtest.py`, canlı sistemle **aynı sıklıkta** (varsayılan 6 saatte bir) geçmişte
bir "kesim noktası" seçip, TimesFM'i **SADECE o ana kadarki veriyle** (son 300 saat) çalıştırır,
aynı `compute_tp_sl`/`signal_strength` fonksiyonlarıyla sinyali üretir, sonra gerçek gelecek
mumlarla (o kesim noktasından SONRAsı) sonucu değerlendirir — gelecek veri hiçbir aşamada
sinyal üretimine sızmaz (look-ahead bias yok).

Çalıştırma (yerel ya da `.github/workflows/backtest.yml` üzerinden manuel):

```bash
BACKTEST_DAYS=75 BACKTEST_STEP_HOURS=6 BACKTEST_LEVERAGE=1 BACKTEST_FEE_PCT=0.05 \
  python scripts/backtest.py
```

Önemli ortam değişkenleri: `BACKTEST_DAYS` (varsayılan 75), `BACKTEST_STEP_HOURS` (6),
`BACKTEST_LEVERAGE` (1), `BACKTEST_FEE_PCT` (işlem başı, %0.05), `BACKTEST_MAX_CUTPOINTS`
(hızlı test için kesim noktası sayısını sınırlar), `BACKTEST_COINS` (virgüllü alt küme).
TimesFM modeli bir kere yüklenip tüm kesim noktalarında yeniden kullanılır (14 coin tek
batch'te); yine de 75 gün × 6 saat ≈ 300 kesim noktası uzun sürebilir — `backtest.yml`
workflow'unun zaman aşımı buna göre (340 dakika) geniş tutuldu.

TimesFM'in kendi tahminine ek olarak üç **karşılaştırma stratejisi** de aynı çerçeveyle
(aynı giriş/ATR/TP-SL) simüle edilir: **her zaman AL**, **momentum** (son 48 saatin yönünü
takip et) ve **rastgele yön** — model bunlardan belirgin şekilde iyi değilse panelde bu
açıkça görülür.

## Metrikler Ne Anlama Gelir

- **Yön doğruluğu**: fiyat, tahmin edilen yöne (6/12/24/48. saatte) gerçekten gitti mi (%).
- **MAE / MAPE**: tahmin edilen fiyat ile gerçekleşen fiyat arasındaki ortalama mutlak
  hata (fiyat biriminde / yüzde olarak).
- **Güven aralığı isabeti**: gerçekleşen fiyat, TimesFM'in kantil bandının (alt/üst sınır)
  içinde kaldı mı (%) — bandın kendisi TP/SL için kullanılmaz, sadece bu ayrı metrikte
  değerlendirilir.
- **İki mod**: "her sinyal bağımsız işlem" (her 6 saatlik sinyal kendi işlemi, üst üste
  binebilir) ve "coin başına tek pozisyon" (önceki işlem kapanmadan yeni sinyal atlanır).
- **TP/SL/Likidasyon tespiti**: her saatlik mumun high/low'una bakılır; aynı mumda hem TP
  hem SL/likidasyon tetiklenirse **temkinli davranılır, kayıp sayılır**. Ufuk (48 saat)
  içinde ikisi de tetiklenmezse "süre doldu" (o anki kapanıştan çıkılır).
- **Kâr/zarar (%)**: sabit pozisyon büyüklüğü varsayımıyla **toplanır** (bileşik faiz değil)
  — işlem başı %0.05+%0.05 komisyon ve (kaldıraç > 1 ise) yaklaşık likidasyon kontrolü
  dahildir; likidasyonda marjinin tamamı kaybedilmiş sayılır (-%100). Bu basitleştirilmiş
  bir simülasyondur, borsanın gerçek marjin motorunu birebir taklit etmez.
- **Profit factor**: toplam kazancın toplam kayba oranı; kayıp yoksa `null` gösterilir.
- **Örnek sayısı az uyarısı**: 30'un altındaki örneklemler istatistiksel olarak güvenilmez
  kabul edilir, panelde ayrıca işaretlenir.

## GitHub Actions Otomasyonu

`.github/workflows/generate_forecasts.yml` (ana döngü):

- **Manuel**: Repo → **Actions** → *TimesFM tahminlerini guncelle* → **Run workflow**.
- **Otomatik**: her 6 saatte bir (`0 */6 * * *` cron, UTC) çalışır.
- Her çalıştırmada bağımlılıkları kurar (torch + timesfm dahil — model ağırlıkları
  runner'da her seferinde indirilir, önceden hiçbir yerde barındırılmaz), `generate_forecasts.py`'yi
  (tahmin + arşive kayıt) ve `scripts/evaluate_archive.py`'yi (arşiv değerlendirmesi)
  çalıştırır, `docs/forecasts.json` + `docs/history/` + `docs/evaluation.json`'ı `[skip ci]`
  etiketiyle doğrudan `main` branch'ine commit'ler.
- Runner CPU üzerinde çalışır (GPU yok); 200M parametrelik model için bu, 14 coin'lik
  bir batch'te makul sürede tamamlanır ama torch kurulumu + ilk indirme dahil workflow'a
  40 dakikalık zaman aşımı payı bırakılmıştır.

`.github/workflows/backtest.yml` (sadece manuel, isteğe bağlı):

- Repo → **Actions** → *Walk-forward backtest calistir* → **Run workflow** — gün sayısı,
  kesim aralığı, kaldıraç, komisyon, kesim noktası sınırı ve coin alt kümesini inputlardan
  ayarlayabilirsiniz. `docs/backtest.json`'ı üretip commit'ler.

## GitHub Secrets / Pages Ayarı

Bu proje **hiçbir API anahtarı gerektirmez** (OKX genel API'si auth istemez, TimesFM
Hugging Face'ten anonim indirilir) — ek bir secret tanımlamanıza gerek yok.

1. **Settings → Actions → General → Workflow permissions** altında **Read and write
   permissions** seçili olmalı (workflow'un `main`'e push edebilmesi için).
2. **Settings → Pages** sekmesinde **Source** olarak `main` branch, klasör olarak `/docs`
   seçip kaydedin.
3. Siteniz birkaç dakika içinde `https://<kullanıcı-adınız>.github.io/<repo-adı>/`
   adresinde yayında olur. İlk workflow çalışması tamamlanana kadar panel "henüz veri yok"
   mesajı gösterir.

## Panel

- Chart.js ile her coin için ayrı grafik: gerçek geçmiş fiyat (düz çizgi) + tahmin
  (kesikli çizgi, son gerçek noktadan devam eder) + varsa güven aralığı (gölgeli bant).
  Sekmelerle coin seçilir.
- **Geçmiş Performans** bölümü: kaynak olarak "Canlı arşiv" ya da "Backtest" seçilir (ikisi
  net şekilde ayrı etiketlenir, karıştırılmaz); geçmiş tahmin çizgileri gerçek fiyatın
  üzerine bindirilir (yön tuttuysa yeşil, tutmadıysa kırmızı), giriş (▲/▼) ve çıkış
  (TP/SL/süre doldu) işaretleri, kümülatif kâr/zarar (equity) eğrisi — model vs. karşılaştırma
  stratejileri, özet kartları (sinyal gücüne ve yöne göre kırılmış) ve tüm coinleri
  karşılaştıran bir tablo içerir. Örnek sayısı azsa uyarı gösterir.
- Açık/koyu tema desteği (sistem tercihine göre başlar, sağ üstteki düğmeyle değiştirilebilir
  ve tercih tarayıcıda hatırlanır) — yeni bölüm dahil tüm panel için geçerli.
- Mobilde de düzgün görünecek şekilde responsive.

## Sorumluluk Reddi

Bu panel istatistiksel bir zero-shot zaman serisi modelinin çıktısını gösterir; yatırım veya
finansal tavsiye değildir. TimesFM'in kripto piyasasına, haberlere ya da duyarlılığa dair
hiçbir bilgisi yoktur — sadece geçmiş sayısal fiyat serisine dayanır. Geçmiş performans
gelecek sonuçların garantisi değildir. Kendi araştırmanızı yapın, sorumlu davranın.
