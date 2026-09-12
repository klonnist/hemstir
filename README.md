# OKX TimesFM Tahmin Paneli

OKX'in genel (public, API anahtarı gerekmez) piyasa verisiyle 10 kripto para için Google'ın
**TimesFM** modelini zero-shot modda (yeniden eğitim yok, doğrudan inference) kullanarak
24-48 saatlik saatlik kapanış fiyatı tahmini üreten ve sonucu GitHub Pages üzerinde statik
bir panelde gösteren proje.

**Bu bir yatırım tavsiyesi değildir.** TimesFM yalnızca geçmiş sayısal fiyat serisine bakar;
haber, duyarlılık (sentiment) ya da temel analiz bilgisi yoktur. Gerçek emir göndermez,
sadece istatistiksel bir tahmin gösterir.

## Dosyalar

- `okx_client.py` — OKX `/api/v5/market/candles` için ince istemci
- `generate_forecasts.py` — veri çekme + TimesFM inference + `docs/forecasts.json` üretimi
- `docs/` — GitHub Pages ile yayınlanan statik panel (`index.html`, `style.css`, `app.js`)
  ve Actions'ın periyodik olarak commit'lediği `forecasts.json`
- `.github/workflows/generate_forecasts.yml` — otomasyon (her 6 saatte bir + manuel tetikleme)
- `requirements.txt` — Python bağımlılıkları

## Coinler

BTC, ETH, SOL, XRP, ADA, AVAX, DOGE, DOT, LINK, LTC — hepsi OKX'te `<COIN>-USDT` spot
paritesi, saatlik (`1H`) mum verisiyle.

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

Script varsayılan olarak **TimesFM 2.5** (`google/timesfm-2.5-200m-pytorch`, **Apache-2.0**
lisanslı, ticari kullanıma açık) checkpoint'ini kullanır. `pip install timesfm[torch]`
zaman içinde farklı bir majör sürüm çözümlerse (API `google-research/timesfm` reposunda
değişebiliyor), `generate_forecasts.py`'deki `load_model()` fonksiyonu otomatik olarak eski
API'ye (`TimesFm` + `TimesFmHparams`/`TimesFmCheckpoint`, checkpoint: `google/timesfm-2.0-500m-pytorch`,
o da Apache-2.0) düşer.

> ⚠️ TimesFM **v3.0** ağırlıkları (varsa/kullanılırsa) Google tarafından **ticari kullanıma
> kapalı** bir lisansla dağıtılıyor. Bu proje kasıtlı olarak v3.0'ı **kullanmaz** — hem
> `CHECKPOINT_REPO` hem de `LEGACY_CHECKPOINT_REPO` varsayılanları v2.5/v2.0'a (Apache-2.0)
> sabittir. Checkpoint'i kendiniz değiştirirseniz lisansı kontrol edin.

## GitHub Actions Otomasyonu

`.github/workflows/generate_forecasts.yml`:

- **Manuel**: Repo → **Actions** → *TimesFM tahminlerini guncelle* → **Run workflow**.
- **Otomatik**: her 6 saatte bir (`0 */6 * * *` cron, UTC) çalışır.
- Her çalıştırmada bağımlılıkları kurar (torch + timesfm dahil — model ağırlıkları
  runner'da her seferinde indirilir, önceden hiçbir yerde barındırılmaz), `generate_forecasts.py`'yi
  çalıştırır ve ürettiği `docs/forecasts.json`'ı `[skip ci]` etiketiyle doğrudan `main`
  branch'ine commit'ler.
- Runner CPU üzerinde çalışır (GPU yok); 200M parametrelik model için bu, 10 coin'lik
  bir batch'te makul sürede tamamlanır ama torch kurulumu + ilk indirme dahil workflow'a
  30 dakikalık zaman aşımı payı bırakılmıştır.

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
- Açık/koyu tema desteği (sistem tercihine göre başlar, sağ üstteki düğmeyle değiştirilebilir
  ve tercih tarayıcıda hatırlanır).
- Mobilde de düzgün görünecek şekilde responsive.

## Sorumluluk Reddi

Bu panel istatistiksel bir zero-shot zaman serisi modelinin çıktısını gösterir; yatırım veya
finansal tavsiye değildir. TimesFM'in kripto piyasasına, haberlere ya da duyarlılığa dair
hiçbir bilgisi yoktur — sadece geçmiş sayısal fiyat serisine dayanır. Geçmiş performans
gelecek sonuçların garantisi değildir. Kendi araştırmanızı yapın, sorumlu davranın.
