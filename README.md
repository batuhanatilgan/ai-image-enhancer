# AI Image Enhancer

**Düşük Kaliteli Güvenlik Kamerası ve IP Kamera Görüntülerini İyileştirme Sistemi**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Özellikler](#-özellikler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Teknik Detaylar](#-teknik-detaylar)
- [Neden Model Eğitilmedi?](#-neden-model-eğitilmedi)
- [Gerçek Hayat Senaryoları](#-gerçek-hayat-senaryoları)
- [Proje Yapısı](#-proje-yapısı)
- [Referanslar](#-referanslar)

---

## 🎯 Proje Hakkında

Bu proje, düşük kaliteli güvenlik kamerası kayıtları, IP kamera görüntüleri ve eski video/fotoğraf arşivlerinin kalitesini artırmak için geliştirilmiş kapsamlı bir görüntü işleme sistemidir.

### Motivasyon

Güvenlik kameraları ve sokak izleme sistemleri genellikle düşük maliyetli donanımlar kullanır. Bu durum şu sorunlara yol açar:

- **Düşük çözünürlük**: 480p veya daha düşük görüntüler
- **Yüksek gürültü**: Sensör kalitesizliği ve sıkıştırma artefaktları
- **Bulanıklık**: Düşük kaliteli lensler ve hareket bulanıklığı
- **Düşük kontrast**: Yetersiz aydınlatma koşulları

Bu sistem, yukarıdaki sorunları çözmek için **önceden eğitilmiş (pre-trained) yapay zeka modelleri** ve **klasik görüntü işleme teknikleri**ni birleştirerek kullanır.

---

## ✨ Özellikler

### İşlem Pipeline'ı

```
Girdi → Gürültü Azaltma → Kontrast İyileştirme → Keskinleştirme → Süper Çözünürlük → Çıktı
```

| Adım | Yöntem | Açıklama |
|------|--------|----------|
| 1️⃣ Gürültü Azaltma | Non-Local Means | Kenar koruyucu gürültü temizleme |
| 2️⃣ Kontrast | CLAHE + Gamma | Adaptif kontrast ve parlaklık |
| 3️⃣ Keskinleştirme | Unsharp Masking | Detay ve kenar vurgulama |
| 4️⃣ Süper Çözünürlük | EDSR/FSRCNN | AI tabanlı 2x-4x büyütme |

### Desteklenen Modeller

| Model | Kalite | Hız | Boyut | Kullanım Alanı |
|-------|--------|-----|-------|----------------|
| **EDSR** | ⭐⭐⭐⭐⭐ | Yavaş | ~40MB | Yüksek kalite gerektiren durumlar |
| **FSRCNN** | ⭐⭐⭐⭐ | Hızlı | ~60KB | Gerçek zamanlı işleme |
| **ESPCN** | ⭐⭐⭐ | Çok Hızlı | ~60KB | Video işleme |
| **LapSRN** | ⭐⭐⭐⭐ | Orta | ~2MB | Progressive upscaling |

---

## 🚀 Kurulum

### Gereksinimler

- Python 3.8 veya üzeri
- OpenCV (opencv-contrib-python)

### Adımlar

```bash
# 1. Repository'yi klonlayın
git clone https://github.com/your-username/ai-image-enhancer.git
cd ai-image-enhancer

# 2. Sanal ortam oluşturun (önerilir)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### Model İndirme

Modeller ilk çalıştırmada otomatik olarak `models/` klasörüne indirilir. Manuel indirmek isterseniz:

- [EDSR_x2.pb](https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb)
- [EDSR_x4.pb](https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb)
- [FSRCNN_x2.pb](https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb)

---

## 📖 Kullanım

### Temel Kullanım

```bash
# Tek görüntü işleme
python main.py --input input_images/test.jpg --output output_images/

# Tüm klasörü işleme
python main.py --input input_images/ --output output_images/

# Görüntü analizi
python main.py --input test.jpg --analyze-only
```

### Gelişmiş Seçenekler

```bash
# FSRCNN modeli ile 4x büyütme
python main.py --input image.jpg --model fsrcnn --scale 4

# Sadece gürültü azaltma ve kontrast
python main.py --input image.jpg --no-sharpen --no-super-res

# Özel parametrelerle çalıştırma
python main.py --input image.jpg \
    --denoise-strength 12 \
    --clahe-clip 2.5 \
    --sharpen-amount 2.0 \
    --model edsr --scale 2
```

### Karşılaştırma

```bash
# Detaylı karşılaştırma (histogram + metrikler)
python compare.py --original input.jpg --enhanced output.jpg

# Yan yana karşılaştırma kaydet
python compare.py --original input.jpg --enhanced output.jpg --side-by-side --save comparison.png

# Yakınlaştırmalı karşılaştırma
python compare.py --original input.jpg --enhanced output.jpg --zoom
```

### Python API

```python
from main import ImageEnhancer

# Enhancer oluştur
enhancer = ImageEnhancer(
    model_name="edsr",
    scale=2,
    denoise_strength=10,
    sharpen_amount=1.5
)

# Görüntüyü işle
result = enhancer.enhance("input.jpg")

# Sonucu kaydet
from src.utils import save_image
save_image(result["enhanced"], "output.jpg")
```

---

## 🔬 Teknik Detaylar

### 1. Gürültü Azaltma (Non-Local Means)

Non-Local Means algoritması, geleneksel filtrelerin aksine, gürültüyü azaltırken kenar detaylarını korur. Algoritma, her piksel için tüm görüntüdeki benzer pikselleri arar ve ağırlıklı ortalama alır.

**Matematiksel formül:**

```
NL[u](p) = Σq w(p,q) * u(q)
```

Burada `w(p,q)`, p ve q pikselleri arasındaki benzerlik ağırlığıdır.

**Avantajları:**
- Kenarları korur
- Tekrarlanan dokuları iyi işler
- Güvenlik kamerası gürültüsü için etkili

### 2. Kontrast İyileştirme (CLAHE)

CLAHE (Contrast Limited Adaptive Histogram Equalization), görüntüyü küçük bloklara bölerek her blokta ayrı histogram eşitleme uygular.

**Parametreler:**
- `clipLimit`: Kontrast amplifikasyon sınırı (varsayılan: 2.0)
- `tileGridSize`: Blok sayısı (varsayılan: 8x8)

**Gamma Correction:**

```
output = ((input / 255) ^ gamma) * 255
```

- `gamma < 1`: Karanlık bölgeleri aydınlatır
- `gamma > 1`: Parlak bölgeleri karartır

### 3. Keskinleştirme (Unsharp Masking)

Unsharp Masking, orijinal görüntüden bulanık versiyonunu çıkararak kenarları vurgular.

**Formül:**

```
sharpened = original + amount * (original - blurred)
```

**Parametreler:**
- `amount`: Keskinleştirme miktarı (önerilen: 1.0-2.5)
- `threshold`: Gürültü filtresi eşiği

### 4. Süper Çözünürlük (Deep Learning)

EDSR (Enhanced Deep Residual Networks) modeli, residual bloklar kullanarak düşük çözünürlüklü görüntülerden yüksek çözünürlüklü görüntüler üretir.

**Mimari özellikleri:**
- 32 residual blok
- Batch normalization yok (daha kararlı eğitim)
- L1 loss fonksiyonu

---

## 🤔 Neden Model Eğitilmedi?

Bu projede kendi modellerimizi eğitmek yerine **önceden eğitilmiş modeller** kullanılmasının birkaç önemli nedeni vardır:

### 1. Kaynak Verimliliği

Model eğitimi için:
- ❌ Yüksek performanslı GPU'lar (RTX 3090, A100 vb.)
- ❌ Büyük veri setleri (DIV2K, Flickr2K - binlerce görüntü)
- ❌ Haftalarca eğitim süresi
- ❌ Hiperparametre optimizasyonu

Pre-trained modeller:
- ✅ Herhangi bir CPU/GPU'da çalışır
- ✅ Anında kullanıma hazır
- ✅ Denenmiş ve optimize edilmiş

### 2. Akademik Geçerlilik

Kullandığımız modeller (EDSR, FSRCNN) peer-reviewed akademik makalelerde yayınlanmış ve binlerce araştırmacı tarafından doğrulanmıştır:

- **EDSR**: CVPRW 2017, 5000+ atıf
- **FSRCNN**: ECCV 2016, 3000+ atıf

### 3. Genelleştirme Yeteneği

Bu modeller, çeşitli görüntü türlerinde eğitilmiştir:
- Doğal görüntüler
- Şehir manzaraları
- İnsan yüzleri
- Metinler ve grafikler

Bu çeşitlilik, modellerin güvenlik kamerası görüntülerinde de iyi performans göstermesini sağlar.

### 4. Pratik Uygulama

Gerçek dünya uygulamalarında:
- Hızlı deployment
- Minimal bakım
- Güvenilir sonuçlar

---

## 🌍 Gerçek Hayat Senaryoları

### 1. Güvenlik ve Gözetim Sistemleri

**Senaryo:** Bir AVM güvenlik kamerası 480p çözünürlükte kayıt yapıyor. Bir olay sonrası şüphelinin yüzünü tanımak gerekiyor.

**Çözüm:**
- Gürültü azaltma → Sensör gürültüsünü temizler
- Kontrast iyileştirme → Düşük ışıklı ortamda detayları ortaya çıkarır
- Süper çözünürlük → Yüz özelliklerini daha net gösterir

### 2. Şehir İzleme Sistemleri (Smart City)

**Senaryo:** Trafik kameraları plaka okuma için yeterli kaliteyi sağlayamıyor.

**Çözüm:**
- Keskinleştirme → Plaka karakterlerini netleştirir
- Süper çözünürlük → Uzak araçların plakalarını okunabilir hale getirir

### 3. Forensic Analiz

**Senaryo:** Adli bilişim uzmanları eski DVR kayıtlarından kanıt çıkarmaya çalışıyor.

**Çözüm:**
- Tam pipeline → Tüm iyileştirme adımları uygulanır
- Karşılaştırma aracı → Jüriye sunulmak üzere önce/sonra görselleri oluşturulur

### 4. Arşiv Restorasyon

**Senaryo:** Eski aile fotoğrafları veya tarihsel görüntüler dijitalleştirilmiş ancak kalitesi düşük.

**Çözüm:**
- Gürültü azaltma → Film grenini temizler
- Kontrast → Solmuş renkleri canlandırır
- Süper çözünürlük → Detayları geri kazandırır

### 5. Tele-tıp ve Uzaktan Tanı

**Senaryo:** Düşük bant genişliği nedeniyle sıkıştırılmış tıbbi görüntüler.

**Çözüm:**
- Sıkıştırma artefaktlarının giderilmesi
- Tanı için kritik detayların iyileştirilmesi

---

## 📁 Proje Yapısı

```
ai-image-enhancer/
├── input_images/           # Girdi görüntüleri
│   └── (test görüntülerinizi buraya koyun)
├── output_images/          # İyileştirilmiş çıktılar
├── models/                 # Pre-trained modeller (otomatik indirilir)
│   ├── EDSR_x2.pb
│   ├── EDSR_x4.pb
│   └── FSRCNN_x2.pb
├── src/
│   ├── __init__.py
│   ├── noise_reduction.py  # Non-Local Means Denoising
│   ├── contrast_enhance.py # CLAHE + Gamma Correction
│   ├── sharpening.py       # Unsharp Masking
│   ├── super_resolution.py # DNN Super Resolution
│   └── utils.py            # Yardımcı fonksiyonlar
├── main.py                 # Ana pipeline ve CLI
├── compare.py              # Karşılaştırma aracı
├── requirements.txt        # Bağımlılıklar
└── README.md               # Bu dosya
```

---

## 📚 Referanslar

### Akademik Makaleler

1. Buades, A., Coll, B., & Morel, J. M. (2005). **A non-local algorithm for image denoising.** *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

2. Zuiderveld, K. (1994). **Contrast limited adaptive histogram equalization.** *Graphics Gems IV*, 474-485.

3. Lim, B., Son, S., Kim, H., Nah, S., & Lee, K. M. (2017). **Enhanced deep residual networks for single image super-resolution.** *IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*.

4. Dong, C., Loy, C. C., & Tang, X. (2016). **Accelerating the super-resolution convolutional neural network.** *European Conference on Computer Vision (ECCV)*.

### Dokümantasyon

- [OpenCV Documentation](https://docs.opencv.org/)
- [OpenCV DNN Super Resolution](https://docs.opencv.org/4.x/d5/d29/tutorial_dnn_superres_upscale_image_single.html)

---

## 📄 Lisans

Bu proje MIT lisansı altında sunulmaktadır. Detaylar için [LICENSE](LICENSE) dosyasına bakınız.

---

## 🤝 Katkıda Bulunma

Katkılarınızı memnuniyetle karşılıyoruz! Lütfen bir pull request göndermeden önce bir issue açın.

---

**Not:** Bu proje akademik ve araştırma amaçlıdır. Elde edilen sonuçların hukuki delil olarak kullanılması, ilgili yasal prosedürlere tabidir.
