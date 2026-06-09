# Genetik Algoritma ile Yazar Tanıma

Türkçe metinlerden, yazıların hangi yazara ait olduğunu tahmin eden bir **genetik algoritma (GA)** uygulaması. Her yazar için karakteristik kelimelerden oluşan bir "kelime listesi" genetik algoritma ile optimize edilir; yeni bir metin, hangi yazarın kelime listesindeki sözcükleri daha çok içeriyorsa o yazara atanır.

> **Hasan Subaşı**

---

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Veri Seti](#veri-seti)
- [Yöntem](#yöntem)
- [Deneyler](#deneyler)
- [Kurulum](#kurulum)
- [Çalıştırma](#çalıştırma)
- [Çıktılar](#çıktılar)
- [Proje Yapısı](#proje-yapısı)

---

## Genel Bakış

Problem, bir **yazar atfı (authorship attribution)** problemidir. Amaç, etiketsiz bir metnin hangi yazara ait olduğunu belirlemektir. Bu projede yaklaşım şudur:

- Her yazar, sözlükten seçilmiş `N` kelimelik bir liste ile temsil edilir.
- Bir metin, içerdiği kelimeler bu listelerdeki kelimelerle karşılaştırılarak puanlanır; en yüksek puanı veren yazara atanır.
- Bu kelime listeleri elle değil, **genetik algoritma** ile (seçilim, çaprazlama, mutasyon) eğitim verisi üzerinde optimize edilir.

Sınıflandırma puanı hesaplanırken kelime frekansları doğrudan toplanmaz; bir kelimenin metinde çok tekrar ederek diğerlerini bastırmasını engellemek için `log(1 + frekans)` dönüşümü uygulanır.

## Veri Seti

`raw_texts/` klasörü altında, her yazar için bir alt klasör bulunur. Toplam **9 yazar**, her yazara ait **10 yazı** vardır.

| | |
|---|---|
| Yazar sayısı | 9 |
| Yazar başına yazı | 10 |
| Eğitim / Test ayrımı | 7 / 3 (yazar başına) |
| Toplam eğitim yazısı | 63 |
| Toplam test yazısı | 27 |
| Metin kodlaması (encoding) | `cp1254` (Türkçe) |

Ön işleme adımları:

- Tüm metin küçük harfe çevrilir, noktalama işaretleri temizlenir.
- 3 karakterden kısa kelimeler, salt rakamlar ve sık geçen Türkçe **durak kelimeleri** (ve, bir, bu, ile, için …) elenir.
- Sözlüğe yalnızca tüm yazılarda en az `min_freq` (varsayılan 3) kez geçen kelimeler alınır.

## Yöntem

Genetik algoritmanın temel bileşenleri:

| Bileşen | Açıklama |
|---|---|
| **Birey (Individual)** | Her yazar için `N` kelimelik bir liste tutar. Başlangıçta sözlükten rastgele seçilir. |
| **Uygunluk (Fitness)** | Eğitim setinde doğru sınıflandırılan yazı oranı (`doğru / 63`). |
| **Seçilim** | Turnuva seçimi (varsayılan `k=5`). |
| **Çaprazlama** | Yazar başına tek noktalı çaprazlama; tekrar eden kelimeler ayıklanır, eksik kalan yerler sözlükten doldurulur. |
| **Mutasyon** | Her kelime, `mutation_rate` olasılıkla ya sözlükten rastgele bir kelimeyle ya da başka bir yazarın listesindeki bir kelimeyle değiştirilir. |
| **Elitizm** | En iyi bireyler (varsayılan 4) bir sonraki nesle değişmeden aktarılır. |
| **Adaptif mutasyon** | 10 nesil boyunca ilerleme olmazsa mutasyon oranı kademeli artırılır, ilerleme oldukça azaltılır (durgunluktan çıkış için). |

Varsayılan hiperparametreler: `N=50`, `pop_size=100`, `n_generations=70`, `mutation_rate=0.03`, `elitism=4`.

## Deneyler

`run_experiments` fonksiyonu, üç hiperparametrenin başarı üzerindeki etkisini ayrı ayrı inceler (diğerleri sabit tutulur):

| Deney | Değişen parametre | Denenen değerler | Sabitler |
|---|---|---|---|
| **1** | `N` (kelime listesi uzunluğu) | 50, 75, 100 | pop=100, mut=0.05, nesil=70 |
| **2** | `pop_size` (popülasyon büyüklüğü) | 50, 100, 150 | N=75, mut=0.05, nesil=70 |
| **3** | `mutation_rate` (mutasyon oranı) | 0.03, 0.07, 0.15 | N=75, pop=100, nesil=70 |

> Not: Sonuçlar her çalıştırmada değişebilir; tekrar üretilebilirlik için kodun başındaki `random.seed(42)` ve `np.random.seed(42)` satırları açılabilir.

## Kurulum

Python 3.8+ gereklidir.

```bash
pip install numpy matplotlib
```

## Çalıştırma

```bash
python HasanSubasi_23011073_YZ_HW1.py
```

Program sırasıyla veri setini yükler, sözlüğü kurar, eğitim/test ayrımını yapar, üç deneyi koşar ve grafik/tabloları `outputs/` klasörüne kaydeder. İlerleme ve özet sonuçlar terminale yazdırılır.

## Çıktılar

Çalışma sonunda `outputs/` klasöründe üretilen görseller:

| Dosya | İçerik |
|---|---|
| `exp1_N_effect.png` | N etkisi: fitness eğrileri ve eğitim/test karşılaştırması |
| `exp2_popsize_effect.png` | Popülasyon büyüklüğü etkisi |
| `exp3_mutation_effect.png` | Mutasyon oranı etkisi |
| `adaptive_mutation.png` | Adaptif mutasyon oranının nesiller boyunca değişimi |
| `all_configs_fitness.png` | Tüm konfigürasyonların en iyi fitness eğrileri |
| `per_author_accuracy.png` | Yazar bazında test doğruluğu |
| `summary_table.png` | Tüm deneylerin özet tablosu |

## Proje Yapısı

```
.
├── Genetic_algorithm.py             # Ana kaynak kod
├── raw_texts/                       # Veri seti (yazar başına klasör)
│   ├── dhizlan/                     # 1.txt ... 10.txt
│   ├── ecelebi/
│   └── ...                          # toplam 9 yazar
├── outputs/                         # Üretilen grafik ve tablolar
└── README.md
```
