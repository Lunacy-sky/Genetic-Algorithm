import os
import re
import random
import copy
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter, defaultdict


DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_texts")
ENCODING     = "cp1254"

# Yok sayılacak kelimeler (çok sık geçen anlamsız kelimeler sözlükte yer almasın)
ignored_words = {
    "ve", "bir", "bu", "da", "de", "ile", "için", "ki", "ne", "ama", "ancak",
    "o", "bu", "şu", "ben", "sen", "biz", "siz", "onlar", "ya", "veya", "hem",
    "de", "da", "daha", "çok", "az", "en", "gibi", "kadar", "sonra", "önce",
    "olan", "olan", "olarak", "ise", "olan", "her", "bazı", "tüm", "bütün",
    "mi", "mu", "mı", "mü", "mi", "var", "yok", "çünkü", "eğer", "ama",
    "olan", "ise", "bile", "dahi", "ile", "ben", "sen", "o", "biz", "siz",
    "bu", "şu", "hangi", "nasıl", "neden", "niçin", "ne", "kim", "nere",
    "hiç", "artık", "zaten", "sadece", "yalnız", "sadece", "hep", "hepsi",
    "öyle", "böyle", "şöyle", "dolayı", "rağmen", "karşın", "göre", "ayrıca",
    "üzerinde", "altında", "yanında", "içinde", "dışında", "arasında",
    "değil", "olan", "olan", "olduğu", "olduğunu", "olarak", "olan",
    "den", "dan", "ten", "tan", "nin", "nın", "nun", "nün", "ın", "in", "un", "ün"
}

def read_text(filepath):
    """Dosyayı Türkçe cp1254 encoding ile oku."""
    with open(filepath, "rb") as f:
        raw = f.read()
    return raw.decode(ENCODING, errors="ignore")

def tokenize(text):
    """Metni kelimelere ayır, noktalama işaretleri ve anlamsız ekleri kaldır."""
    text = text.lower()
    #Noktalama işaretlerini kaldır
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    tokens = text.split()
    # 2 karakterden kısa, ignored_word olan veya sadece rakam olan kelimeleri at
    tokens = [t for t in tokens
              if len(t) >= 3
              and t not in ignored_words
              and not t.isdigit()]
    return tokens

def load_dataset(dataset_path):
    """Tüm veri setini yükle. Yazarlar ve yazarlara ait metinlerde geçen kelimeleri döndürür."""
    authors = sorted([d for d in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, d))])
    data = {}
    for author in authors:
        author_dir = os.path.join(dataset_path, author)
        articles = [] #Her bir yazara ait metinlerde geçen kelimeleri tutar.
        for fname in sorted(os.listdir(author_dir)):
            if fname.endswith(".txt"):
                fpath = os.path.join(author_dir, fname)
                tokens = tokenize(read_text(fpath))
                articles.append(tokens)
        data[author] = articles #data: Tüm kelimeleri tutan liste
    print(f"Yüklendi: {len(authors)} yazar")
    for a, arts in data.items():
        total_tokens = sum(len(t) for t in arts)
        print(f"  {a}: {len(arts)} yazı, toplam {total_tokens} token")
    return authors, data


def build_vocabulary(data, min_freq=3, max_author_ratio=1):
    """
    Tüm yazılardaki tekil kelimeleri içeren sözlük oluştur.
    min_freq        : Sözlüğe girebilmek için tüm yazılarda en az kaç kez geçmeli.
    max_author_ratio: Kelimeyi kullanan yazar sayısı / tüm yazar sayısı < max_author_ratio olmalı (sözlüğe daha karakteristik kelimler ekleinir.)
    """
    freq         = Counter()   # kelimenin tüm yazılardaki toplam frekansı
    author_count = Counter()   # kelimenin kaç farklı yazarda geçtiği
    n_authors    = len(data)
    for articles in data.values():
        author_words = set()        # bu yazara ait tekil kelimeler
        for tokens in articles:
            freq.update(tokens)
            author_words.update(tokens)
        author_count.update(author_words)   # her yazar için bir kez say
 
    vocab = sorted([
        w for w, f in freq.items()
        if f >= min_freq                                      # çok nadir olanları at
        and author_count[w] / n_authors <= max_author_ratio   # çok yaygın olanları at
    ])
 
    print(f"\nSözlük boyutu: {len(vocab)} kelime "
          f"(min_freq={min_freq}, max_author_ratio={max_author_ratio})")
    return vocab


def split_train_test(data, train_count=7):
    """Her yazardan 7 yazıyı eğitime, 3'ünü teste ayır."""
    train_data = {}
    test_data  = {}
    for author, articles in data.items():
        # 0-6 arası train, 7-9 arası test
        train_data[author] = articles[:train_count]
        test_data[author]  = articles[train_count:]
    return train_data, test_data


class Individual:
    """
    Genetic algorithm için birey nesnesinin oluşturulması
    word_lists: {yazar_adı: [kelime1, kelime2, ...]}  (her liste N elemanlı)
    """
    def __init__(self, authors, vocabulary, N):
        self.authors    = authors #yazarlar
        self.vocabulary = vocabulary #sözlük
        self.N          = N #Kelime listesi büyüklüğü
        self.word_lists = {} #Kelime listeleri
        self.fitness    = 0.0 #fitness değeri
        # Her yazar için sözlükten rastgele N kelime seç
        for author in authors:
            self.word_lists[author] = random.sample(vocabulary, N) #Initial olarak her bireye sözlükten rastgele seçilerek kelime listelerinin oluşturulması

    def classify(self, tokens):
        """
        Bir metni sınıflandır: hangi yazara ait kelimeler daha çok geçiyor?
        Score hesabı kelimlerin lineer sayılıp toplandığı değil, logaritma fonksiyonundan geçirilerek toplanmasıyla elde edildi.
        Bu yaklaşım aynı kelimenin metinde çok fazla kez tekrar ederek diğer kelimeleri domine etmesini azaltmak için yapıldı.
        Tahmin edilen yazar adı return edilir
        """
        token_set   = Counter(tokens) 
        scores      = {}
        for author, wlist in self.word_lists.items():
            # score = sum(token_set.get(w, 0) for w in wlist) (eski sınıflandırma fonksiyonu)
            score = sum(math.log(1 + token_set.get(w, 0)) for w in wlist) # gürültüsünü azaltmak için logaritma ile score hesabı yapıldı
            scores[author] = score

        max_score = max(scores.values())
        # Beraberlik durumunda rastgele yazar ata
        candidates = [a for a, s in scores.items() if s == max_score]
        return random.choice(candidates)

    def evaluate(self, train_data):
        #Fitness fonk = Bir birey için Fitness değerini (eğitim setinde doğru sınıflandırılan metin sayısı / 63) hesaplar.
        correct = 0
        total   = 0
        for true_author, articles in train_data.items():
            for tokens in articles:
                pred = self.classify(tokens)
                if pred == true_author:
                    correct += 1
                total += 1
        self.fitness = correct / total if total > 0 else 0.0
        return self.fitness
    

# Genetic Algorithm crossover / mutation

def crossover(parent1, parent2):
    """
    İki ebeveynden iki çocuk üret.
    Her yazara ait liste için tek noktalı crossover uygula.
    """
    child1 = copy.deepcopy(parent1) #deepcopy: bağımsız kopya oluşturur.
    child2 = copy.deepcopy(parent2)

    for author in parent1.authors:
        N    = parent1.N
        pt   = random.randint(1, N - 1)
        l1   = parent1.word_lists[author]
        l2   = parent2.word_lists[author]

        new1 = list(dict.fromkeys(l1[:pt] + l2[pt:]))[:N] #tek noktadan (pt) crossover yapılır.
        new2 = list(dict.fromkeys(l2[:pt] + l1[pt:]))[:N] #dict.fromkeys ile duplicate sözcükler (iki parenttan da gelen) engellenir.

        # Eksik kalan slotları sözlükten doldur
        while len(new1) < N:
            cand = random.choice(parent1.vocabulary)
            if cand not in new1:
                new1.append(cand)
        while len(new2) < N:
            cand = random.choice(parent2.vocabulary)
            if cand not in new2:
                new2.append(cand)
        child1.word_lists[author] = new1
        child2.word_lists[author] = new2

    return child1, child2

def mutate(individual, mutation_rate):
    """
    Mutasyon: her kelimeyi mutation_rate olasılıkla;
      - sözlükten rastgele bir kelime ile değiştir (%50)
      - başka bir yazarın listesindeki bir kelime ile değiştir (%50)
    """
    authors = individual.authors
    vocab   = individual.vocabulary
    N       = individual.N

    for author in authors:
        wlist = individual.word_lists[author]
        for i in range(N):
            if random.random() < mutation_rate:
                if random.random() < 0.5:
                    # Sözlükten rastgele kelime ekleme
                    new_word = random.choice(vocab)
                else:
                    # Başka bir yazarın listesinden kelime ekleme 
                    other = random.choice([a for a in authors if a != author])
                    other_list = individual.word_lists[other]
                    new_word = random.choice(other_list)
                
                if new_word not in wlist: # yeni kelime halihazırda listede yoksa listeye ekle.
                    wlist[i] = new_word

def tournament_selection(population, k=5): #değişiklik
    """Turnuva seçimi: k birey arasından en fitness'ı en yüksek olanı seç."""
    contestants = random.sample(population, k)
    return max(contestants, key=lambda ind: ind.fitness)


# GENETİC ALGORITHM

def genetic_algorithm(authors, vocabulary, train_data, test_data,
                      N=50, pop_size=100, n_generations=70,
                      mutation_rate=0.03, elitism=4,  
                      adaptive_mutation=True, verbose=True):
    """
    Genetik algoritma ile bireyler optimize edilir.
    
    @Parameters:
      N               : Her yazarın kelime listesinin uzunluğu
      pop_size        : Popülasyon büyüklüğü
      n_generations   : Nesil sayısı
      mutation_rate   : Başlangıç mutasyon oranı
      elitism         : Sonraki nesle değişmeden aktarılan birey sayısı
      adaptive_mutation: Durgunlukta mutasyon oranını artır
    """
    start_time = time.time()

    #Başlangıç popülasyonu oluşturulur.
    population = [Individual(authors, vocabulary, N) for _ in range(pop_size)]
    for ind in population:
        ind.evaluate(train_data)

    # İstatistik takibi
    history = { #Her nesil için önemli bilginin tutulacağı liste
        "best_fitness":  [],
        "avg_fitness":   [],
        "worst_fitness": [],
        "mutation_rate": [],
    }

    best_ever = max(population, key=lambda x: x.fitness) # en iyi birey
    stagnation_counter = 0
    current_mut_rate   = mutation_rate

    for gen in range(n_generations): #nesil iterasyonu
        population.sort(key=lambda x: x.fitness, reverse=True) #her nesildeki en iyi, ortalama, en kötü fitness değerli bireyler
        best_fitness  = population[0].fitness
        avg_fitness   = np.mean([ind.fitness for ind in population])
        worst_fitness = population[-1].fitness

        history["best_fitness"].append(best_fitness)
        history["avg_fitness"].append(avg_fitness)
        history["worst_fitness"].append(worst_fitness)
        history["mutation_rate"].append(current_mut_rate)

        if best_fitness > best_ever.fitness:
            best_ever = copy.deepcopy(population[0])
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # Adaptif mutasyon: 10 nesil ilerleme yoksa mutasyon oranını arttır.
        if adaptive_mutation:
            if stagnation_counter >= 10:
                current_mut_rate = min(0.30, current_mut_rate * 1.05)
            else:
                current_mut_rate = max(mutation_rate, current_mut_rate * 0.95)

        if verbose and (gen % 10 == 0 or gen == n_generations - 1):
            elapsed = time.time() - start_time
            print(f"  Nesil {gen+1:3d}/{n_generations} | "
                  f"En iyi: {best_fitness:.4f} | "
                  f"Ortalama: {avg_fitness:.4f} | "
                  f"Mutasyon: {current_mut_rate:.4f} | "
                  f"Süre: {elapsed:.1f}s")

        # ******* Yeni nesil üret *******
        new_population = population[:elitism]   # geçmiş nesildeki en iyi bireyleri koru

        while len(new_population) < pop_size: # popülasyon büyüklüğüne ulaşıncaya kadar
            p1 = tournament_selection(population) #belirli sayıda bireyden en iyisini seç, yeni nesil için kullan
            p2 = tournament_selection(population) #belirli sayıda bireyden en iyisini seç, yeni nesil için kullan
            c1, c2 = crossover(p1, p2)
            mutate(c1, current_mut_rate)
            mutate(c2, current_mut_rate)
            c1.evaluate(train_data) #yeni bireylerin fitness değerlerini hesapla
            c2.evaluate(train_data)
            new_population.extend([c1, c2])

        population = new_population[:pop_size] #yeni nesli oluştur.

    # ---- Test değerlendirmesi ----
    train_acc = best_ever.fitness
    test_correct = 0    #totalde doğru tahmin edilen yazı sayısını tutar
    test_total   = 0    #totalde tahmin edilen yazı sayısını tutar
    test_detail  = {}   #Her yazar için doğruluk oranını tutar
    for true_author, articles in test_data.items():
        correct_for_author = 0 #Her yazar için doğru tahmin edilen yazı sayısını tutar.
        for tokens in articles:
            pred = best_ever.classify(tokens) # en son nesildeki en iyi bireyin test verisi üzerindeki doğruluğu hesaplanır.
            if pred == true_author:
                correct_for_author += 1
                test_correct += 1
            test_total += 1
        test_detail[true_author] = correct_for_author / len(articles)
    test_acc = test_correct / test_total if test_total > 0 else 0.0

    elapsed = time.time() - start_time
    print(f"\n-> Eğitim doğruluğu (en iyi birey): {train_acc:.4f} ({train_acc*100:.1f}%)")
    print(f"  -> Test doğruluğu:                  {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"  -> Toplam süre: {elapsed:.1f} saniye")

    return best_ever, history, train_acc, test_acc, test_detail

# DENEY KOŞUSU: FArklı hiperparametrelerin algoritma başarısı üzerindeki etkisi izlenir 

def run_experiments(authors, vocabulary, train_data, test_data):
    """
    Hiperparametre deneyleri:
      - N (kelime listesi uzunluğu): 50, 75, 100
      - pop_size (popülasyon büyüklüğü): 50, 100, 200
      - mutation_rate (mutasyon oranı): 0.03, 0.07, 0.15
    """
    all_results = []

    #Deney 1: N değerleri (N=50,75,100; pop=20, mut=0.05)
    print("\n" + "="*60)
    print("DENEY 1: N:Kelime listesi uzunluğu etkisi")
    print("  Sabit: pop_size=100, mutation_rate=0.05, nesil=70")
    print("="*60)

    exp1_results = []
    for N in [50, 75, 100]: # Farklı N değerleri için algoritma testi. Popülasyon büyüklüğü ve mutation_rate sabit. Adaptive_mutation açık.
        print(f"\n  >> N={N}")
        best, history, train_acc, test_acc, test_detail = genetic_algorithm(
            authors, vocabulary, train_data, test_data,
            N=N, pop_size=100, n_generations=70,
            mutation_rate=0.05, adaptive_mutation=True, verbose=True
        )
        print("Kelime Listeleri:")
        for yazar in authors:
            print(f"{yazar}: {', '.join(best.word_lists[yazar])}")
        exp1_results.append({
            "label": f"N={N}",
            "N": N, "pop_size": 100, "mutation_rate": 0.05,
            "train_acc": train_acc, "test_acc": test_acc,
            "history": history, "test_detail": test_detail
        })
        all_results.append(exp1_results[-1])

    #  Deney 2: Popülasyon büyüklüğü (pop=50,100,200; N=75, mut=0.05)
    print("\n" + "="*60)
    print("DENEY 2: Popülasyon büyüklüğü etkisi")
    print("  Sabit: N=75, mutation_rate=0.05, nesil=70")
    print("="*60)

    exp2_results = []
    for pop_size in [50, 100, 150]: #Popülasyon büyüklüğü için algoritma testi. Kelime listesi uzunluğu ve mutation_rate sabit. Adaptive_mutation açık.
        print(f"\n  >> pop_size={pop_size}")
        best, history, train_acc, test_acc, test_detail = genetic_algorithm(
            authors, vocabulary, train_data, test_data,
            N=75, pop_size=pop_size, n_generations=70,
            mutation_rate=0.05, adaptive_mutation=True, verbose=True
        )
        print("Kelime Listeleri:")
        for yazar in authors:
            print(f"{yazar}: {', '.join(best.word_lists[yazar])}")
        exp2_results.append({
            "label": f"Pop={pop_size}",
            "N": 75, "pop_size": pop_size, "mutation_rate": 0.05,
            "train_acc": train_acc, "test_acc": test_acc,
            "history": history, "test_detail": test_detail
        })
        all_results.append(exp2_results[-1])

    # Deney 3: Mutasyon oranı (mut=0.03,0.07,0.15; N=75, pop=20)
    print("\n" + "="*60)
    print("DENEY 3: Mutasyon oranı etkisi")
    print("  Sabit: N=75, pop_size=100, nesil=70")
    print("="*60)

    exp3_results = []
    for mut in [0.03, 0.07, 0.15]: #mutasyon oranı için algoritma testi. Kelime listesi uzunluğu ve popülasyon büyüklüğü sabit. adaptive_mutation açık.
        print(f"\n  >> mutation_rate={mut}")
        best, history, train_acc, test_acc, test_detail = genetic_algorithm(
            authors, vocabulary, train_data, test_data,
            N=75, pop_size=100, n_generations=70,
            mutation_rate=mut, adaptive_mutation=False, verbose=True
        )
        print("Kelime Listeleri:")
        for yazar in authors:
            print(f"{yazar}: {', '.join(best.word_lists[yazar])}")
        exp3_results.append({
            "label": f"Mut={mut}",
            "N": 75, "pop_size": 100, "mutation_rate": mut,
            "train_acc": train_acc, "test_acc": test_acc,
            "history": history, "test_detail": test_detail
        })
        all_results.append(exp3_results[-1])

    return exp1_results, exp2_results, exp3_results, all_results

#  GRAFik ve Tablolar

COLORS = ["#2196F3", "#F44336", "#3BCC40", "#FF9800", "#9C27B0", "#00BCD4"]
def plot_experiment(results, title, param_name, out_path):
    """Tek bir deney için fitness eğrileri çiz."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    ax1, ax2 = axes

    # Sol: En iyi & Ortalama fitness eğrileri
    for i, r in enumerate(results):
        gen = range(1, len(r["history"]["best_fitness"]) + 1)
        ax1.plot(gen, r["history"]["best_fitness"],
                 color=COLORS[i], linewidth=2, label=f"{r['label']} (en iyi)")
        ax1.plot(gen, r["history"]["avg_fitness"],
                 color=COLORS[i], linewidth=1.2, linestyle="--",
                 alpha=0.6, label=f"{r['label']} (ort.)")

    ax1.set_xlabel("Nesil", fontsize=11)
    ax1.set_ylabel("Fitness (Doğruluk)", fontsize=11)
    ax1.set_title("Nesil Başına En İyi ve Ortalama Fitness", fontsize=12)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#f8f9fa")

    # Sağ: Eğitim vs Test doğruluk karşılaştırması - sütun grafiği
    labels     = [r["label"] for r in results]
    train_accs = [r["train_acc"] * 100 for r in results]
    test_accs  = [r["test_acc"]  * 100 for r in results]

    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax2.bar(x - w/2, train_accs, w, label="Eğitim", color="#42A5F5", edgecolor="white")
    bars2 = ax2.bar(x + w/2, test_accs,  w, label="Test",   color="#EF5350", edgecolor="white")

    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel("Doğruluk (%)", fontsize=11)
    ax2.set_title("Eğitim ve Test Doğruluğu Karşılaştırması", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_facecolor("#f8f9fa")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grafik kaydedildi: {out_path}")


def plot_mutation_rate_over_time(results, out_path):
    """Adaptif mutasyon oranının nesiller içindeki değişimini çiz."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, r in enumerate(results):
        if "mutation_rate" in r["history"] and r["history"]["mutation_rate"]:
            gen = range(1, len(r["history"]["mutation_rate"]) + 1)
            ax.plot(gen, r["history"]["mutation_rate"],
                    color=COLORS[i], linewidth=2, label=r["label"])

    ax.set_xlabel("Nesil", fontsize=11)
    ax.set_ylabel("Mutasyon Oranı", fontsize=11)
    ax.set_title("Adaptif Mutasyon Oranının Nesiller Boyunca Değişimi", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grafik kaydedildi: {out_path}")


def plot_per_author_accuracy(results_list, labels_list, authors, out_path):
    """Yazar bazında test doğruluklarını göster (en iyi konfigürasyon için)."""
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(authors))
    n = len(results_list)
    w = 0.8 / n

    for i, (r, lbl) in enumerate(zip(results_list, labels_list)):
        accs = [r["test_detail"].get(a, 0) * 100 for a in authors]
        # Tam konfigürasyon bilgisini yaz
        full_label = (f"Deney {i+1} en iyisi\n"
                      f"N={r['N']}, Pop={r['pop_size']}, Mut={r['mutation_rate']}\n"
                      f"(Test: {r['test_acc']*100:.1f}%)")
        ax.bar(x + (i - n/2 + 0.5) * w, accs, w,
               label=full_label, color=COLORS[i % len(COLORS)], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(authors, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Test Doğruluğu (%)", fontsize=11)
    ax.set_title("Yazar Bazında Test Doğruluğu\n(Her deneyin en başarılı konfigürasyonu)", fontsize=12)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_facecolor("#f8f9fa")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grafik kaydedildi: {out_path}")


def plot_summary_table(all_results, out_path):
    """Tüm deneylerin özet tablosu görselleştirilir."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis("off")

    headers = ["Konfigürasyon", "N", "Popülasyon", "Mutasyon\nOranı",
               "Eğitim Doğruluğu", "Test Doğruluğu"]
    rows = []
    for r in all_results:
        rows.append([
            r["label"],
            str(r["N"]),
            str(r["pop_size"]),
            f"{r['mutation_rate']:.2f}",
            f"{r['train_acc']*100:.1f}%",
            f"{r['test_acc']*100:.1f}%",
        ])

    table = ax.table(cellText=rows, colLabels=headers,
                     cellLoc="center", loc="center",
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Başlıklar
    for j in range(len(headers)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Satır rengi
    for i in range(1, len(rows) + 1):
        bg = "#E3F2FD" if i % 2 == 0 else "white"
        for j in range(len(headers)):
            table[i, j].set_facecolor(bg)
        # Test doğruluğu nu göster
        val = float(rows[i-1][5].replace("%", ""))
        if val >= 70:
            table[i, 5].set_facecolor("#C8E6C9")
        elif val >= 50:
            table[i, 5].set_facecolor("#FFF9C4")
        else:
            table[i, 5].set_facecolor("#FFCDD2")

    ax.set_title("Tüm Deneyler — Özet Tablo", fontsize=13,
                 fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grafik kaydedildi: {out_path}")


def plot_combined_fitness(all_results, out_path):
    """Tüm konfigürasyonların en iyi fitness eğrisi tek grafikte gösterilir."""
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, r in enumerate(all_results):
        gen = range(1, len(r["history"]["best_fitness"]) + 1)
        ax.plot(gen, [v * 100 for v in r["history"]["best_fitness"]],
                color=COLORS[i % len(COLORS)], linewidth=2,
                label=f"{r['label']} (test={r['test_acc']*100:.1f}%)")

    ax.set_xlabel("Nesil", fontsize=12)
    ax.set_ylabel("En İyi Fitness (%)", fontsize=12)
    ax.set_title("Tüm Konfigürasyonlar — Nesil Başına En İyi Fitness", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grafik kaydedildi: {out_path}")

# main 
if __name__ == "__main__":
    #random.seed(42) -> Programın farklı sonuçlar vermesi için yorum satırına alındı. Tersi istenirse kullanılabilir.
    #np.random.seed(42)

    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Genetik Algoritma ile Yazar Tanıma")
    print("=" * 60)

    # Veri yükleme
    print("\n1. Veri seti yükleniyor")
    print(DATASET_PATH)
    authors, data = load_dataset(DATASET_PATH)
   
    print("\n2. Sözlük oluşturuluyor")
    vocabulary = build_vocabulary(data)

    # Eğitim/Test ayrımı
    print("\n[3] Eğitim/Test ayrımı yapılıyor (7/3)")
    train_data, test_data = split_train_test(data, train_count=7)

    # Farklı hüperparametreler ile hazırlanan deneyler koşulur.
    print("\n[4] Deneyler başlatılıyor\n")
    exp1, exp2, exp3, all_results = run_experiments(
        authors, vocabulary, train_data, test_data
    )

    # Grafikler oluşturulur.
    print("\n[5] Grafikler üretiliyor...")

    plot_experiment(exp1, "Deney 1: N (Kelime Listesi Uzunluğu) Etkisi",
                    "N", f"{OUT_DIR}/exp1_N_effect.png")

    plot_experiment(exp2, "Deney 2: Popülasyon Büyüklüğü Etkisi",
                    "pop_size", f"{OUT_DIR}/exp2_popsize_effect.png")

    plot_experiment(exp3, "Deney 3: Mutasyon Oranı Etkisi",
                    "mutation_rate", f"{OUT_DIR}/exp3_mutation_effect.png")

    plot_mutation_rate_over_time(exp1, f"{OUT_DIR}/adaptive_mutation.png")

    plot_combined_fitness(all_results, f"{OUT_DIR}/all_configs_fitness.png")

    # En iyi 3 konfigürasyonu yazar bazında karşılaştır
    best_each = [
        max(exp1, key=lambda r: r["test_acc"]),
        max(exp2, key=lambda r: r["test_acc"]),
        max(exp3, key=lambda r: r["test_acc"]),
    ]
    plot_per_author_accuracy(
        best_each,
        [r["label"] for r in best_each],
        authors,
        f"{OUT_DIR}/per_author_accuracy.png"
    )

    plot_summary_table(all_results, f"{OUT_DIR}/summary_table.png")

    print("\n" + "=" * 60)
    print("TÜM DENEYLER TAMAMLANDI")
    print("=" * 60)
    print("\nÖzet:")
    for r in all_results:
        print(f"  {r['label']:20s} | Eğitim: {r['train_acc']*100:5.1f}% | Test: {r['test_acc']*100:5.1f}%")
