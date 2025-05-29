# Laporan Proyek Terapan - Prediksi Penyakit Jantung

**Nama:** Eva Meivina Dwiana  
**Judul Proyek:** Prediksi Penyakit Jantung Menggunakan Machine Learning  

## Domain Proyek
Penyakit jantung adalah penyebab kematian nomor satu di dunia menurut WHO. Deteksi dini penyakit ini sangat krusial untuk mencegah komplikasi yang lebih serius. Proyek ini bertujuan membangun model machine learning yang dapat memprediksi risiko penyakit jantung berdasarkan data medis.

Referensi:  
- World Health Organization. (2023). [Cardiovascular Diseases](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))  
- Centers for Disease Control and Prevention. (2023). [Heart Disease Facts](https://www.cdc.gov/heartdisease/facts.htm)

## Business Understanding

### Problem Statement  
Bagaimana memprediksi kemungkinan seseorang terkena penyakit jantung berdasarkan data rekam medis?

### Goals  
Membangun model machine learning yang mampu mengklasifikasi risiko penyakit jantung dengan akurasi tinggi.

### Solution Statement  
1. Membangun model baseline menggunakan Logistic Regression.  
2. Menggunakan model Random Forest untuk meningkatkan performa.  
3. Melakukan hyperparameter tuning pada Random Forest.  
4. Evaluasi menggunakan metrik: accuracy, precision, recall, f1-score.

## Data Understanding

### Sumber Dataset
Dataset yang digunakan adalah `heart.csv`, yang berisi informasi kesehatan pasien dan label apakah pasien menderita penyakit jantung atau tidak.

---

### Ukuran Dataset
- **Jumlah baris (data)**: 918
- **Jumlah kolom (fitur)**: 11 fitur input + 1 fitur target (`HeartDisease`)
- **Missing Value**: Tidak ada
- **Duplikat**: Tidak ditemukan

---

### Distribusi Target (`HeartDisease`)
- `1` → Mengidap penyakit jantung: **55.3%**
- `0` → Tidak mengidap penyakit jantung: **44.7%**

---

### Deskripsi Setiap Fitur

| Fitur              | Tipe           | Deskripsi                                                                 |
|-------------------|----------------|--------------------------------------------------------------------------|
| **Age**           | Numerik        | Usia pasien dalam tahun                                                  |
| **Sex**           | Kategorikal    | Jenis kelamin pasien:<br>`M` = Male (Laki-laki), `F` = Female (Perempuan) |
| **ChestPainType** | Kategorikal    | Jenis nyeri dada:<br>• `TA` = Typical Angina<br>• `ATA` = Atypical Angina<br>• `NAP` = Non-Anginal Pain<br>• `ASY` = Asymptomatic |
| **RestingBP**     | Numerik        | Tekanan darah saat istirahat (mmHg)                                      |
| **Cholesterol**   | Numerik        | Kadar kolesterol serum (mg/dl)                                           |
| **FastingBS**     | Kategorikal    | Gula darah puasa > 120 mg/dl:<br>`1` = Ya, `0` = Tidak                   |
| **RestingECG**    | Kategorikal    | Hasil elektrokardiografi:<br>• `Normal`<br>• `ST` = Kelainan gelombang ST-T<br>• `LVH` = Kemungkinan hipertrofi ventrikel kiri |
| **MaxHR**         | Numerik        | Detak jantung maksimum saat uji tekanan (bpm)                            |
| **ExerciseAngina**| Kategorikal    | Nyeri dada saat olahraga:<br>`Y` = Ya, `N` = Tidak                       |
| **Oldpeak**       | Numerik        | Depresi ST relatif terhadap baseline akibat olahraga (dalam mm)         |
| **ST_Slope**      | Kategorikal    | Kemiringan segmen ST saat puncak:<br>• `Up` = Meningkat<br>• `Flat` = Datar<br>• `Down` = Menurun |
| **HeartDisease**  | Target (0/1)   | Label target:<br>`1` = Mengidap penyakit jantung<br>`0` = Tidak          |

---

### Tujuan Data Understanding
- Memahami struktur dan isi dataset secara menyeluruh
- Menilai kualitas data (misal: missing value, outlier, atau duplikasi)
- Memberikan makna dari masing-masing fitur (baik kategorikal maupun numerik)
- Memberi konteks bagi pembaca sebelum eksplorasi dan analisis data
- Menyiapkan dasar yang kuat untuk proses preprocessing dan modeling

---

### Temuan Awal
- Dataset bersih dan siap diproses lebih lanjut
- Tidak terdapat missing value atau duplikasi
- Perlu encoding untuk fitur kategorikal sebelum modeling
- Target cukup seimbang dan cocok untuk klasifikasi biner

## Data Preparation

Tahap persiapan data ini bertujuan agar dataset siap dan optimal untuk digunakan dalam pelatihan model machine learning. Proses yang dilakukan adalah sebagai berikut:

### 1. Memisahkan Fitur dan Target

Pertama, data dipisahkan menjadi fitur (variabel input) dan target (variabel yang ingin diprediksi). Hal ini penting agar model dapat belajar hanya dari fitur tanpa melihat nilai target secara langsung, sehingga proses pelatihan menjadi lebih tepat.

### 2. Mengidentifikasi Tipe Fitur: Numerik dan Kategorikal

Dataset terdiri dari dua tipe fitur, yaitu numerik (berupa angka) dan kategorikal (berupa kategori). Perbedaan tipe ini menentukan bagaimana data tersebut akan diproses selanjutnya, karena masing-masing memerlukan metode transformasi yang berbeda.

### 3. Menentukan Metode Preprocessing yang Tepat untuk Masing-Masing Tipe Fitur

Fitur numerik distandarisasi menggunakan metode standardisasi agar semua fitur memiliki skala yang seragam dengan rata-rata nol dan standar deviasi satu. Ini penting agar model tidak bias terhadap fitur dengan skala besar. Sedangkan fitur kategorikal diubah menjadi bentuk numerik melalui teknik one-hot encoding, yang mengonversi setiap kategori menjadi variabel biner agar dapat diproses model.

### 4. Menggabungkan Preprocessing dengan Pendekatan Terstruktur

Transformasi untuk fitur numerik dan kategorikal digabungkan menggunakan metode yang memungkinkan penerapan berbeda pada masing-masing tipe fitur secara bersamaan dan efisien. Pendekatan ini membuat pipeline preprocessing menjadi terstruktur dan mudah digunakan.

### 5. Membagi Dataset Menjadi Data Latih dan Data Uji dengan Stratifikasi

Dataset kemudian dibagi menjadi dua bagian, yaitu data latih sebanyak 80% dan data uji sebanyak 20%. Pembagian ini menggunakan teknik stratifikasi berdasarkan target agar proporsi kelas di kedua subset tetap seimbang. Hal ini penting untuk mencegah bias model terhadap kelas tertentu dan memastikan evaluasi model yang lebih akurat.

### 6. Menerapkan Preprocessing ke Data Latih dan Data Uji

Proses preprocessing diterapkan pada data latih dengan cara mempelajari parameter transformasi dari data tersebut, lalu transformasi diterapkan. Pada data uji, hanya dilakukan transformasi berdasarkan parameter yang sudah dipelajari dari data latih agar model dapat diuji secara objektif tanpa kebocoran informasi dari data uji.

## Modeling

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan prediksi penyakit jantung. Beberapa algoritma klasifikasi diuji dan dibandingkan untuk memperoleh model terbaik berdasarkan performa.

Model yang digunakan:
1. Logistic Regression (Baseline)
2. Random Forest Classifier (Default)
3. Random Forest Classifier (Hyperparameter Tuning)

### Model 1: Logistic Regression (Baseline)

#### Cara Kerja
Logistic Regression adalah model klasifikasi linier yang memprediksi probabilitas suatu kelas menggunakan fungsi sigmoid. Model ini menghitung kemungkinan suatu input termasuk ke dalam kelas 1 (positif) berdasarkan kombinasi linier dari fitur input.

#### Parameter
- `random_state=42`: Menjamin replikasi hasil.
- `max_iter=1000`: Menambah jumlah iterasi agar model lebih stabil.
- Parameter lainnya menggunakan nilai default.

#### Kelebihan
- Cepat dan efisien pada dataset berukuran kecil-menengah.
- Mudah diinterpretasi dan digunakan sebagai baseline.

#### Kekurangan
- Tidak mampu menangani hubungan non-linear antar fitur.
- Cenderung memiliki akurasi lebih rendah jika fitur saling kompleks.

### Model 2: Random Forest Classifier (Default)

#### Cara Kerja
Random Forest adalah algoritma _ensemble_ berbasis decision tree. Model ini membangun banyak pohon keputusan dan menggabungkan hasilnya melalui voting. Dengan pendekatan bootstrap dan random feature selection, model ini cenderung lebih tahan terhadap overfitting.

#### Parameter
- `random_state=42`: Menjamin hasil konsisten.
- Parameter lainnya default:
  - `n_estimators=100`
  - `max_depth=None`
  - `min_samples_split=2`

#### Kelebihan
- Mampu menangani fitur non-linear dan interaksi antar fitur.
- Tahan terhadap overfitting dan outlier.
- Tidak memerlukan normalisasi data.

#### Kekurangan
- Kurang interpretatif dibanding Logistic Regression.
- Performa masih bisa ditingkatkan dengan tuning parameter.

### Model 3: Random Forest Classifier (Tuned)

#### Cara Kerja
Random Forest dioptimalkan melalui pencarian hyperparameter menggunakan `GridSearchCV`. Teknik ini melakukan pencarian grid pada kombinasi parameter terbaik berdasarkan kinerja validasi silang (cross-validation) dan metrik F1-score.

#### Parameter Tuning
- `n_estimators`: [100, 200]
- `max_depth`: [None, 10, 20]
- `min_samples_split`: [2, 5]
- `cv=5`, `scoring='f1'`, `n_jobs=-1`

#### Kelebihan
- Memberikan hasil lebih optimal dibanding versi default.
- Mengurangi risiko overfitting/underfitting.
- Meningkatkan akurasi dan stabilitas model.

#### Kekurangan
- Proses pelatihan lebih lama karena banyak kombinasi parameter yang diuji.

### Pemilihan Model Terbaik

Berdasarkan evaluasi performa menggunakan metrik F1-score dan validasi silang, **Random Forest Classifier dengan Hyperparameter Tuning** dipilih sebagai **model terbaik**. Model ini menunjukkan keseimbangan yang baik antara akurasi, generalisasi, dan kemampuan menangani data yang kompleks.

Alasan memilih model ini:
- Memiliki performa lebih tinggi dibanding model baseline dan versi default.
- Menghasilkan prediksi yang lebih stabil dan presisi untuk kasus klasifikasi biner seperti penyakit jantung.

## Evaluation

### Metrik yang Digunakan  
- **Accuracy:** Persentase prediksi yang benar dari total data.  
- **Precision:** Proporsi prediksi positif yang benar.  
  Rumus: Precision = TP / (TP + FP)  
- **Recall:** Proporsi kasus positif yang berhasil dideteksi.  
  Rumus: Recall = TP / (TP + FN)  
- **F1-score:** Harmonik rata-rata precision dan recall.  
  Rumus: F1-score = 2 * (Precision * Recall) / (Precision + Recall)  
- **Confusion Matrix:** Menunjukkan jumlah prediksi benar dan salah untuk tiap kelas.

### Hasil Evaluasi

| Model                  | Accuracy | Precision (class 0) | Precision (class 1) | Recall (class 0) | Recall (class 1) | F1-score (class 0) | F1-score (class 1) |
|------------------------|----------|---------------------|---------------------|------------------|------------------|--------------------|--------------------|
| Logistic Regression     | 0.89     | 0.91                | 0.87                | 0.83             | 0.93             | 0.87               | 0.90               |
| Random Forest (Default) | 0.89     | 0.89                | 0.89                | 0.87             | 0.91             | 0.88               | 0.90               |
| Tuned Random Forest     | 0.90     | 0.91                | 0.89                | 0.85             | 0.93             | 0.88               | 0.91               |

## Kesimpulan

- Model terbaik adalah Tuned Random Forest dengan akurasi 90%.  
- Model ini seimbang antara precision dan recall, cocok untuk deteksi dini penyakit jantung.  
- Model dapat membantu pengambilan keputusan medis dengan hasil yang akurat dan andal.
