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

**Sumber Dataset:**  
[heart.csv](https://raw.githubusercontent.com/Evameivina/heart_ml/refs/heads/main/heart.csv)

**Ukuran Dataset:**  
- Jumlah Data: 918 baris  
- Fitur: 11 fitur + 1 target (HeartDisease)  
- Tidak ada nilai yang hilang

**Distribusi Target:**  
- Positif (1): 55.3%  
- Negatif (0): 44.7%  

**Contoh 5 Data Teratas:**

| Age | Sex | ChestPainType | RestingBP | Cholesterol | FastingBS | RestingECG | MaxHR | ExerciseAngina | Oldpeak | ST_Slope | HeartDisease |
|------|-----|--------------|-----------|-------------|-----------|------------|-------|---------------|---------|----------|--------------|
| 40   | M   | ATA          | 140       | 289         | 0         | Normal     | 172   | N             | 0.0     | Up       | 0            |
| 49   | F   | NAP          | 160       | 180         | 0         | Normal     | 156   | N             | 1.0     | Flat     | 1            |
| 37   | M   | ATA          | 130       | 283         | 0         | ST         | 98    | N             | 0.0     | Up       | 0            |
| 48   | F   | ASY          | 138       | 214         | 0         | Normal     | 108   | Y             | 1.5     | Flat     | 1            |
| 54   | M   | NAP          | 150       | 195         | 0         | Normal     | 122   | N             | 0.0     | Up       | 0            |

## Data Preparation

1. Memisahkan fitur (`X`) dan target (`y`).  
2. Mengubah fitur kategorikal dengan one-hot encoding.  
3. Melakukan standardisasi fitur numerik menggunakan StandardScaler.  
4. Membagi data menjadi data latih (80%) dan data uji (20%) secara stratified.

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
