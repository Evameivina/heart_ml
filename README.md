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
1. Membangun model baseline menggunakan **Logistic Regression**.
2. Menggunakan model **Random Forest** untuk meningkatkan performa.
3. Melakukan **hyperparameter tuning** pada Random Forest.
4. Evaluasi menggunakan metrik: accuracy, precision, recall, f1-score.

## Data Understanding
Dataset digunakan dari: https://raw.githubusercontent.com/Evameivina/heart_ml/refs/heads/main/heart.csv

### Informasi Dataset:
- Jumlah data: 918
- Fitur: 11 fitur + 1 target (`HeartDisease`)
- Tidak ada nilai yang hilang (missing values)

### Statistik Deskriptif:
- Rata-rata usia: 53 tahun
- Rata-rata tekanan darah: 132
- MaxHR (detak jantung maksimum): 136
- Target seimbang: 55% positif, 45% negatif

Distribusi target HeartDisease:
1    0.553377
0    0.446623

Distribusi target HeartDisease:
1    0.553377
0    0.446623

### 5 Data Teratas:
   Age Sex ChestPainType  RestingBP  Cholesterol  FastingBS RestingECG  MaxHR  ExerciseAngina  Oldpeak ST_Slope  HeartDisease  
0   40   M           ATA        140          289          0     Normal    172              N      0.0       Up             0  
1   49   F           NAP        160          180          0     Normal    156              N      1.0     Flat             1  
2   37   M           ATA        130          283          0         ST     98              N      0.0       Up             0  
3   48   F           ASY        138          214          0     Normal    108              Y      1.5     Flat             1  
4   54   M           NAP        150          195          0     Normal    122              N      0.0       Up             0

## Data Preparation
### Teknik yang digunakan:
- **One-hot encoding** pada fitur kategorikal.
- **StandardScaler** untuk fitur numerik.
- **Train-test split** (80:20)

### Alasan:
- Encoding agar fitur kategorik bisa diproses model.
- Scaling untuk menyamakan skala fitur numerik.
- Pembagian data untuk mengukur performa generalisasi.

## Modeling
### Algoritma yang digunakan:
1. Logistic Regression (baseline)
2. Random Forest Classifier
3. Tuned Random Forest (GridSearchCV)

### Kelebihan dan Kekurangan:
- **Logistic Regression**: cepat, mudah diinterpretasi, tapi tidak menangkap relasi kompleks.
- **Random Forest**: kuat terhadap overfitting dan menangkap hubungan kompleks, tapi butuh waktu komputasi lebih lama.

## Evaluation
### Metrik yang digunakan:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

**Penjelasan metrik:**
- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- `F1-score = 2 * (Precision * Recall) / (Precision + Recall)`

### Hasil Evaluasi:
#### Logistic Regression:
Accuracy: 0.89
Precision: 0.91 (class 0), 0.87 (class 1)
Recall: 0.83 (class 0), 0.93 (class 1)
F1-score: 0.87 (class 0), 0.90 (class 1)

#### Random Forest:
Accuracy: 0.89
Precision: 0.89 (class 0), 0.89 (class 1)
Recall: 0.87 (class 0), 0.91 (class 1)
F1-score: 0.88 (class 0), 0.90 (class 1)

#### Tuned Random Forest (Best Model):
Accuracy: 0.90
Precision: 0.91 (class 0), 0.89 (class 1)
Recall: 0.85 (class 0), 0.93 (class 1)

**Model terbaik:** Tuned Random Forest  
Dipilih karena memiliki akurasi dan F1-score tertinggi.
F1-score: 0.88 (class 0), 0.91 (class 1)

## Kesimpulan
Proyek ini berhasil membangun model prediksi penyakit jantung dengan performa tinggi. Dengan tuning Random Forest, diperoleh akurasi sebesar **90%**. Model ini dapat dijadikan dasar untuk sistem deteksi dini berbasis data kesehatan.

