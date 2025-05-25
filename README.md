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

**Sumber Dataset:**  
[heart.csv](https://raw.githubusercontent.com/Evameivina/heart_ml/refs/heads/main/heart.csv)

**Ukuran Dataset:**
- Jumlah Data: 918 baris
- Fitur: 11 fitur + 1 target (`HeartDisease`)
- Tidak ada nilai yang hilang (missing values)

**Distribusi Target:**
- Positif (1): 55.3%
- Negatif (0): 44.7%
- Data cukup seimbang

**Statistik Deskriptif:**
- Rata-rata usia: 53 tahun
- Rata-rata tekanan darah istirahat: 132 mmHg
- Rata-rata detak jantung maksimum (MaxHR): 136 bpm

**Contoh 5 Data Teratas:**

| Age | Sex | ChestPainType | RestingBP | Cholesterol | FastingBS | RestingECG | MaxHR | ExerciseAngina | Oldpeak | ST_Slope | HeartDisease |
|-----|-----|----------------|-----------|-------------|-----------|-------------|--------|------------------|---------|-----------|----------------|
| 40  | M   | ATA            | 140       | 289         | 0         | Normal      | 172    | N                | 0.0     | Up        | 0              |
| 49  | F   | NAP            | 160       | 180         | 0         | Normal      | 156    | N                | 1.0     | Flat      | 1              |
| 37  | M   | ATA            | 130       | 283         | 0         | ST          | 98     | N                | 0.0     | Up        | 0              |
| 48  | F   | ASY            | 138       | 214         | 0         | Normal      | 108    | Y                | 1.5     | Flat      | 1              |
| 54  | M   | NAP            | 150       | 195         | 0         | Normal      | 122    | N                | 0.0     | Up        | 0              |

**Catatan Kualitas Data:**
- Tidak ditemukan nilai duplikat pada data
- Distribusi kelas seimbang
- Tidak ada nilai kosong (null)
- Tidak ada outlier ekstrem yang memengaruhi distribusi secara signifikan
  
## Data Preparation

### 1. Pemisahan Fitur dan Target
Data dipisahkan menjadi:
- **Fitur (`X`)**: semua kolom kecuali kolom `target`.
- **Target (`y`)**: kolom `target`, yaitu label penyakit jantung (1 = ada, 0 = tidak ada).

### 2. One-Hot Encoding
Fitur kategorikal seperti `cp`, `thal`, dan `slope` diubah menggunakan **one-hot encoding** agar bisa dibaca oleh model machine learning.

### 3. Standardisasi Fitur Numerik
Fitur numerik seperti `age`, `chol`, `trestbps`, `thalach`, dan `oldpeak` dilakukan **scaling menggunakan StandardScaler** agar memiliki distribusi dengan mean = 0 dan standar deviasi = 1. Hal ini penting terutama untuk model seperti **Logistic Regression**.

### 4. Train-Test Split
Data dibagi menjadi:
- **80% data latih (training set)**
- **20% data uji (test set)**  
Pembagian dilakukan secara **stratified** berdasarkan `target` agar proporsi kelas tetap seimbang.


## Modeling

### Algoritma yang Digunakan:
1. **Logistic Regression (Baseline)**
2. **Random Forest Classifier (Default)**
3. **Random Forest dengan Hyperparameter Tuning (GridSearchCV)**

### Penjelasan Model:

#### 1. Logistic Regression
- Model linier untuk klasifikasi biner.
- Parameter penting:
  - `C = 1.0`: regularisasi (default).
  - `solver = 'liblinear'`: solver yang cocok untuk dataset kecil.
- **Kelebihan**: cepat, simpel, mudah diinterpretasikan.
- **Kekurangan**: terbatas dalam menangkap relasi non-linear kompleks.

#### 2. Random Forest Classifier (Default)
- Model ansambel berbasis banyak pohon keputusan.
- Parameter default digunakan, seperti:
  - `n_estimators = 100`
  - `max_depth = None`
- **Kelebihan**: tahan terhadap overfitting, menangkap interaksi antar fitur.
- **Kekurangan**: lebih lambat dibanding Logistic Regression.

#### 3. Tuned Random Forest (GridSearchCV)
- Tuning menggunakan GridSearchCV dengan 5-fold cross-validation.
- Parameter yang dituning:
  - `n_estimators = [100, 150, 200]`
  - `max_depth = [None, 5, 10]`
  - `min_samples_split = [2, 5]`
- Parameter terbaik:
  - `n_estimators = 150`
  - `max_depth = 10`
  - `min_samples_split = 2`
  
## Evaluation## Evaluation

**Accuracy:**  
Persentase prediksi yang benar dari seluruh data.

**Precision:**  
Rasio prediksi positif yang benar:  
$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**Recall:**  
Rasio kasus positif yang berhasil terdeteksi:  
$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**F1-score:**  
Harmonik rata-rata precision dan recall:  
$$
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Confusion Matrix:**  
Matriks yang menampilkan jumlah prediksi benar dan salah untuk masing-masing kelas.

### Hasil Evaluasi

| Model                   | Accuracy | Precision (class 0) | Precision (class 1) | Recall (class 0) | Recall (class 1) | F1-score (class 0) | F1-score (class 1) |
|-------------------------|----------|---------------------|---------------------|------------------|------------------|--------------------|--------------------|
| Logistic Regression      | 0.89     | 0.91                | 0.87                | 0.83             | 0.93             | 0.87               | 0.90               |
| Random Forest (Default)  | 0.89     | 0.89                | 0.89                | 0.87             | 0.91             | 0.88               | 0.90               |
| Tuned Random Forest      | 0.90     | 0.91                | 0.89                | 0.85             | 0.93             | 0.88               | 0.91               |


### Kesimpulan

- Model terbaik adalah **Tuned Random Forest**, karena memiliki nilai **accuracy tertinggi 90%** serta F1-score terbaik untuk kedua kelas.
- Model ini mampu mendeteksi penyakit jantung dengan baik, seimbang antara precision dan recall.
- Dengan hasil ini, model dapat menjadi dasar sistem deteksi dini berbasis data kesehatan yang akurat dan andal.
- Dampak terhadap **Business Understanding**:  
  Model menjawab problem statement utama yaitu prediksi risiko penyakit jantung secara akurat sehingga dapat mendukung pengambilan keputusan medis yang lebih cepat dan tepat. Goals dari proyek seperti meningkatkan deteksi dini dan mengurangi kesalahan diagnosis telah tercapai.

