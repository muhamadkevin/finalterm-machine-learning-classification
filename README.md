# IEEE-CIS Fraud Detection - Deep Learning (PyTorch MLP)

Sistem deteksi transaksi penipuan (fraud) berbasis **Deep Learning** menggunakan data tabular berskala besar. Proyek ini dibangun sebagai bagian dari Ujian Tengah Semester (UTS) Machine Learning, mengimplementasikan **Multi-Layer Perceptron (MLP)** dengan PyTorch, optimasi hyperparameter dengan **Optuna**, dan tracking eksperimen menggunakan **MLflow**.

---

## 1. Exploratory Data Analysis (EDA) & Feature Engineering

Dataset yang digunakan adalah `train_transaction.csv` dari kompetisi IEEE-CIS Fraud Detection. Dataset asli sangat masif dan berdimensi tinggi:
* **Jumlah Data**: 590.540 baris (transaksi)
* **Jumlah Fitur Awal**: 394 kolom

### Analisis Class Imbalance
Distribusi target variabel (`isFraud`) menunjukkan bahwa dataset ini **sangat tidak seimbang (*highly imbalanced*)**:
* **Kelas 0 (Bukan Fraud)**: 569.877 transaksi (96,5%)
* **Kelas 1 (Fraud)**: 20.663 transaksi (3,5%)

### Strategi Reduksi Dimensi & Feature Engineering
Melalui analisis korelasi, ditemukan bahwa banyak kolom (terutama grup C, D, dan Vesta/V) memiliki korelasi linier yang sangat tinggi satu sama lain dan banyak *missing value*. Untuk menyederhanakan ruang fitur (*feature space*) tanpa kehilangan banyak informasi, dilakukan agregasi:

1. **Drop Fitur Sparse**: Kolom seperti `dist1` dan `dist2` di-drop karena terlalu banyak data kosong.
2. **Agregasi Grup C**: 14 fitur `C1-C14` dirata-ratakan menjadi satu fitur baru `C_avg`.
3. **Agregasi Grup D**: 15 fitur `D1-D15` dirata-ratakan menjadi fitur baru `D_avg`.
4. **Agregasi Grup V**: 339 fitur Vesta dirangkum menjadi 4 fitur rata-rata berdasarkan kelompoknya (`V_001_100_avg`, `V_101_200_avg`, `V_201_300_avg`, `V_301_339_avg`).
5. **Drop ID**: `TransactionID` dibuang karena tidak memiliki nilai prediktif.

**Hasil**: Dimensi dataset berhasil direduksi secara drastis dari **394 kolom menjadi 31 kolom**, mempercepat proses training dan mencegah *curse of dimensionality* pada model Neural Network.

---

## 2. Gambaran Umum Model MLP (Multi-Layer Perceptron)

Model utama yang digunakan adalah **MLP (Multi-Layer Perceptron)**, arsitektur dasar dari jaringan saraf tiruan (*feedforward neural network*). Alasan dan konfigurasi penggunaannya:
* **Cocok untuk Klasifikasi Non-Linier**: MLP dapat mempelajari pola yang sangat kompleks dari kombinasi fitur numerik dan kategorikal.
* **BatchNorm1d**: Setiap *hidden layer* dilewati pada *Batch Normalization* untuk menstabilkan dan mempercepat konvergensi gradien selama proses backpropagation.
* **ReLU (Rectified Linear Unit)**: Digunakan sebagai fungsi aktivasi karena efisien secara komputasi dan mengurangi risiko *vanishing gradient*.
* **Penanganan Imbalance (Dual Strategy)**:
  1. **SMOTE**: Mensintesis sampel dari kelas fraud (minoritas) di level data training.
  2. **BCEWithLogitsLoss (`pos_weight`)**: Model memberikan denda (*loss*) yang jauh lebih besar ketika salah menebak kelas fraud.

---

## 3. Eksekusi Model & Pipeline Scikit-Learn

Agar pemrosesan data rapi dan terstandarisasi sebelum masuk ke PyTorch, digunakan arsitektur **Pipeline**:
1. **Numeric Pipeline**: `SimpleImputer(strategy='median')` -> `StandardScaler()`. Scaling sangat krusial bagi MLP agar *weight updates* merata.
2. **Categorical Pipeline**: `SimpleImputer(strategy='constant', fill_value='Unknown')` -> `OneHotEncoder(handle_unknown='ignore')`.
3. **DataLoader (PyTorch)**: Data yang telah diproses diubah ke PyTorch Tensor dan dimuat dengan sistem Batch (256/2048) untuk mencegah *Out of Memory* (OOM) pada GPU (CUDA).

---

## 4. Hyperparameter Tuning dengan Optuna

Untuk mencari arsitektur terbaik, dilakukan pencarian otomatis selama 30 percobaan (*trials*) menggunakan **Optuna** (dengan algoritma pemangkasan otomatis untuk *trial* yang buruk).

### Parameter Final Terbaik & Penjelasannya
Berikut adalah parameter optimal yang ditemukan dan digunakan pada model akhir:

| Parameter | Nilai Final | Penjelasan & Arti Parameter |
|-----------|-------------|-----------------------------|
| **`optimizer`** | **Adam** | Algoritma pembaruan bobot jaringan. Adam (*Adaptive Moment Estimation*) terpilih karena efisien menangani data yang berisik dan *sparse*. |
| **`learning_rate`** | **0.00159** | Kecepatan model dalam belajar (*step size*). Nilai ~0.0015 cukup ideal agar *loss* turun stabil tanpa melompati titik minimum global. |
| **`n_layers`** | **3** | Jumlah lapisan tersembunyi (*hidden layers*). 3 layer membuktikan bahwa pola fraud pada data ini butuh tingkat abstraksi menengah ke dalam. |
| **`n_units_l0`** | **230** | Jumlah neuron pada hidden layer ke-1. (Layer penangkap pola terbesar dari input). |
| **`n_units_l1`** | **158** | Jumlah neuron pada hidden layer ke-2. (Fase kompresi representasi fitur). |
| **`n_units_l2`** | **181** | Jumlah neuron pada hidden layer ke-3. (Ekstraksi akhir sebelum layer output klasifikasi). |
| **`dropout_rate`** | **0.137** | Peluang 13,7% menonaktifkan neuron secara acak saat training. Berfungsi murni untuk **mencegah overfitting** (model menghafal data train). |
| **`smote_ratio`** | **0.156** | Mengatur agar setelah oversampling, jumlah data fraud menjadi sekitar 15,6% dari total data mayoritas. Tidak dibuat 50-50 agar model tidak terlalu sering menebak *False Positive*. |

*(Model final ini dan preprocesssornya juga disave dan di-track secara otomatis ke dalam **MLflow** agar siap digunakan untuk produksi).*

---

## 5. Hasil Evaluasi Model

Karena dataset imbalanced, *Accuracy* bukanlah metrik utama. Evaluasi dilakukan menggunakan **Threshold Tuning** (Threshold optimal ditemukan pada probabilitas **0.9855**) dengan hasil:

* **AUC-ROC Score**: **0.9328** (Kemampuan model membedakan kelas Fraud dan Non-Fraud sangat baik).
* **F1-Score (Fraud)**: **0.5415** (Keseimbangan yang sangat baik antara *Precision* 0.57 dan *Recall* 0.51 pada threshold tinggi).
* **Overfitting Gap**: Sangat rendah. AUC Train (0.9737) vs AUC Test (0.9328), membuktikan parameter *Dropout* bekerja efektif pada unseen data.

### Top Feature Importance (Metode Permutasi)
Karena MLP adalah *Black-box*, dilakukan metode pengacakan (Permutasi) untuk mengecek fitur terpenting. 3 Fitur teratas yang mengindikasikan fraud adalah:
1. `V_101_200_avg` (Pola agregasi dari grup Vesta)
2. `TransactionAmt` (Nominal uang yang ditransaksikan)
3. `TransactionDT` (Pola waktu transaksi dilakukan)
