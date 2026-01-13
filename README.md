# 🌦️ ClearSky-DeepLearning ~ Weather Prediction (CNN + BiLSTM + TCN Conditional Ensemble)

## 📌 Deskripsi Proyek
Proyek ini bertujuan untuk **memprediksi kondisi cuaca 1 jam ke depan (H+1)** dengan menggabungkan:
- **CNN berbasis transfer learning (InceptionV3/MobileNetV2)** untuk klasifikasi gambar awan.
- **Forecasting time series (BiLSTM + TCN)** untuk memanfaatkan data atmosfer (suhu, kelembapan, tekanan, angin, radiasi).
- **Conditional Ensemble** yang menggabungkan CNN dan forecasting dengan bobot berbeda untuk siang/malam.

Pendekatan ini menghasilkan model yang lebih **robust, adaptif, dan akurat** dibandingkan model tunggal.

---

## 🏗️ Arsitektur
### 1. CNN (Cloud Image Classification)
- Backbone: InceptionV3 (pretrained ImageNet).
- Input: gambar awan 224×224 piksel.
- Augmentasi: rotasi, flip, zoom, brightness jitter.
- Output: probabilitas 3 kelas → `Cerah`, `Berawan`, `Hujan`.

### 2. Forecasting (Time Series)
- Input: data 24 jam terakhir (suhu, kelembapan, tekanan, angin, radiasi).
- Model:
  - **BiLSTM** → menangkap dependensi temporal maju & mundur.
  - **TCN** → konvolusi temporal dengan dilasi untuk pola jangka panjang.
- Ensemble: soft voting (rata-rata probabilitas BiLSTM + TCN).

### 3. Conditional Ensemble
- Siang (06–18): CNN lebih dominan (α=0.7).
- Malam: forecasting lebih dominan (α=1.0).
- Output: kombinasi probabilitas adaptif → prediksi final cuaca H+1.

---

## 📂 Struktur Dataset
- **Dataset Gambar Awan**
Dataset
Awan/train/val/test/
  
- **Dataset Time Series**
Dataset/Suhu/Prakiraan/dataset_cuaca_manokwari.csv

---

## ⚙️ Library Utama
- **Deep Learning**: TensorFlow, Keras
- **Preprocessing**: NumPy, Pandas, Scikit-learn
- **Visualisasi**: Matplotlib
- **Utility**: os, json

---

## 🚀 Training & Evaluasi
### CNN
- Akurasi validasi: ~86%
- Akurasi test: ~84%
- Tantangan: kelas *Berawan* sering salah → solusi dengan class weighting/focal loss.

### Forecasting (BiLSTM + TCN)
- Akurasi validasi: ~90%
- Macro F1: ~0.89

### Conditional Ensemble
- Akurasi test: **92.5%**
- Macro F1: **0.91**
- Performa stabil di semua kelas, terutama meningkatkan recall *Berawan* & *Hujan*.

---

## 📊 Hasil Evaluasi
- **Confusion Matrix (Conditional Ensemble H+1)**  

[[124   6   4]   → Berawan
[  2  75   0]   → Cerah
[  6   1  35]]  → Hujan

- **Classification Report**
- Berawan → Precision 0.94, Recall 0.93, F1 0.93
- Cerah   → Precision 0.91, Recall 0.97, F1 0.94
- Hujan   → Precision 0.90, Recall 0.83, F1 0.86
- Accuracy: 92.5% | Macro F1: 0.91

---

## 📌 Kesimpulan
- CNN efektif menangkap pola visual awan, tetapi bias pada kelas mayoritas.
- Forecasting BiLSTM + TCN kuat dalam menangkap dinamika atmosfer.
- Conditional ensemble adaptif siang/malam menghasilkan performa terbaik.
- Model siap digunakan untuk **prakiraan cuaca operasional H+1**.

---
