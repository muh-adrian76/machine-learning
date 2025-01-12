# Laporan Submission Machine Learning Terapan
- **Nama:** Muhammad Adriano Khairur Rizky Setyawan
- **Email:** wedoung87@gmail.com
- **ID Dicoding:** m_adriano_krs_76

## Domain Proyek
Domain yang dipilih untuk proyek machine learning ini adalah *pertanian*, dengan judul **Prediksi Kualitas Pisang**  

### Latar Belakang

![image](https://github.com/user-attachments/assets/1cbe0453-6ab7-48fc-9b43-9c8d2ba65b4d)

Tanaman pisang merupakan salah satu tanaman herba yang terdiri dari akar, batang, daun, dan buah. Tanaman ini termasuk kedalam suku Musaceae, dapat tumbuh baik di dataran rendah maupun di dataran tinggi sehingga tanaman ini banyak ditanam oleh masyarakat sebagai bahan pangan. Pisang termasuk kedalam golongan buah klimakterik dan bersifat mudah rusak setelah dipanen <a href="https://doi.org/10.23960/jat.v11i2.6168" target="_blank">[1]</a>. 
Tingkat kematangan buah pisang saat dipanen sangat mempengaruhi daya simpan dan kualitas buah. Waktu panen sangat penting untuk mendapatkan buah yang matang dan berkualitas <a href="https://jurnal.fp.unila.ac.id/index.php/JAT/article/view/7883" target="_blank">[2]</a>. Buah yang dipanen terlalu muda memiliki daya simpan yang rendah dan kualitas yang kurang baik ketika matang [1]. Umumnya usia panen terbaik dalam penanganan pasca panen untuk memperpanjang umur pisang antara 95 hari sampai dengan 110 hari setelah antesis (HSA) disesuaikan dengan varietasnya masing-masing, dan usia penyimpanan berkisar antara 10-11 hari <a href="http://ejournal.upnjatim.ac.id/index.php/teknologi-pangan/article/view/3903" target="_blank">[3]</a>.

Masalah ini perlu diselesaikan karena kualitas dan daya simpan buah pisang yang buruk dapat menyebabkan kerugian bagi petani, pedagang, dan konsumen. Buah pisang yang dipanen pada tingkat kematangan yang tidak tepat berisiko cepat rusak, menurunkan nilai jual, dan meningkatkan limbah pangan. Oleh karena itu, diperlukan metode yang efektif untuk mengidentifikasi kualitas pisang secara akurat. Pendekatan berbasis machine learning dapat menjadi solusi untuk memprediksi kualitas pisang dengan menganalisis data sensorik, seperti parameter fisik dan kimiawi, sehingga membantu menentukan waktu panen yang optimal dan memperpanjang umur simpan buah.

# 1. Business Understanding

Pengembangan model prediksi kualitas pisang berdasarkan waktu panen dan tingkat kematangan memiliki potensi besar untuk meningkatkan efisiensi dalam distribusi dan pemasaran buah pisang. Model ini dapat membantu petani menentukan waktu panen yang optimal untuk menghasilkan pisang berkualitas tinggi, memperpanjang umur simpan, dan meningkatkan nilai jual di pasar. Selain itu, distributor juga dapat memanfaatkan model ini untuk memastikan kualitas produk yang dipasarkan sesuai dengan standar konsumen.

## 1.1 Problem Statements

Berdasarkan latar belakang di atas, berikut ini merupakan rincian masalah yang dapat diselesaikan pada proyek ini:
- Bagaimana membuat model machine learning yang dapat memprediksi kualitas pisang berdasarkan data sensorik?
- Model seperti apa yang memiliki akurasi paling baik dalam memprediksi kualitas pisang?
- Bagaimana model ini dapat membantu petani dan distributor dalam meningkatkan kualitas dan nilai jual pisang?

## 1.2 Goals

Tujuan dari proyek ini adalah:
- Membuat model machine learning yang dapat memprediksi kualitas pisang berdasarkan data sensorik.
- Membandingkan beberapa algoritma model untuk menemukan akurasi terbaik dalam memprediksi kualitas pisang.
- Mengembangkan aplikasi atau sistem yang dapat membantu petani dan distributor dalam menggunakan model machine learning untuk memprediksi kualitas pisang.

## 1.3 Solution Statements

- Menganalisis data dengan melakukan univariate analysis dan multivariate analysis. Memahami data juga dapat dilakukan dengan visualisasi untuk mengetahui korelasi antar fitur dan mendeteksi outlier.
- Melakukan proses data cleaning dan normalisasi data agar mendapatkan hasil prediksi yang optimal.
- Membuat beberapa variasi model untuk mendapatkan model terbaik dalam memprediksi kualitas pisang. Di antaranya adalah menggunakan:
    * **K-Nearest Neighbor (KNN):** Algoritma sederhana yang mengklasifikasikan data baru berdasarkan kesamaan dengan data terdekat. Cocok untuk mengklasifikasikan tingkat kematangan pisang berdasarkan parameter fisik.
    * **Random Forest:** Algoritma ensemble yang terdiri dari banyak decision tree untuk menghasilkan prediksi yang lebih akurat. Cocok digunakan karena dapat menangani data dengan fitur yang kompleks.
    * **Support Vector Machine (SVM):** Algoritma yang efektif untuk klasifikasi dengan memisahkan data menggunakan hyperplane di ruang fitur. SVM dapat mengklasifikasikan kualitas pisang berdasarkan parameter yang ada.

# 2. Data Understanding
**Informasi Dataset**
| Kategori | Keterangan |
| ------ | ------ |
| Judul | 🍌 Banana Quality |
| Usability | 10.00 |
| URL | [Kaggle](https://www.kaggle.com/datasets/l3llff/banana?select=banana_quality.csv) |
| Baris | 8000 |
| Kolom | 8 |

## 2.1 Exploratory Data Analysis (EDA)
| Size      | Weight    | Sweetness | Softness  | HarvestTime | Ripeness  | Acidity  | Quality |
|-----------|-----------|-----------|-----------|-------------|-----------|----------|---------|
| -1.924968 | 0.468078  | 3.077832  | -1.472177 | 0.294799    | 2.435570  | 0.271290 | Good    |
| -2.409751 | 0.486870  | 0.346921  | -2.495099 | -0.892213   | 2.067549  | 0.307325 | Good    |
| -0.357607 | 1.483176  | 1.568452  | -2.645145 | -0.647267   | 3.090643  | 1.427322 | Good    |
| -0.868524 | 1.566201  | 1.889605  | -1.273761 | -1.006278   | 1.873001  | 0.477862 | Good    |
| 0.651825  | 1.319199  | -0.022459 | -1.209709 | -1.430692   | 1.078345  | 2.812442 | Good    |

*Tabel 1. 5 Data Pertama dari Dataset Asli*  

**Penjelasan Variabel**:
- **Size** : Ukuran buah pisang.  
- **Weight** : Berat buah pisang.  
- **Sweetness** : Tingkat kemanisan buah pisang.  
- **Softness** : Tingkat kelembutan buah pisang.  
- **HarvestTime** : Lama waktu sejak buah dipanen.  
- **Ripeness** : Tingkat kematangan buah pisang.  
- **Acidity** : Tingkat keasaman buah pisang.  
- **Quality** : Kategori kualitas buah pisang (`Good` atau `Bad`).  

### 2.1.1 Univariate Analysis

| Quality | Jumlah Sampel | Persentase (%) |
|---------|---------------|----------------|
| Good    | 4006          | 50.10          |
| Bad     | 3994          | 49.90          |

*Tabel 2. Analisis Univariat (Data Kategori)*

![image](https://github.com/user-attachments/assets/4db48920-a16a-4604-95be-317778583392)

*Gambar 1. Analisis Univariat (Data Numerik)*

Berdasarkan Tabel 2, dapat dilihat bahwa distribusi data katagorik _Quality_ yang terdiri dari _Good_ dan _Bad_ kualitas pisang, yang mana nilai data **Bad** terdiri dari `3994` dan **good** terdiri dari `4006`, yang mana menunjukan perbandingan data yang tidak terlalu jauh dan cukup seimbang. Sedangkan distribusi data numerik pada Gambar 1 memiliki karakteristik terpusat, yaitu distribusi nilai dari semua fitur dominan di tengah (sekitar rata-rata) dengan sedikit outlier di kedua ujung

### 2.1.2 Multivariate Analysis
![image](https://github.com/user-attachments/assets/6cf860e9-7aa5-4a7a-a6e0-ba92a198f730)

*Gambar 2. Analisis Matriks Korelasi (Data Numerik)*

Gambar 2 merupakan matriks korelasi yang menunjukkan hubungan antar fitur numerik dalam nilai korelasi. Dari matriks korelasi, dapat diketahui bahwa:
- Fitur *HarvestTime* memiliki korelasi positif yang cukup kuat dengan *Size*. Artinya, lama waktu panen dapat mempengaruhi ukuran buah pisang. Semakin lama buah tidak dipanen, ukurannya akan semakin besar.
- Fitur *Weight* juga memiliki korelasi positif yang cukup kuat dengan *Acidity* dan *Sweetness*. Artinya, berat pisang dapat mempengaruhi tingkat rasa manis dan asam. Semakin berat buah pisang, maka akan semakin manis dan asam.
- Fitur *Ripeness* memiliki korelasi negatif dengan *Acidity* dan memiliki korelasi positif dengan *Sweetness*. Artinya, saat pisang mencapai kematangan yang optimal, rasa asamnya cenderung berkurang, dan rasa manisnya menjadi lebih dominan.


![image](https://github.com/user-attachments/assets/aaae8328-e515-437d-b7cc-bfff445b693e)
![image](https://github.com/user-attachments/assets/029145f0-c323-4ed2-8046-4ec2c5c2853e)

*Gambar 3. Analisis Multivariat (Data Kategorik)*

Gambar 3 merupakan grafik distribusi *Quality* berdasarkan *HarvestTime* dan *Ripeness*. Karena *HarvestTime* dan *Ripeness* adalah variabel numerik dan Quality bersifat kategorik, untuk itu digunakan boxplot untuk melihat distribusi datanya. Dari kedua grafik boxplot, dapat disimpulkan bahwa:
- Pisang berkualitas baik memiliki waktu panen yang lebih cepat dan stabil dibandingkan dengan pisang berkualitas buruk, yang memiliki waktu panen yang lebih lama dan bervariasi. Hal ini menunjukkan bahwa waktu panen berpengaruh terhadap kualitas pisang, dimana **pisang berkualitas baik umumnya dapat dipanen lebih awal**.
- Adanya nilai outlier pada kategori "Bad" mengindikasikan bahwa ada beberapa panen yang secara signifikan lebih lama, mungkin disebabkan oleh faktor-faktor eksternal seperti cuaca atau metode panen yang kurang efektif.
- Pisang berkualitas baik (Good) memiliki tingkat kematangan yang lebih tinggi dan variasi yang lebih besar dibandingkan dengan pisang berkualitas buruk (Bad). Hal ini menunjukkan bahwa tingkat kematangan pisang berpengaruh terhadap kualitas pisang, di mana **pisang berkualitas baik umumnya lebih matang**.
- Adanya nilai outlier pada kategori "Good" mengindikasikan bahwa pisang berkualitas baik cenderung memiliki tingkat kematangan yang sangat tinggi.

# 3. Data Preparation
Pada tahap ini dilakukan proses _Data Gathering_, _Data Assessing_, dan _Data Cleaning_. 

## 3.1 Data Gathering
Pada proses Data Gathering, data diimpor sedemikian rupa agar bisa dibaca dengan baik menggunakan dataframe Pandas. Dataset yang dipakai memiliki 8000 sampel dengan 8 fitur, dimana 7 fitur bertipe numerik (`float64`) dan 1 fitur bertipe objek (`Quality`) yang dapat dilihat menggunakan atribut `shape` dan fungsi `info()`.  

## 3.2 Data Assessing
Untuk proses Data Assessing, berikut adalah beberapa pengecekan yang dilakukan:
- Duplicate data (data yang serupa dengan data lainnya), menggunakan fungsi `duplicated()`.
- Missing value (data atau informasi yang "hilang" atau tidak tersedia), menggunakan fungsi `isnull()`.
- Outlier (data yang menyimpang dari rata-rata sekumpulan data yang ada), menggunakan grafik boxplot.

| **Fitur**     | **Jumlah Missing Value** |
|---------------|--------------------------|
| Size          | 0                        |
| Weight        | 0                        |
| Sweetness     | 0                        |
| Softness      | 0                        |
| HarvestTime   | 0                        |
| Ripeness      | 0                        |
| Acidity       | 0                        |
| Quality       | 0                        |

**dtype**: int64

*Tabel 3. Keterangan Missing Values*

![image](https://github.com/user-attachments/assets/6261c042-72cd-44e1-b369-c938579b5df9)

*Gambar 4. Visualisasi Nilai Outlier pada Fitur Numerik*

Pada proyek kasus ini tidak ditemukannya data duplikat dan *missing value*. Namun ditemukan nilai *outlier* pada fitur atau variabel numerik. Adapun untuk menghilangkan *outlier* dilakukan dengan menggunakan metode IQR. IQR dihitung dengan mengurangkan kuartil ketiga (Q3) dari kuartil pertama (Q1) sebagaimana rumus berikut.

$$IQR = Q_3 - Q_1$$

*Keterangan*:
- Q1 adalah kuartil pertama 
- Q3 adalah kuartil ketiga.

Setelah menggunakan metode IQR untuk menghilangkan *outlier* pada dataset, jumlah dataset berkurang menjadi `7645` yang awalnya adalah `8000`.

## 3.3 Data Cleaning
Pada proses _Data Cleaning_ yang dilakukan adalah:
- Encoding (mengubah tipe data suatu kolom), menggunakan fungsi `astype()`.
- Train Test Split (membagi data menjadi data latih dan data uji), menggunakan fungsi `train_test_split` pada library *sklearn* dengan proporsi pembagian sebesar 80:20 dan random state sebesar 50, dimana kolom **Quality** dijadikan sebagai **label/target** dan kolom lainnya dijadikan sebagai **Fitur** untuk memprediksi label.
- Normalization (mentransformasi data ke dalam skala yang seragam sehingga semua fitur atau variabel memiliki rentang nilai yang sebanding), menggunakan fungsi `MinMaxScaler` pada library *sklearn*.

# 4. Model Development
Pengembangan model pada proyek ini dilakukan menggunakan 3 algoritma, yaitu:

## 4.1 *K-Nearest Neighbors (KNN)*

*KNN* adalah algoritma machine learning yang sederhana dan mudah dipahami untuk tugas klasifikasi dan regresi. Algoritma ini bekerja dengan mencari *k* tetangga terdekat dari data baru dan menggunakan kategori atau nilai rata-rata dari tetangga tersebut untuk memprediksi kategori atau nilai data baru. Proses pelatihan algoritma ini dapat dilakukan dengan fungsi `KNeighborsClassifier()` dari library *sklearn*, dan pengujian menggunakan fungsi `predict()`. 

### Parameter yang Digunakan:
- `n_neighbors=5`: Parameter ini menentukan jumlah *neighbors* atau tetangga terdekat yang digunakan dalam klasifikasi. Pemilihan nilai *k* sebesar 5 bertujuan untuk menjaga keseimbangan antara bias dan variansi.
   - Nilai k yang terlalu kecil (misal 1 atau 2) membuat model terlalu sensitif terhadap noise (overfitting).
   - Nilai k yang terlalu besar dapat membuat model menjadi terlalu sederhana (underfitting) karena prediksi dipengaruhi terlalu banyak tetangga.
   - Nilai 5 dianggap optimal untuk mendapatkan hasil prediksi yang stabil dan akurat berdasarkan hasil eksperimen awal.
- `weights='distance'`: Parameter ini menentukan jenis pembobotan pada tetangga. Penggunaan weight berbasis jarak (distance) memberikan bobot lebih besar pada tetangga yang lebih dekat dengan data uji, sehingga tetangga yang lebih dekat kemungkinan besar lebih relevan dalam menentukan kelas. Dengan distance weighting, prediksi menjadi lebih akurat karena pengaruh tetangga jauh diminimalkan.

### Keunggulan *KNN*:
- Dapat digunakan untuk klasifikasi dan regresi.
- Sederhana dan mudah diimplementasikan.
- Tidak memerlukan proses pelatihan yang kompleks.

### Kelemahan *KNN*:
- Sensitif terhadap *outlier*.
- Membutuhkan memori dan waktu komputasi besar untuk dataset yang besar.
- Sulit menentukan nilai *k* yang optimal.

---

## 4.2 *Random Forest*

*Random Forest* adalah algoritma machine learning berbasis ensemble yang menggabungkan banyak decision tree untuk meningkatkan akurasi prediksi. Algoritma ini membuat banyak decision tree secara acak dan menggunakan metode voting untuk menentukan hasil prediksi. Proses pelatihan dapat dilakukan menggunakan fungsi `RandomForestClassifier()` dari library *sklearn*, dan pengujian dengan fungsi `predict()`.

Parameter yang Digunakan:
- `max_depth=20`: Parameter ini menentukan kedalaman maksimum dari setiap decision tree, yang membantu mengontrol kompleksitas model dan mencegah overfitting. Nilai 20 dipilih karena mampu menangkap pola kompleks pada data tanpa terlalu kompleks, menjaga keseimbangan antara bias dan variansi.
   - Nilai *depth* yang terlalu besar dapat menyebabkan overfitting karena model menangkap noise data.
   - Sebaliknya, *depth* yang terlalu kecil dapat menyebabkan underfitting, di mana model tidak mampu menangkap pola yang kompleks.

Keunggulan *Random Forest*:
- Memiliki akurasi prediksi yang tinggi.
- Mampu menangani dataset dengan dimensi tinggi.
- Tidak sensitif terhadap *outlier*.
- Lebih tahan terhadap overfitting dibandingkan decision tree tunggal.

Kelemahan *Random Forest*:
- Cenderung overfit pada dataset yang kecil jika tidak diatur dengan baik.
- Memerlukan sumber daya komputasi yang lebih besar.
- Sulit diinterpretasikan karena kompleksitas model.

## 4.3 *Support Vector Machine (SVM)*
*SVM* merupakan algoritma machine learning yang digunakan untuk klasifikasi dan regresi. Algoritma ini bekerja dengan mencari hyperplane yang memisahkan data menjadi dua kelas dengan margin terbesar. Proses pelatihan algoritma ini dapat diterapkan dengan menggunakan fungsi `SVC()` pada library *sklearn*, sedangkan proses pengujiannya dapat menggunakan fungsi `predict()`. Parameter yang digunakan pada pembuatan model adalah parameter bawaan.
 
Keunggulan  *Support Vector Machine (SVM)* :
- Memiliki akurasi prediksi yang tinggi.
- Mampu menangani dataset dengan dimensi tinggi.
- Dapat digunakan untuk klasifikasi dan regresi.

Kerugian  *Support Vector Machine (SVM)* :
- Sensitif terhadap outlier. 

# 5. Model Evaluation
Dalam tahap evaluasi, dilakukan pengujian data testing dan perhitungan metrik `Accuracy`. Metrik ini didapatkan dengan menghitung jumlah prediksi yang benar dibagi dengan jumlah seluruh prediksi. Metrik ini didefinisikan menggunakan persamaan berikut:

$$\text{Accuracy} = \frac{\text{TP + TN}}{\text{TN + TP + FN + FP}}%$$

*Keterangan*:
- TP (True Positive): Jumlah data positif yang diprediksi dengan benar sebagai positif.
- TN (True Negative): Jumlah data negatif yang diprediksi dengan benar sebagai negatif.
- FP (False Positive): Jumlah data negatif yang diprediksi secara tidak benar sebagai positif (Kesalahan Tipe I).
- FN (False Negative): Jumlah data positif yang diprediksi secara tidak benar sebagai negatif (Kesalahan Tipe II).

Berikut hasil `Accuracy` dari ketiga buah model yang dilatih:

| Model | Akurasi | Waktu Prediksi (s) |
| ------ | ------ | ------ |
| KNN | 0.98 | 0.05 |
| RandomForest  | 0.97 | 0.11 |
| SVM | 0.98 | 0.06 |

*Tabel 4. Hasil Accuracy dan Waktu Prediksi*

![image](https://github.com/user-attachments/assets/7c1dd26d-45d5-4ae5-a2d4-65b2ffdd82de)

*Gambar 5. Visualisasi Perbandingan Akurasi dan Waktu Prediksi*

Tabel 4 merupakan data hasil perbandingan akurasi dari setiap model yang diuji. Dapat diketahui bahwa model dengan algoritma *KNN* dan *SVM* sama-sama memiliki Accuracy tertinggi, yaitu `98%`. Untuk itu, pemilihan model terbaik ditentukan oleh lama waktu prediksi. Oleh karena itu, model KNN yang akan dipilih untuk digunakan.

### **Keterkaitan dengan Problem Statements**

1. **Bagaimana membuat model machine learning yang dapat memprediksi kualitas pisang berdasarkan data sensorik?**  
   ✔️ Masalah ini telah terjawab dengan pembuatan dan pengujian model *KNN*, *Random Forest*, dan *SVM* menggunakan data sensorik seperti ukuran, berat, tingkat kemanisan, kelembutan, waktu panen, kematangan, dan keasaman.  
   
2. **Model seperti apa yang memiliki akurasi paling baik dalam memprediksi kualitas pisang?**  
   ✔️ Hasil evaluasi menunjukkan bahwa model *KNN* dan *SVM* mencapai akurasi tertinggi sebesar **98%**, sedangkan *Random Forest* mencapai **97%**. Namun, *KNN* dipilih karena memiliki waktu prediksi yang lebih cepat (**0,05 detik**).  

3. **Bagaimana model ini dapat membantu petani dan distributor dalam meningkatkan kualitas dan nilai jual pisang?**  
   ✔️ Model *KNN* yang akurat dan efisien dapat digunakan dalam aplikasi atau alat monitoring kualitas pisang secara otomatis. Ini membantu petani dan distributor dalam:  
   - **Mengurangi risiko kesalahan penilaian kualitas** secara manual.  
   - **Meningkatkan efisiensi proses sortir** dan distribusi pisang berkualitas tinggi.  
   - **Memaksimalkan nilai jual** dengan memastikan hanya pisang berkualitas baik yang dipasarkan.

### **Pencapaian Goals**

1. **Membuat model machine learning untuk memprediksi kualitas pisang berdasarkan data sensorik.**  
   ✔️ Berhasil dilakukan dengan model *KNN*, *Random Forest*, dan *SVM*.  

2. **Membandingkan algoritma untuk mendapatkan model dengan akurasi terbaik.**  
   ✔️ Evaluasi menunjukkan *KNN* dan *SVM* memiliki akurasi tertinggi (**98%**), dan *KNN* dipilih karena waktu prediksinya lebih cepat.  

3. **Mengembangkan sistem yang membantu petani dan distributor.**  
   ✔️ Model *KNN* siap diintegrasikan ke dalam aplikasi yang dapat diakses petani dan distributor untuk prediksi kualitas pisang secara real-time.  

### **Dampak dan Efektivitas Solusi**

- **Akurasi Tinggi**: Dengan akurasi **98%**, model dapat secara andal memprediksi kualitas pisang, mengurangi risiko kesalahan distribusi produk berkualitas rendah.  
- **Efisiensi Waktu**: *KNN* memiliki waktu prediksi tercepat (**0,05 detik**), mendukung proses sortir yang lebih cepat.  
- **Pengambilan Keputusan yang Lebih Baik**: Model ini membantu petani dan distributor dalam pengambilan keputusan berbasis data, meningkatkan keuntungan dan meminimalkan kerugian.

### **Kesimpulan**

Model *KNN* yang dikembangkan telah menjawab seluruh *problem statements*, mencapai semua *goals*, dan memberikan solusi yang berdampak nyata bagi petani dan distributor. Dengan integrasi model ini ke dalam sistem atau aplikasi, diharapkan dapat meningkatkan kualitas produk, efisiensi operasional, dan nilai jual pisang di pasar.

# 6. Referensi
- [1] Widodo, S. E., Waluyo, S., Karyanto, A., Zulferiyeni, Z., Febrianingrum, N., Latansya, R., & Putri, M. D. (2023). APLIKASI THERMAL IMAGE PENDETEKSI TINGKAT KEMATANGAN BUAH PISANG DAN APOKAT. Jurnal Agrotek Tropika, 11(2), 165. <a href="https://doi.org/10.23960/jat.v11i2.6168" target="_blank">https://doi.org/10.29244/agrob.7.2.162-171</a>
- [2] Irhamni, D., Hayati, R., & Hasanuddin, H. (2023). Pengaruh Tingkat Kematangan dan Lama Penyimpanan terhadap Kualitas Pisang Mas (Musa acuminata Colla). JURNAL AGROTROPIKA, 22(2), 145. <a href="https://jurnal.fp.unila.ac.id/index.php/JAT/article/view/7883" target="_blank">https://doi.org/10.23960/ja.v22i2.7883</a>
- [3] Safitri, D. S., Arti, I. M., Miska, M. E. E., & Kalsum, U. (2023). KARAKTERISTIK BUAH PISANG MAS KIRANA PADA BERBAGAI UMUR PANEN DAN TEKNIK PENYIMPANAN. JURNAL TEKNOLOGI PANGAN, 17(2), 70–82. <a href="http://ejournal.upnjatim.ac.id/index.php/teknologi-pangan/article/view/3903" target="_blank">https://doi.org/10.33005/jtp.v17i2.3903</a>
