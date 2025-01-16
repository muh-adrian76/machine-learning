# Laporan Submission Machine Learning Terapan
- **Nama:** Muhammad Adriano Khairur Rizky Setyawan
- **Email:** wedoung87@gmail.com
- **ID Dicoding:** m_adriano_krs_76

## Project Overview
Topik yang dipilih untuk proyek machine learning ini adalah *rekomendasi buku*, dengan judul **Sistem Rekomendasi Manhwa (Komik Korea)**.

### Latar Belakang

![image](https://github.com/user-attachments/assets/6efd4aa8-f4eb-4dbb-b4ce-c8cac06657da)

Komik adalah rangkaian gambar dan kata yang disusun untuk menyampaikan informasi atau cerita kepada pembacanya. Selain dalam bentuk buku, komik kini hadir dalam format digital yang dapat diakses melalui smartphone atau komputer. Komik memiliki beragam alur cerita dan genre, seperti aksi, petualangan, romantis, dan fantasi. Setiap negara juga memiliki gaya komik yang khas, seperti manga dari Jepang, komik dari Barat, dan manhwa dari Korea Selatan. **Manhwa** adalah komik asal Korea Selatan yang biasanya diterbitkan dalam format digital dan sering dibaca dengan cara menggulir vertikal, sehingga lebih nyaman diakses melalui perangkat mobile. Manhwa menawarkan berbagai genre yang menarik, seperti aksi, romansa, fantasi, dan drama, dengan gaya seni yang khas dan alur cerita yang memikat <a href="https://ejournal.bsi.ac.id/ejurnal/index.php/ji/article/view/16113" target="_blank">[1]</a>.

Namun, banyaknya manhwa dengan beragam genre dan alur cerita sering kali membuat pembaca kesulitan dalam menemukan judul yang sesuai dengan preferensi mereka. Permasalahan ini dapat diatasi dengan membangun sistem rekomendasi berbasis *content-based filtering* menggunakan model machine learning <a href="https://ejournal.bsi.ac.id/ejurnal/index.php/ji/article/view/16113" target="_blank">[1]</a>. Sistem ini dapat merekomendasikan manhwa yang relevan dengan preferensi pengguna berdasarkan kesamaan konten, seperti genre, sinopsis, dan komikus. Dengan demikian, pembaca dapat lebih mudah menemukan manhwa yang sesuai dengan minat mereka.

Masalah ini menjadi penting karena tanpa sistem rekomendasi yang efektif, pembaca mungkin merasa kewalahan dengan banyaknya pilihan yang tersedia, yang dapat mengurangi minat mereka untuk menjelajahi judul baru. Di sisi lain, platform penyedia manhwa juga berpotensi kehilangan kesempatan untuk meningkatkan keterlibatan pengguna dan pendapatan dari manhwa yang kurang terekspos. Oleh karena itu, pengembangan sistem rekomendasi yang akurat dan relevan menjadi solusi penting untuk menjawab kebutuhan pembaca dan mendukung pertumbuhan industri manhwa digital.

## 1. Business Understanding

Pengembangan sistem rekomendasi manhwa berbasis *content-based filtering* memiliki potensi besar untuk meningkatkan pengalaman pengguna dalam menjelajahi dan membaca manhwa. Sistem ini dapat membantu pembaca menemukan manhwa yang sesuai dengan preferensi mereka, sehingga meningkatkan kepuasan dan loyalitas pengguna. Selain itu, platform penyedia manhwa juga dapat memanfaatkan sistem ini untuk meningkatkan engagement, memperpanjang waktu penggunaan aplikasi, dan mendorong konsumsi konten yang lebih luas.

### 1.1 Problem Statements

Berdasarkan latar belakang di atas, berikut ini merupakan rincian masalah yang dapat diselesaikan pada proyek ini:
- Bagaimana membangun model *machine learning* yang dapat merekomendasikan manhwa sesuai dengan preferensi pengguna berdasarkan genre anime?
- Bagaimana sistem rekomendasi ini dapat meningkatkan pengalaman membaca pengguna dan mendukung pengembangan industri manhwa digital?  

### 1.2 Goals

Tujuan dari proyek ini adalah:  
- Membangun model *machine learning* berbasis *content-based filtering* yang dapat merekomendasikan manhwa sebanyak Top-N Rekomendasi kepada pengguna berdasarkan genre dan sesuai dengan preferensi pengguna.  
- Mengembangkan sistem atau aplikasi yang dapat membantu pembaca dalam menemukan manhwa yang relevan dengan minat mereka, sehingga meningkatkan kepuasan pengguna dan mendukung pertumbuhan industri manhwa digital.
  
# 2. Data Understanding
**Informasi Dataset**
| Kategori | Keterangan |
| ------ | ------ |
| Judul | Manhwa dataset |
| Usability | 10.00 |
| URL | [Kaggle](https://www.kaggle.com/datasets/crxxom/manhwa-dataset) |
| Baris | 2943 |
| Kolom | 15 |

*Tabel 1. Deskripsi Singkat Dataset* 

| Unnamed: 0 | Type   | Title                     | Chapters | Status    | Genres                        | Favorites | Popularity | Rank | Score | Members | Synopsis                                                       | Volumns | Authors                                      | Publish Time                  |
|------------|--------|---------------------------|----------|-----------|-------------------------------|-----------|------------|------|-------|---------|----------------------------------------------------------------|---------|----------------------------------------------|--------------------------------|
| 0          | Manhwa | Solo Leveling            | 201      | Finished  | Action, Adventure, Fantasy    | 40,014    | #7         | #56  | 8.68  | 431,289 | Ten years ago, "the Gate" appeared and connected two worlds... | Unknown | Chugong (Story), Jang, Sung-rak (Art)        | Mar 4, 2018 to May 31, 2023   |
| 1          | Manhwa | The Horizon              | 21       | Finished  | Adventure, Drama             | 4,047     | #187       | #58  | 8.67  | 75,806  | In a world ravaged by war, a young boy walks down a desolate... | 3       | Jeong, Ji-Hoon (Story & Art)                | Mar 30, 2016 to Jul 21, 2016  |
| 2          | Manhwa | Wind Breaker            | Unknown  | Publishing| Action, Drama, Sports        | 2,688     | #368       | #94  | 8.58  | 42,434  | Burdened with expectations since childhood, se...             | Unknown | Jo, Yongseok (Story & Art)                 | Dec 15, 2013 to ?             |
| 3          | Manhwa | Bastard                 | 94       | Finished  | Drama, Horror, Mystery, Romance | 6,455  | #84        | #140 | 8.50  | 126,088 | There is nowhere that Seon Jin can find solace...              | 5       | Kim, Carnby (Story), Hwang, Young-chan (Art)| Jul 4, 2014 to May 6, 2016   |
| 4          | Manhwa | Who Made Me a Princess  | 125      | Finished  | Comedy, Fantasy, Romance     | 2,648     | #349       | #175 | 8.44  | 44,428  | In the novel *The Lovely Princess*, the secondary...           | 9       | Plutus (Story), Spoon (Art)                 | Dec 20, 2017 to Apr 30, 2022  |

*Tabel 2. 5 Data Pertama dari Dataset Asli*  

**Penjelasan Variabel**:
Berikut adalah penjelasan dari masing-masing variabel atau fitur dalam dataset **Manhwa**:

| **Variabel**   | **Deskripsi**                                                                 |
|----------------|--------------------------------------------------------------------------------|
| **type**       | Jenis komik. Dalam dataset ini seluruh entri merupakan *manhwa* (komik Korea Selatan). |
| **title**      | Judul dari manhwa.                                                            |
| **chapters**   | Jumlah bab (chapter) yang telah diterbitkan untuk masing-masing manhwa.        |
| **status**     | Status publikasi manhwa, apakah sudah selesai (*Finished*) atau masih berlangsung (*Publishing*). |
| **genres**     | Kategori atau jenis cerita dalam manhwa, seperti *Action*, *Romance*, *Fantasy*, dan lainnya. |
| **favorites**  | Jumlah pengguna di MyAnimeList (MAL) yang menambahkan manhwa tersebut ke daftar favorit mereka. |
| **popularity** | Peringkat popularitas manhwa di MyAnimeList berdasarkan minat pembaca.         |
| **rank**       | Peringkat keseluruhan manhwa berdasarkan skor rata-rata yang diberikan pengguna. |
| **score**      | Nilai rata-rata manhwa yang diberikan oleh pengguna MyAnimeList.               |
| **members**    | Jumlah pengguna MyAnimeList yang telah menambahkan manhwa tersebut ke daftar bacaan mereka. |
| **synopsis**   | Ringkasan atau deskripsi singkat mengenai alur cerita dari manhwa.             |
| **volumns**    | Jumlah volume yang telah diterbitkan untuk masing-masing manhwa.               |
| **authors**    | Nama penulis (*story writer*) dan ilustrator (*artist*) dari manhwa tersebut.  |
| **publish_time** | Periode waktu perilisan manhwa, mulai dari tanggal rilis pertama hingga terakhir. |

Variabel **Unnamed: 0** merupakan indexing, sehingga tidak akan digunakan.

*Tabel 3. Penjelasan Variabel* 

### **Jumlah Missing Value pada Dataset Manhwa**

| **Fitur**      | **Jumlah Missing Value** |
|----------------|--------------------------|
| type           | 0                        |
| title          | 0                        |
| chapters       | 1478                     |
| status         | 0                        |
| genres         | 0                        |
| favorites      | 0                        |
| popularity     | 0                        |
| rank           | 332                      |
| score          | 1446                     |
| members        | 0                        |
| synopsis       | 154                      |
| volumns        | 1979                     |
| authors        | 57                       |
| publish_time   | 747                      |

**dtype**: int64

*Tabel 4. Keterangan Missing Values*

Berikut adalah kondisi data dalam dataset yang akan digunakan:
- Terdapat 6 data yang terduplikasi.
- Pada *Tabel 4* terlihat bahwa terdapat *missing value* pada 6 dari 14 total fitur dalam dataset.

# 3. Data Preparation
Pada tahap ini dilakukan proses _Data Gathering_, _Data Assessing_, dan _Data Cleaning_. 

## 3.1 Data Gathering
Pada proses Data Gathering, data diimpor sedemikian rupa agar bisa dibaca dengan baik menggunakan dataframe Pandas. Dataset yang dipakai memiliki 2943 sampel dengan 15 fitur, dimana terdapat 2 kolom numerik dengan tipe data int64 dan float64, serta terdapat 13 kolom kategorik dengan tipe data object yang dapat dilihat menggunakan atribut `shape` dan fungsi `info()`. Namun, terdapat adanya indikasi kesalahan tipe data pada dataset:

1. *chapters* seharusnya `Float64` atau `Int64` karena kolom ini berisi jumlah chapter, tipe data numerik lebih tepat. Namun, ada nilai 'Unknown' yang perlu diatasi sebelum konversi.

2. *favorites, popularity, dan members* seharusnya `Int64` karena menyatakan jumlah berupa angka. Namun, kemungkinan ada format angka atau simbol seperti koma ("40,014") atau ("#") yang perlu dibersihkan.

3. *rank dan volumns* seharusnya `Int64` karena menyatakan peringkat, sehingga lebih tepat jika bertipe numerik. Kemungkinan juga terdapat data 'Unknown' atau kosong yang perlu diatasi.

4. *publish_time* seharusnya `datetime64[ns]` karena
berisi rentang tanggal.  

## 3.2 Data Assessing
Untuk proses Data Assessing, berikut adalah beberapa pengecekan yang dilakukan:
- Menghapus fitur yang tidak diperlukan, dan hanya menyisakan fitur `title` dan `genres`.
- Duplicate data (data yang serupa dengan data lainnya), menggunakan fungsi `duplicated()`.
- Missing value (data atau informasi yang "hilang" atau tidak tersedia), menggunakan fungsi `isnull()`.

Setelah melakukan ketiga proses tersebut, jumlah dataset berkurang menjadi `2723` yang awalnya adalah `2943`.

## 3.3 Data Cleaning
Pada proses _Data Cleaning_ yang dilakukan adalah:
- TF-IDF Vektorisasi, teknik ini digunakan pada sistem rekomendasi untuk menemukan representasi fitur penting dari setiap *genre* manhwa.

# 4. Model Development
Pengembangan model pada proyek ini dilakukan menggunakan metode Cosine Similarity.

*Cosine similarity* adalah metode untuk mengukur seberapa mirip dua vektor dalam ruang multidimensi. Ini adalah pengukuran kosinus sudut antara dua vektor yang dimensi dan magnitudonya direpresentasikan sebagai titik dalam ruang. Nilai similaritas kosinus berkisar antara -1 hingga 1, di mana nilai 1 menunjukkan kedua vektor sepenuhnya sejajar (100% mirip), 0 menunjukkan vektor tegak lurus (tidak ada keterkaitan), dan -1 menunjukkan kedua vektor sepenuhnya berlawanan arah (100% tidak mirip). Metode ini sering digunakan dalam pemrosesan teks dan pengelompokan data untuk menentukan tingkat kesamaan antara dokumen atau fitur dalam dataset.

Cosine Similarity dituliskan dalam rumus: 

$$Cosine Similarity (A, B) = (A · B) / (||A|| * ||B||)$$ 

dimana: 
- (A·B)menyatakan produk titik dari vektor A dan B.
- ||A|| mewakili norma Euclidean (magnitudo) dari vektor A.
- ||B|| mewakili norma Euclidean (magnitudo) dari vektor B.

Kelebihan _Cosine Similarity_:
- Kompleksitas yang rendah, membuatnya efisien dalam perhitungan.
- Cocok digunakan pada dataset dengan dimensi yang besar karena tidak terpengaruh oleh jumlah dimensi.

Kekurangan _Cosine Similarity_:
- Hanya memperhitungkan arah dari vektor, tanpa memperhitungkan magnitudo (besarnya).
- Perbedaan dalam magnitudo vektor tidak sepenuhnya diperhitungkan, yang berarti nilai-nilai yang sangat berbeda dapat dianggap mirip jika arah vektornya sama.

# 5. Model Evaluation
Untuk melakukan pengujian model, digunakan potongan kode berikut.
```python
manhwa_recommendations('White Day', 10)
```

Output dari sistem akan menampilkan 10 daftar manhwa yang berkaitan dengan judul tersebut berdasarkan genre-nya.

| **Title**                         | **Genres** |
|-----------------------------------|------------|
| Crown Princess Project            | Romance    |
| Talking About...                  | Romance    |
| Love Fantasy                      | Romance    |
| Imitation                         | Romance    |
| Secret Playlist                   | Romance    |
| Utopia of Homosexuality           | Romance    |
| At the End of Love and Death      | Romance    |
| Blind Märchen                     | Romance    |
| Sour & Sweet                      | Romance    |
| When the Day Comes                | Romance    |

*Table 5. Hasil Pengujian Model *Content-Based Filtering* (dengan Filter Genres)*.

Berdasarkan *Table 5*, sistem rekomendasi berhasil menampilkan manhwa dengan genre **Romance** yang relevan dengan **White Day**. Ini menunjukkan bahwa model telah mampu memahami preferensi pengguna berdasarkan genre favorit mereka.

## Metrik Precision

Metrik *Precision* digunakan untuk mengevaluasi hasil dari rekomendasi pada tabel 5. _Precision_ dapat didefinisikan sebagai berikut:  

$\text{Precision} = \frac{r}{i}$

- r= total rekomendasi yang relevan
- i= jumlah rekomendasi yang diberikan

Dari hasil rekomendasi di tabel 5, diketahui bahwa manhwa berjudul `White Day` memiliki genre *Romance*. Dari 10 _item_ yang direkomendasikan, 10 _item_ memiliki genre *Romance* (_similar_). Artinya, _precision_ sistem sebesar 10/10 atau 100%.

## **Keterkaitan dengan Problem Statements**

1. **Bagaimana membangun model *machine learning* yang dapat merekomendasikan manhwa sesuai dengan preferensi pengguna berdasarkan genre?**  
   ✔️ Masalah ini terjawab dengan membangun model *content-based filtering* menggunakan *Cosine Similarity*, yang merekomendasikan manhwa berdasarkan kesamaan genre.  

2. **Bagaimana sistem rekomendasi ini dapat meningkatkan pengalaman membaca pengguna dan mendukung pengembangan industri manhwa digital?**  
   ✔️ Dengan memberikan rekomendasi manhwa yang relevan dan sesuai preferensi, pengguna dapat lebih mudah menemukan judul yang diminati. Hal ini meningkatkan pengalaman membaca dan mendorong pertumbuhan industri manhwa digital.  

## **Pencapaian Goals**

1. **Membangun model *machine learning* berbasis *content-based filtering*.**  
   ✔️ Model berhasil dibangun dengan algoritma *Cosine Similarity* untuk mengukur kemiripan berdasarkan genre.  

2. **Mengembangkan sistem rekomendasi yang meningkatkan kepuasan pengguna.**  
   ✔️ Model telah diimplementasikan dan diuji dengan menghasilkan rekomendasi yang relevan dan personal.  

## **Dampak dan Efektivitas Solusi**

- **Rekomendasi yang relevan:** Model memberikan rekomendasi manhwa yang sesuai dengan minat pengguna, meningkatkan kepuasan dalam membaca.  
- **Pengalaman pengguna yang lebih baik:** Pengguna tidak perlu lagi mencari manhwa secara manual, sehingga lebih hemat waktu.  
- **Mendukung industri manhwa digital:** Sistem rekomendasi mendorong pembaca untuk mengeksplorasi lebih banyak manhwa, yang berdampak positif pada industri.  

## **Kesimpulan**

Model *content-based filtering* berbasis *Cosine Similarity* telah berhasil dibangun dan diuji untuk merekomendasikan manhwa sesuai dengan preferensi pengguna. Model ini tidak hanya menjawab *problem statements* yang diajukan, tetapi juga memenuhi seluruh tujuan proyek. Dengan integrasi lebih lanjut ke dalam platform baca manhwa digital, sistem ini diharapkan dapat meningkatkan pengalaman pengguna dan mendukung pertumbuhan industri manhwa.

# 6. Referensi
- [1] Kurniaji, Arba'i & Santi, Rina. (2023). Implementasi Metode Content Based Filtering Pada Pemilihan Komik. Jurnal Informatika. 10. 109-117. <a href="https://ejournal.bsi.ac.id/ejurnal/index.php/ji/article/view/16113" target="_blank">https://doi.org/10.31294/inf.v10i2.16113</a>.
