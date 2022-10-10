# Laporan Proyek Machine Learning - Cheliza Sriayu Simarsoit

## Daftar Isi

- [Domain Proyek](#domain-proyek)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Referensi](#referensi)

## Domain Proyek

Pada proyek ini akan dibahas mengenai permasalahan dalam bidang keuangan, investasi, *arts and entertainment* untuk mengetahui prediksi harga *gemstone* atau batu permata berdasarkan data fitur-fitur yang terdapat pada batu permata tersebut, seperti nilai *carat*, *cut*, *color*, *clarity*, *depth*, *table*, dan ukuran batu permata tersebut. Data batu permata ini dikumpulkan lebih dari 27.000 baris data batu permata *cubic zirconia* atau batu zirkonia.

![cubic-zirconia](https://user-images.githubusercontent.com/77439245/194769780-4580109c-c83a-4045-9a84-5bae9e8d9aaa.jpg)  
**Gambar 1. Ilustrasi Batu Permata (*Gemstone*) Cubic Zirconia**

Investasi merupakan sebuah hal yang penting untuk menjadi jaminan di masa depan. Bagi orang yang sudah *familiar* dengan investasi, pasti sudah tidak asing dengan produk-produk investasi yang dapat dilakukan, mulai dari investasi berupa surat-surat berharga, aset fisik baik yang bergerak maupun tidak bergerak, contohnya seperti properti, tanah, perhiasan, dan sebagainya.

Dilansir dari Finansialku.com, terdapat beberapa manfaat dan keuntungan dalam berinvestasi batu permata, yaitu sebagai alternatif investasi yang aman jika terjadi ketidakstabilan perekonomian seperti nilai uang yang umumnya akan semakin menurun. Selain itu, investasi batu permata juga tahan terhadap inflasi perekonomian seperti kurs mata uang yang cenderung semakin lemah. Batu permata dapat menjadi instrumen berinvestasi yang cukup aman. Tidak hanya itu, ketika membeli batu permata akan mendapatkan sertifikat GIA yang menjadi perlindungan kelayakan batu permata itu sendiri. Kemudian, harga batu permata juga cenderung akan selalu meningkat dari tahun ke tahun dikarenakan pengaruh permintaan pasar yang terus naik. [[1]](https://www.finansialku.com/prospek-investasi-berlian-di-masa-depan)

Harga dari batu permata itu sendiri tidak hanya ditentukan dari harga permintaan pasar, tetapi berdasarkan kualitas batu permata itu sendiri. Investasi batu permata juga tidak diperlukan biaya perawatan tambahan karena karakteristik dari batu permata itu sendiri yang keras, tidak mudah dihancurkan, dan tahan lama. Selain itu, dalam penyimpanan batu permata ini juga mudah dan tidak memakan ruang yang besar, serta batu permata juga dapat dijadikan sebagai alat perhiasan selain menjadi alat investasi. [[2]](https://www.cermati.com/artikel/segera-beralih-dari-emas-ini-dia-alasan-kenapa-investasi-berlian-lebih-menguntungkan)

Adapun cara ataupun syarat dalam menentukan harga batu permata yang ingin dibeli seperti berat atau nilai karat dari batu permata, tampilan *cutting* proporsi *crown* yang simetris, tingkat kejernihan atau *clarity* dari batu permata tersebut, serta tingkat keputihan atau *color* yang baik. [[1]](https://www.finansialku.com/prospek-investasi-berlian-di-masa-depan)[[2]](https://www.cermati.com/artikel/segera-beralih-dari-emas-ini-dia-alasan-kenapa-investasi-berlian-lebih-menguntungkan) Berdasarkan syarat-syarat di atas, terdapat beberapa *tips* dalam memilih investasi batu permata seperti memilih batu permata yang utuh, model batu permata klasik untuk menjaga nilai jual batu permata itu sendiri, serta memperhatikan 4C yaitu *cut*, *clarity*, *carat*, dan *color*. [[2]](https://www.cermati.com/artikel/segera-beralih-dari-emas-ini-dia-alasan-kenapa-investasi-berlian-lebih-menguntungkan)

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, maka diperoleh rumusan masalah pada proyek ini, yaitu:
1. Bagaimana cara melakukan tahap persiapan data sebelum digunakan untuk pelatihan model *machine learning*?
2. Bagaimana cara membuat model *machine learning* untuk memprediksi harga dari batu permata?

### Goals

Berdasarkan rumusan masalah di atas, maka diperoleh tujuan dari proyek ini, yaitu:
1. Untuk melakukan tahap persiapan data atau *data preparation*, agar data yang digunakan dapat dipakai untuk melatih model *machine learning*.
2. Untuk membuat model *machine learning* dalam memprediksi harga dari batu permata dengan tingkat *error* model *machine learning* yang cukup rendah.

### Solution Statements

Berdasarkan rumusan masalah dan tujuan di atas, maka disimpulkan beberapa solusi yang dapat dilakukan untuk mencapai tujuan dari proyek ini, yaitu:
1. Tahap persiapan data atau *data preparation* dilakukan dengan menggunakan beberapa teknik persiapan data, yaitu:
   - Melakukan proses *encoding* fitur kategori dataset, yaitu kategori *cut*, *color*, dan *clarity* menggunakan *one-hot-encoding*, sehingga diperoleh fitur baru yang mewakili masing-masing variabel kategori.
   - Melakukan proses reduksi dimensi atau *dimension reduction* menggunakan *Principal Component Analysis* (PCA) untuk mengurangi jumlah fitur numerik dengan tetap mempertahankan informasi pada data, sehingga diperoleh sebuah fitur baru yang merupakan hasil dari beberapa fitur numerik.
   - Melakukan proses membagi dataset menjadi data latih dan data uji dengan perbandingan 90 : 10 dari total seluruh dataset yang akan digunakan saat membuat model *machine learning*.
   - Melakukan proses standarisasi fitur numerik menjadi bentuk data yang lebih mudah dipahami dan diolah oleh model *machine learning*.

2. Tahap membuat model *machine learning* untuk memprediksi harga batu permata dilakukan menggunakan model *machine learning* dengan 3 algoritma yang berbeda dan kemudian akan dilakukan evaluasi model untuk membandingkan performa model yang terbaik. Algoritma yang akan digunakan, yaitu Algoritma K-Nearest Neighbor, Algoritma Random Forest, dan Boosting Algorithm.

   - **Algoritma K-Nearest Neighbor**
   
     Algoritma K-Nearest Neighbor, atau biasa disingkat dengan algoritma KNN, merupakan algoritma klasifikasi yang bekerja dengan mengambil sejumlah k data terdekat atau tetangganya untuk digunakan sebagai acuan dalam menentukan kelas data baru. [[3]](https://ilmudatapy.com/algoritma-k-nearest-neighbor-knn-untuk-klasifikasi) Algoritma ini melakukan klasifikasi data berdasarkan kemiripan atau *similarity* ataupun seberapa dekatnya suatu data terhadap data lainnya.
    
     Cara kerja algoritma K-Nearest Negihbor, yaitu: [[3]](https://ilmudatapy.com/algoritma-k-nearest-neighbor-knn-untuk-klasifikasi)
     - Tentukan jumlah tetangga (k) terdekat yang akan digunakan dalam pertimbangan penentuan kelas atau klasifikasi.
     - Hitung jarak dari data baru ke masing-masing *data point* di dataset.
     - Ambil sejumlah k data dengan jarak terdekat, lalu tentukan kelas dari data baru tersebut.
     
     ![knn](https://user-images.githubusercontent.com/77439245/194770422-d01f1540-1baf-49fd-9bef-8de6708bf445.png)  
     **Gambar 2. Ilustrasi algoritma K-Nearest Neighbor**
     
     Berdasarkan gambar di atas, terdapat sejumlah *data point* yang dibagi menjadi dua kelas, yaitu kelas A (biru) dan kelas B (kuning). Misalkan ada data baru (hitam) yang akan diprediksi kelasnya menggunakan algoritma K-Nearest Neighbor (KNN). Nilai k yang digunakan adalah 3. Setelah perhitungan jarak antar titik hitam ke masing-masing *data point* lainnya, diperoleh 3 titik terdekat yang terdiri dari 2 titik kuning (kelas B) dan 1 titik biru (kelas A) di dalam kotak merah. Sehingga kelas untuk data baru (titik hitam) tersebut masuk ke dalam kelas B (kuning).
     
     Terdapat beberapa teknik untuk menghitung jarak suatu data ke data tetangga terdekat lain menggunakan metrik, yaitu:
     - Euclidean Distance
       $$d(a,b)=\sqrt{\sum_{i=1}^n (a_i-b_i)^2}$$
     - Manhattan Distance
       $$d(a,b)=\sum_{i=1}^n |a_i-b_i|$$
     - Hamming Distance
       $$d(a,b)=\frac{1}{n}\sum_{n=1}^{n=n} |a_i-b_i|$$
     - Minkowski Distance
       $$d(a,b)=\left(\sum_{i=1}^n |a_i-b_i|^p\right)^\frac{1}{p}$$

   - **Algoritma Random Forest**
   
     Algoritma Random Forest merupakan algoritma *machine learning* yang menggabungkan *output* atau keluarannya dari beberapa *decision tree* untuk mencapai suatu hasil. Algoritma ini dibentuk dari banyak *tree* atau pohon yang diperoleh melalui proses *bagging* atau *bootstrap aggregating*, di mana akan terdapat beberapa model yang dilatih dengan cara *random sampling with replacement*. [[4]](https://www.trivusi.web.id/2022/08/algoritma-random-forest.html) Prediksi masing-masing *tree* nantinya akan digabungkan sehingga diperoleh akurasi yang lebih tinggi dan juga mencegah masalah overfitting.
     
     ![random-forest](https://user-images.githubusercontent.com/77439245/194770071-8d4e710a-718b-46cd-a196-59df6757ec74.jpg)  
     **Gambar 3. Ilustrasi algoritma Random Forest**
     
     Cara kerja algoritma Random Forest, yaitu: [[4]](https://www.trivusi.web.id/2022/08/algoritma-random-forest.html)
     - Algoritma memilih sampel acak dari dataset yang disediakan.
     - Membuat *decision tree* untuk setiap sampel yang dipilih, lalu didapatkan hasil prediksi dari setiap *decision tree* yang telah dibuat.
     - Melakukan proses voting untuk setiap hasil prediksi, baik menggunakan nilai modus (nilai yang paling sering muncul) untuk kasus klasifikasi, maupun nilai *mean* (nilai rata-rata) untuk kasus regresi
     - Algoritma memilih hasil prediksi yang paling banyak dipilih (*vote* terbanyak) sebagai hasil prediksi akhir.

   - **Boosting Algorithm**
   
     Algoritma Boosting yang digunakan dalam proyek ini adalah algoritma Adaptive Boosting atau biasa disingkat dengan AdaBoost, merupakan algoritma *ensemble* yang memanfaatkan *bagging* dan *boosting* untuk mengembangkan peningkatan akurasi prediksi model *machine learning* yang dibangun. [[5]](https://www.dqlab.id/algoritma-machine-learning-yang-perlu-dipelajari)
     
     ![boosting](https://user-images.githubusercontent.com/77439245/194770091-1f3d0d34-c068-4a95-9cf8-65c5ce999399.jpg)  
     **Gambar 4. Ilustrasi Boosting Algorithm**
     
     Algoritma ini menggunakan beberapa pohon keputusan untuk mendapatkan data prediksi secara berurutan dan prosesnya iteratif. Data latih akan diberikan bobot yang sama untuk kemudian dilakukan pemeriksaan, dan bobot yang lebih tinggi akan masuk ke model yang salah sehingga akan lanjut ke tahap selanjutnya secara berulang hingga tingkat akurasi yang diinginkan.

## Data Understanding

![gemstone-dataset](https://user-images.githubusercontent.com/77439245/194770177-88b4044c-1911-48fb-919e-e29f4044c12b.png)  
**Gambar 5. Kaggle Dataset Gemstone Price Prediction**

Dataset yang digunakan dalam proyek ini adalah dataset [Gemstone Price Prediction](https://www.kaggle.com/datasets/colearninglounge/gemstone-price-prediction?select=cubic_zirconia.csv) yang diambil dari platform Kaggle. *File* yang digunakan berupa *file* csv, yaitu `cubic_zirconia.csv`.

Dari dataset tersebut, dilakukan penghapusan kolom pertama yaitu Unnamed yang berisikan nomor masing-masing data.

Kemudian dilakukan proses *Exploratory Data Analysis* (EDA) yang merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data.

1. **Deskripsi Variabel**

   Pada deskripsi variabel dilakukan pengecekan informasi variabel dari dataset yaitu jumlah kolom, nama kolom, jumlah data per kolom dan tipe datanya.
   
   Berikut adalah informasi variabel dari dataset Gemstone Price Prediction:
   
   **Tabel 1. Deskripsi variabel**
   | # | Column  | Non-Null Count | Dtype   |
   |---|---------|----------------|---------|
   | 0 | carat   | 26967 non-null | float64 |
   | 1 | cut     | 26967 non-null | object  |
   | 2 | color   | 26967 non-null | object  |
   | 3 | clarity | 26967 non-null | object  |
   | 4 | depth   | 26967 non-null | float64 |
   | 5 | table   | 26967 non-null | float64 |
   | 6 | x       | 26967 non-null | float64 |
   | 7 | y       | 26967 non-null | float64 |
   | 8 | z       | 26967 non-null | float64 |
   | 9 | price   | 26967 non-null | object  |
   
   Berdasarkan *output* di atas, terdapat 6 kolom dengan tipe data float64, 3 kolom dengan tipe data objek, dan 1 kolom dengan tipe data int64.
   
2. **Deskripsi Statistik**

   Tahap ini dilakukan untuk mengecek deskripsi statistik data dengan fitur describe().
   
   **Tabel 2. Deskripsi statistik**
   |       | carat        | depth        | table        | x            | y            | z            | price        |
   |-------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
   | count | 26967.000000 | 26270.000000 | 26967.000000 | 26967.000000 | 26967.000000 | 26967.000000 | 26967.000000 |
   | mean  | 0.798375     | 61.745147    | 57.456080    | 5.729854     | 5.733569     | 3.538057     | 3939.518115  |
   | std   | 0.477745     | 1.412860     | 2.232068     | 1.128516     | 1.166058     | 0.720624     | 4024.864666  |
   | min   | 0.200000     | 50.800000    | 49.000000    | 0.000000     | 0.000000     | 0.000000     | 326.000000   |
   | 25%   | 0.400000     | 61.000000    | 56.000000    | 4.710000     | 4.710000     | 2.900000     | 945.000000   |
   | 50%   | 0.700000     | 61.800000    | 57.000000    | 5.690000     | 5.710000     | 3.520000     | 2375.000000  | 
   | 75%   | 1.050000     | 62.500000    | 59.000000    | 6.550000     | 6.540000     | 4.040000     | 5360.000000  |
   | max   | 4.500000     | 73.600000    | 79.000000    | 10.230000    | 58.900000    | 31.800000    | 18818.000000 |
   
   Berdasarkan output di atas, didapatkan deskripsi statistik yaitu:
   - count : Jumlah sampel data
   - mean : Nilai rata-rata
   - std : Standar deviasi
   - min : Nilai minimum
   - 25% : Kuartil bawah/Q1
   - 50% : Kuartil tengah/Q2/median
   - 75% : Kuartil atas/Q3
   - max : Nilai maksimum
   
3. **Menangani Missing Value**

   Dilakukan pengecekan nilai yang hilang atau missing valie pada kolom dimensi, yaitu x, y, dan z yang bernilai 0. Terdapat missing value pada kolom x sebanyak 3, y sebanyak 3, dan z sebanyak 9. Pembersihan data yang hilang atau missing value pada kolom z yang bernilai 0, kemudian menghapus missing value setiap baris yang memiliki nilai x, y, atau z adalah 0.
   
   Setelah dilakukan pembersihan, dilakukan pengecekan ulang dataset menggunakan fungsi describe().
   
   **Tabel 3. Pengecekan ulang *missing value***
   |       | carat | cut     | color | clarity | depth | table | x    | y    | z   | price |
   |-------|-------|---------|-------|---------|-------|-------|------|------|-----|-------|
   | 5821  | 0.71  | Good    | F     | SI2     | 64.1	 | 60.0  | 0.00 | 0.00 | 0.0 | 2130  |
   | 6034  | 2.02  | Premium | H     | VS2     | 62.7	 | 53.0  | 8.02 | 7.95 | 0.0 | 18207 |
   | 6215  | 0.71  | Good    | F     | SI2     | 64.1	 | 60.0  | 0.00 | 0.00 | 0.0 | 2130  |
   | 10827 | 2.20  | Premium | H     | SI1     | 61.2	 | 59.0  | 8.42 | 8.37 | 0.0 | 17265 |
   | 12498 | 2.18  | Premium | H     | SI2     | 59.4	 | 61.0  | 8.49 | 8.45 | 0.0 | 12631 |
   | 12689 | 1.10  | Premium | G     | SI2     | 63.0	 | 59.0  | 6.50 | 6.47 | 0.0 | 3696  |
   | 17506 | 1.14  | Fair    | G     | VS1     | 57.5	 | 67.0  | 0.00 | 0.00 | 0.0 | 6381  |
   | 18194 | 1.01  | Premium | H     | I1      | 58.1	 | 59.0  | 6.66 | 6.60 | 0.0 | 3167  |
   | 23758 | 1.12  | Premium | G     | I1      | 60.4	 | 59.0  | 6.71 | 6.67 | 0.0 | 2383  |
   
   Berdasarkan *output* di atas, dapat dilihat bahwa sudah tidak terdapat lagi *missing value*. Nilai min (minimal) dari setiap fitur numerik sudah tidak 0 lagi.
   
4. **Menangani Outliers**

   Outliers merupakan sampel yang nilainya sangat jauh dari cakupan umum data utama dan hasil pengamatan yang kemunculannya sangat jarang dan berbeda dari data hasil pengamatan lainnya.
   
   ![outliers-1](https://user-images.githubusercontent.com/77439245/194770492-e098a150-5852-4c94-9328-ccc7d68af635.jpg)  
   **Gambar 6. Grafik boxplot sebelum pembersihan *outliers***
   
   Berdasarkan *output* diagram di atas terlihat bahwa ada *outliers* pada fitur carat, depth, table, price, x, y, dan z. Selanjutnya dilakukan pembersihan *outliers* dengan metode IQR (*Inter Quartile Range*).
   
   $$IQR = Q3 - Q1$$
   
   Lalu membuat batas bawah dengan mengurangi Q1 dengan 1.5 IQR dan batas atas menambah 1.5 IQR dengan Q3.
   
   ![outliers-2](https://user-images.githubusercontent.com/77439245/194770493-5f1cc877-3eb6-4d3c-aee0-caaa8c9b7e0c.jpg)  
   **Gambar 7. Grafik boxplot setelah pembersihan *outliers***
   
   Berdasarkan output diagram di atas, terlihat bahwa outliers telah dibersihkan meskipun masih ada sedikit outliers pada fitur carat, depth dan price, tetapi masih dapat ditoleransi.
   
5. **Univariate Analysis**

   Proses univariate data analysis pada masing-masing fitur kategorial dan numerik.
   
   - Categorical Features
     
     ![univariate-categorical](https://user-images.githubusercontent.com/77439245/194770681-5bcbf7d6-0873-4721-916e-880890746dc6.png)  
     **Gambar 8. Grafik univariat distribusi fitur kategorikal, yaitu cut, color, dan clarity**
     
     Pada fitur cut, terdapat 5 kategori, yaitu Ideal, Premium, Very Good, Good, dan Fair dengan persentase tertinggi terdapat pada kategori Ideal sebesar 42.9%.
     
     Pada fitur color, terdapat 7 kategori, yaitu J, I, H, G, F, E, dan D dengan persentase tertinggi terdapat pada kategori G sebesar 21.3%.
     
     Pada fitur clarity, terdapat 8 kategori, yaitu I1, SI2, SI1, VS2, VS1, VVS2, VVS1, dan IF dengan persentase tertinggi pada terdapat pada kategori SI1 sebesar 24.5%.
     
   - Numerical Features
     
     ![univariate-numerical](https://user-images.githubusercontent.com/77439245/194770710-eb1d25aa-7207-459d-ac81-5dffd3e4d44d.jpg)  
     **Gambar 9. Grafik univariat distribusi fitur numerikal, yaitu carat, depth, table, x, y, z, dan price**
     
     Berdasarkan grafik histogram di atas, dapat disimpulkan sebagai berikut:
     - Pada fitur carat menunjukkan histogram *right-skewed*.
     - Pada fitur depth menunjukkan histogram *zero-skewed* atau simetris/normal.
     - Pada fitur table menunjukkan histogram *right-skewed*.
     - Pada fitur price menunjukkan histogram *right-skewed*.
     - Peningkatan harga batu permata sebanding dengan penurunan jumlah sampel.
   
6. **Multivariate Analysis**

   Proses multivariate data analysis pada masing-masing fitur kategorial dan numerik.
   
   - Categorical Features
     Melakukan pengecekan rata-rata harga terhadap masing-masing fitur kategori, yaitu cut, color, dan clarity untuk mengetahui pengaruh fitur tersebut terhadap harga.
     
     ![multivariate-categorical](https://user-images.githubusercontent.com/77439245/194770804-45d31560-5231-414d-9f03-106fbdd00e23.jpg)  
     **Gambar 10. Grafik multivariat fitur kategorikal, yaitu cut, color, dan clarity terhadap price**
     
     Berdasarkan histogram di atas, dapat disimpulkan:
     - Rata-rata fitur price terhadap fitur cut cenderung sama dengan rentang harga sekitar 2700 sampai 4000. Pada fitur cut, grade terendah yaitu Fair justru memiliki harga rata-rata paling tinggi sehingga fitur cut tidak terlalu berpengaruh terhadap rata-rata harga.
     - Urutan kategori warna dari yang paling buruk hingga paling bagus adalah J, I, H, G, F, E, dan D. Pada fitur color, kualitas warna yang paling buruk yaitu J memiliki rata-rata harga yang paling mahal dari yang lainnya sehingga fitur color tidak terlalu berpengaruh terhadap harga.
     - Urutan kategori clarity paling buruk ke yang paling baik, yaitu: I1, SI2, SI1, VS2, VS1, VVS2, VVS1, dan IF. Pada fitur clarity, grade yang paling tinggi yaitu IF justru memiliki harga yang paling rendah sehingga fitur clarity memiliki pengaruh yang rendah juga terhadap harga.
     - Kesimpulan dari fitur kategori adalah memiliki pengaruh yang rendah terhadap harga.
     
   - Numerical Features
     Melakukan pengecekan rata-rata harga terhadap masing-masing fitur numerik, yaitu carat, depth, table, x, y, dan z untuk mengetahui pengaruh fitur tersebut terhadap harga.
     
     ![multivariate-numerical](https://user-images.githubusercontent.com/77439245/194770821-5c9f1ae4-3397-4db1-b17e-d2ce020c8243.jpg)  
     **Gambar 11. Grafik multivariat fitur antar fitur numerik**
     
     Berdasarkan grafik atau diagram di atas, diperoleh kesimpulan sebagai berikut:
     - Pada fitur price terhadap carat, x, y dan z dapat dilihat memiliki pola sebaran data dengan korelasi positif.
     - Pada fitur price terhadap depth dan table dapat dilihat memiliki pola sebaran data yang acak atau tidak beraturan sehingga tidak memiliki korelasi data.
   
7. **Correlation Matrix**

   Pengecekan korelasi atau hubungan antar fitur numerik menggunakan *heatmap correlation matrix*.
   
   ![correlation-matrix](https://user-images.githubusercontent.com/77439245/194770861-db1ecf84-7f9c-4db1-aaad-60962690a24b.jpg)  
   **Gambar 12. Diagram *heatmap* *Correlation Matrix* fitur numerik**
   
   Berdasarkan diagram *heatmap* di atas, disimpulkan bahwa:
   - Rentang nilai dari 1 sampai -0.2.
   - Jika nilai mendekati 1, maka korelasi antar fitur numerik semakin kuat positif.
   - Jika nilai mendekati 0, maka korelasinya semakin rendah atau semakin tidak ada korelasi.
   - Jika nilai mendekati -1, maka korelasi antar fitur numerik semakin kuat negatif.
   - Korelasi antar fitur numerik yang memiliki korelasi kuat positif yaitu fitur price terhadap carat, x, y, dan z.
   - Korelasi antar fitur numerik yang tidak memiliki korelasi yaitu fitur price terhadap depth dan table.

8. **Menghapus Fitur dengan Korelasi Rendah**

   Penghapusan fitur atau kolom dataset yaitu fitur depth dan table karena fitur tersebut memiliki korelasi yang rendah terhadap fitur price.
   
   **Tabel 4. Pengecekan ulang dataset setelah menghapus fitur dengan korelasi rendah**
   |   | carat | cut       | color | clarity | x    | y    | z    | price |
   |---|-------|-----------|-------|---------|------|------|------|-------|
   | 0 | 0.30  | Ideal     | E     | SI1     | 4.27 | 4.29 | 2.66 | 499   |
   | 1 | 0.33  | Premium   | G     | IF      | 4.42 | 4.46 | 2.70 | 984   |
   | 2 | 0.90  | Very Good | E     | VVS2    | 6.04 | 6.12 | 3.78 | 6289  |
   | 3 | 0.42  | Ideal     | F     | VS1     | 4.82 | 4.80 | 2.96 | 1082  |
   | 4 | 0.31  | Ideal     | F     | VVS1    | 4.35 | 4.43 | 2.65 | 779   |

## Data Preparation

Pada tahap persiapan data atau *data preparation* dilakukan beberapa proses, yaitu *encoding* pada fitur kategori, reduksi dimensi dengan menggunakan Principal Component Analysis (PCA), pembagian atau *split* dataset, dan proses standarisasi data.

1. **Encoding Fitur Kategori**

   Proses *encoding* fitur kategori yaitu cut, color, dan clarity dengan teknik *one-hot-encoding*, sehingga diperoleh fitur baru yang mewakili masing-masing variabel kategori.
   
   **Tabel 5. *Encoding* fitur kategori**
   |   | carat | x    | y    | z    | price | cut_Fair | cut_Good | cut_Ideal | cut_Premium | cut_Very Good | ... | color_I | color_J | clarity_I1 | clarity_IF | clarity_SI1 | clarity_SI2 | clarity_VS1 | clarity_VS2 | clarity_VVS1 | clarity_VVS2 |
   |---|-------|------|------|------|-------|---|---|---|---|---|-----|---|---|---|---|---|---|---|---|---|---|
   | 0 | 0.30  | 4.27 | 4.29 | 2.66 | 499   | 0 | 0 | 1 | 0 | 0 | ... | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
   | 1 | 0.33  | 4.42 | 4.46 | 2.70 | 984   | 0 | 0 | 0 | 1 | 0 | ... | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
   | 2 | 0.90  | 6.04 | 6.12 | 3.78 | 6289  | 0 | 0 | 0 | 0 | 1 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
   | 3 | 0.42  | 4.82 | 4.80 | 2.96 | 1082  | 0 | 0 | 1 | 0 | 0 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
   | 4 | 0.31  | 4.35 | 4.43 | 2.65 | 779   | 0 | 0 | 1 | 0 | 0 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
   
2. **Reduksi Dimensi dengan PCA**

   Proses persiapan data atau *data preparation* dengan teknik reduksi dimensi atau *dimension reduction* merupakan teknik mengurangi jumlah fitur dengan tetap mempertahankan informasi pada data. Teknik pengurangan dimensi yang digunakan dalam kasus ini adalah Principal Component Analysis (PCA) untuk mereduksi dimensi, mengekstraksi fitur, dan mentransformasi data dari "n-dimensional space" ke dalam sistem berkoordinat baru dengan dimensi m, di mana m lebih kecil dari n.
   
   ![reduksi-dimensi](https://user-images.githubusercontent.com/77439245/194770934-17e5073b-71e1-475a-a459-9387d6df1486.jpg)  
   **Gambar 13. Grafik fitur ukuran batu permata, x, y, dan z**
    
   Hasil proporsi informasi dari fitur x, y, dan z dengan menggunakan Principal Component Analysis (PCA), yaitu
   `array([0.998, 0.002, 0.001])`
   
3. **Pembagian Dataset**

   Proses pembagian dataset menjadi data latih dan data uji dengan rasio perbandingan data latih dan data uji, yaitu 90 : 10. Total sampel dataset, data latih, dan data uji setelah membagi atau *split* dataset. Terdapat 23.806 total sampel data dalam dataset, sedangkan untuk total sampel data latih sebanyak 21.425 data dan total sampel data uji sebanyak 2.381 data.
   
4. **Standarisasi**

   Proses standarisasi fitur numerik, yaitu carat dan dimension menggunakan StandardScaler sehingga fitur data menjadi bentuk yang lebih mudah diolah oleh model machine learning.
   
   **Tabel 6. Standarisasi fitur numerik**
   |       | carat     | dimension |
   |-------|-----------|-----------|
   | 18677 | 0.827437  | 0.987784  |
   | 17512 | 2.106236  | 1.748637  |
   | 20365 | -0.504646 | -0.343164 |
   | 21010 | 1.333628  | 1.299652  |
   | 18132 | 2.106236  | 1.765440  |

## Modeling

Pada tahap modeling, dilakukan pemilihan algoritma yang akan digunakan dalam membuat model *machine learning*, serta pengembangan dan pelatihan model *machine learning* agar dapat digunakan untuk melakukan analisis prediksi.

Sebelum melakukan pengembangan model, dilakukan persiapan *dataframe* untuk menganalisis model dengan algoritma K-Nearest Neighbor (KNN), Random Forest, dan Boosting Algorithm.

1. **Algoritma K-Nearest Neighbor (KNN)**

   Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). Pada algoritma K-Nearest Neighbor menggunakan parameter `n-neighbors` dengan nilai k = 10 dan `metric` bawaan yaitu minkowski. [[6]](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
   
   ```python
   knn = KNeighborsRegressor(n_neighbors=10)
   ```
   
2. **Algoritma Random Forest**

   Algoritma *random forest* adalah salah satu algoritma *supervised learning* yang digunakan untuk menyelesaikan masalah klasifikasi dan regresi. *Random forest* merupakan salah satu model *machine learning* yang termasuk ke dalam kategori *ensemble* (*group*) *learning*. Pada algoritma Random Forest menggunakan parameter `n-estimator` dengan jumlah 50 trees (pohon), `max-depth` dengan nilai kedalaman atau panjang pohon sebesar 12, `random-state` dengan nilai 55, dan `n-jobs` yang bernilai -1 yang berarti pekerjaan dilakukan secara paralel.
   
   ```python
   RF = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=55, n_jobs=-1)
   ```
   
3. **Boosting Algorithm**

   Algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi dengan menggabungkan beberapa model sederhana dan dianggap lemah (*weak learners*) sehingga membentuk suatu model yang kuat (*strong ensemble learner*). Pada algoritma Boosting menggunakan parameter `learning-rate` dengan nilai bobot setiap *regressor* adalah 0.05, dan `random-state` dengan nilai 55.
   
   ```python
   boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
   ```

Dari ketiga model *machine learning* dengan algoritma K-Nearest Neighbor, Random Forest, dan Boosting Algorithm, akan dilakukan pengujian performa dan pemilihan 1 model dengan prediksi terbaik dan *error* yang paling rendah.

## Evaluation

Pada tahap evaluasi, dilakukan pengujian model dengan ketiga algoritma yang telah dibuat pada tahap modeling. Sebelum melakukan evaluasi, dilakukan proses *scaling* pada fitur-fitur numerik pada data uji sehingga skala antara data latih dan data uji sama.

```python
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])
```

Proses evaluasi model *machine learning* yang telah dibangun akan menggunakan metrik evaluasi *Mean Squared Error* (MSE) untuk mengevaluasi besaran *error* atau kesalahan dalam model *machine learning*.

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2$$

$n$ = jumlah dataset  
$\sum$ = *summation*/penjumlahan  
$Y_i$ = nilai sebenarnya  
$\hat{Y}_i$ = nilai prediksi  

Berdasarkan hasil evaluasi dengan metrik Mean Square Error (MSE) untuk masing-masing model algoritma machine learning terhadap data latih dan data uji, diperoleh nilai evaluasi sebagai berikut.

**Tabel 7. Nilai evaluasi model *machine learning***
|          | train      | test       |
|----------|------------|------------|
| KNN      | 202.514631 | 261.056355 |
| RF       | 108.567674 | 191.358713 |
| Boosting | 924.027159 | 918.473432 |

![evaluation-graph](https://user-images.githubusercontent.com/77439245/194771116-39e81269-f2ab-4cce-814f-fd5c37f850bc.jpg)  
**Gambar 14. Grafik evaluasi model *machine learning***

Berdasarkan grafik di atas, dapat disimpulkan yaitu:
- Model dengan algoritma Random Forest memberikan nilai *error* yang paling kecil yaitu *train* sebesar 108.567674 dan *test* sebesar 191.358713.
- Model dengan algoritma KKN memberikan nilai *error* *train* sebesar 202.514631 dan *test* sebesar 261.056355.
- Model dengan algoritma Booting memberikan nilai *error* yang paling besar yaitu *train* sebesar 924.027159 dan *test* sebesar 918.473432.

Kemudian dilakukan pengujian prediksi model menggunakan data uji.

**Tabel 8. Hasil pengujian prediksi model**
|      | y_true | prediksi_KNN | prediksi_RF | prediksi_Boosting |
|------|--------|--------------|-------------|-------------------|
| 8697 | 706    | 792.1        | 715.7       | 787.4             |

Berdasarkan *output* tabel di atas dapat dilihat bahwa urutan algoritma yang paling mendekati dengan nilai y_true adalah Random Forest. Nilai y_true sebesar 706 dan nilai prediksi Random Forest sebesar 715.7.

Kesimpulan yang diperoleh dari hasil analisis dan pemodelan *machine learning* untuk kasus ini adalah model yang digunakan untuk melakukan analisis prediksi harga batu permata menghasilkan tingkat *error* yang paling rendah menggunakan algoritma Random Forest dan memberikan hasil prediksi yang paling mendekati dengan data sebenarnya jika dibandingkan dengan algoritma lainnya, yaitu K-Nearest Neighbor dan Boosting Algorithm.

## Referensi

[1] I. Nofalia, "Bagaimana Prospek Investasi Berlian di Masa Depan? Apakah Menguntungkan?", *Finansialku.com*, 2018, Retrieved from: https://www.cermati.com/artikel/segera-beralih-dari-emas-ini-dia-alasan-kenapa-investasi-berlian-lebih-menguntungkan.

[2] S. Hadijah, "Keuntungan, Fakta, dan Tips Investasi Berlian yang Bisa Didapatkan", *Cermati.com*, 2022, Retrieved from: https://www.cermati.com/artikel/segera-beralih-dari-emas-ini-dia-alasan-kenapa-investasi-berlian-lebih-menguntungkan.

[3] L. Afifah, "Algoritma K-Nearest Neighbor (KNN) untuk Klasifikasi", *IlmudataPy*, Retreived from: https://ilmudatapy.com/algoritma-k-nearest-neighbor-knn-untuk-klasifikasi.

[4] Trivusi, "Algoritma Random Forest: Pengertian dan Kegunaannya", *Trivusi*, 2022, Retrieved from: https://www.trivusi.web.id/2022/08/algoritma-random-forest.html.

[5] G. N. Kurniawati, "Algoritma Machine Learning yang Harus Kamu Pelajari di Tahun 2021", *DQLab*, 2021, Retrieved from: https://www.dqlab.id/algoritma-machine-learning-yang-perlu-dipelajari.

[6] scikit-learn, "sklearn.neighbors.KNeighborsRegressor", Retrieved from: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
