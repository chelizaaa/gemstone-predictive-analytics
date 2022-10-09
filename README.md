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
     
     Cara kerja algoritma Random Forest, yaitu: [[4]](https://www.trivusi.web.id/2022/08/algoritma-random-forest.html)
     - Algoritma memilih sampel acak dari dataset yang disediakan.
     - Membuat *decision tree* untuk setiap sampel yang dipilih, lalu didapatkan hasil prediksi dari setiap *decision tree* yang telah dibuat.
     - Melakukan proses voting untuk setiap hasil prediksi, baik menggunakan nilai modus (nilai yang paling sering muncul) untuk kasus klasifikasi, maupun nilai *mean* (nilai rata-rata) untuk kasus regresi
     - Algoritma memilih hasil prediksi yang paling banyak dipilih (*vote* terbanyak) sebagai hasil prediksi akhir.

   - **Boosting Algorithm**
   
     Algoritma Boosting yang digunakan dalam proyek ini adalah algoritma Adaptive Boosting atau biasa disingkat dengan AdaBoost, merupakan algoritma *ensemble* yang memanfaatkan *bagging* dan *boosting* untuk mengembangkan peningkatan akurasi prediksi model *machine learning* yang dibangun. [[5]](https://www.dqlab.id/algoritma-machine-learning-yang-perlu-dipelajari)
     
     ![boosting](https://user-images.githubusercontent.com/77439245/194770091-1f3d0d34-c068-4a95-9cf8-65c5ce999399.jpg)
     
     Algoritma ini menggunakan beberapa pohon keputusan untuk mendapatkan data prediksi secara berurutan dan prosesnya iteratif. Data latih akan diberikan bobot yang sama untuk kemudian dilakukan pemeriksaan, dan bobot yang lebih tinggi akan masuk ke model yang salah sehingga akan lanjut ke tahap selanjutnya secara berulang hingga tingkat akurasi yang diinginkan.

## Data Understanding

![gemstone-dataset](https://user-images.githubusercontent.com/77439245/194770177-88b4044c-1911-48fb-919e-e29f4044c12b.png)

Dataset yang digunakan dalam proyek ini adalah dataset [Gemstone Price Prediction](https://www.kaggle.com/datasets/colearninglounge/gemstone-price-prediction?select=cubic_zirconia.csv) yang diambil dari platform Kaggle. *File* yang digunakan berupa *file* csv, yaitu `cubic_zirconia.csv`.

Dari dataset tersebut, dilakukan penghapusan kolom pertama yaitu Unnamed yang berisikan nomor masing-masing data.

Kemudian dilakukan proses *Exploratory Data Analysis* (EDA) yang merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data.

1. **Deskripsi Variabel**

   Pada deskripsi variabel dilakukan pengecekan informasi variabel dari dataset yaitu jumlah kolom, nama kolom, jumlah data per kolom dan tipe datanya.
   
   Berikut adalah informasi variabel dari dataset Gemstone Price Prediction:
   
   ![variabel](https://user-images.githubusercontent.com/77439245/194770314-64f0100e-e4cf-41df-81b9-fcce8ee81067.png)
   
   Berdasarkan *output* di atas, terdapat 6 kolom dengan tipe data float64, 3 kolom dengan tipe data objek, dan 1 kolom dengan tipe data int64.
   
2. **Deskripsi Statistik**

   Tahap ini dilakukan untuk mengecek deskripsi statistik data dengan fitur describe().
   
   ![statistik](https://user-images.githubusercontent.com/77439245/194770334-824b3e3e-c02d-466d-8bea-03d1ac35e88f.png)
   
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
   
   ![missing-value](https://user-images.githubusercontent.com/77439245/194770467-e62e6791-e85e-4148-b80d-25f8f17d68e9.png)
   
   Berdasarkan *output* di atas, dapat dilihat bahwa sudah tidak terdapat lagi *missing value*. Nilai min (minimal) dari setiap fitur numerik sudah tidak 0 lagi.
   
4. **Menangani Outliers**

   Outliers merupakan sampel yang nilainya sangat jauh dari cakupan umum data utama dan hasil pengamatan yang kemunculannya sangat jarang dan berbeda dari data hasil pengamatan lainnya.
   
   ![outliers-1](https://user-images.githubusercontent.com/77439245/194770492-e098a150-5852-4c94-9328-ccc7d68af635.jpg)
   
   Berdasarkan *output* diagram di atas terlihat bahwa ada *outliers* pada fitur carat, depth, table, price, x, y, dan z. Selanjutnya dilakukan pembersihan *outliers* dengan metode IQR (*Inter Quartile Range*).
   
   $$IQR = Q3 - Q1$$
   
   Lalu membuat batas bawah dengan mengurangi Q1 dengan 1.5 IQR dan batas atas menambah 1.5 IQR dengan Q3.
   
   ![outliers-2](https://user-images.githubusercontent.com/77439245/194770493-5f1cc877-3eb6-4d3c-aee0-caaa8c9b7e0c.jpg)
   
   Berdasarkan output diagram di atas, terlihat bahwa outliers telah dibersihkan meskipun masih ada sedikit outliers pada fitur carat, depth dan price, tetapi masih dapat ditoleransi.
   
5. **Univariate Analysis**

   Proses univariate data analysis pada masing-masing fitur kategorial dan numerik.
   
   - Categorical Features
     
     ![univariate-categorical](https://user-images.githubusercontent.com/77439245/194770681-5bcbf7d6-0873-4721-916e-880890746dc6.png)
     
     Pada fitur cut, terdapat 5 kategori, yaitu Ideal, Premium, Very Good, Good, dan Fair dengan persentase tertinggi terdapat pada kategori Ideal sebesar 42.9%.
     
     Pada fitur color, terdapat 7 kategori, yaitu J, I, H, G, F, E, dan D dengan persentase tertinggi terdapat pada kategori G sebesar 21.3%.
     
     Pada fitur clarity, terdapat 8 kategori, yaitu I1, SI2, SI1, VS2, VS1, VVS2, VVS1, dan IF dengan persentase tertinggi pada terdapat pada kategori SI1 sebesar 24.5%.
     
   - Numerical Features
     
     ![univariate-numerical](https://user-images.githubusercontent.com/77439245/194770710-eb1d25aa-7207-459d-ac81-5dffd3e4d44d.jpg)
     
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
     
     Berdasarkan histogram di atas, dapat disimpulkan:
     - Rata-rata fitur price terhadap fitur cut cenderung sama dengan rentang harga sekitar 2700 sampai 4000. Pada fitur cut, grade terendah yaitu Fair justru memiliki harga rata-rata paling tinggi sehingga fitur cut tidak terlalu berpengaruh terhadap rata-rata harga.
     - Urutan kategori warna dari yang paling buruk hingga paling bagus adalah J, I, H, G, F, E, dan D. Pada fitur color, kualitas warna yang paling buruk yaitu J memiliki rata-rata harga yang paling mahal dari yang lainnya sehingga fitur color tidak terlalu berpengaruh terhadap harga.
     - Urutan kategori clarity paling buruk ke yang paling baik, yaitu: I1, SI2, SI1, VS2, VS1, VVS2, VVS1, dan IF. Pada fitur clarity, grade yang paling tinggi yaitu IF justru memiliki harga yang paling rendah sehingga fitur clarity memiliki pengaruh yang rendah juga terhadap harga.
     - Kesimpulan dari fitur kategori adalah memiliki pengaruh yang rendah terhadap harga.
     
   - Numerical Features
     Melakukan pengecekan rata-rata harga terhadap masing-masing fitur numerik, yaitu carat, depth, table, x, y, dan z untuk mengetahui pengaruh fitur tersebut terhadap harga.
     
     ![multivariate-numerical](https://user-images.githubusercontent.com/77439245/194770821-5c9f1ae4-3397-4db1-b17e-d2ce020c8243.jpg)
     
     Berdasarkan grafik atau diagram di atas, diperoleh kesimpulan sebagai berikut:
     - Pada fitur price terhadap carat, x, y dan z dapat dilihat memiliki pola sebaran data dengan korelasi positif.
     - Pada fitur price terhadap depth dan table dapat dilihat memiliki pola sebaran data yang acak atau tidak beraturan sehingga tidak memiliki korelasi data.
   
7. **Correlation Matrix**

   Pengecekan korelasi atau hubungan antar fitur numerik menggunakan *heatmap correlation matrix*.
   
   ![correlation-matrix](https://user-images.githubusercontent.com/77439245/194770861-db1ecf84-7f9c-4db1-aaad-60962690a24b.jpg)
   
   Berdasarkan diagram heatmap di atas, disimpulkan bahwa:
   - Rentang nilai dari 1 sampai -0.2.
   - Jika nilai mendekati 1, maka korelasi antar fitur numerik semakin kuat positif.
   - Jika nilai mendekati 0, maka korelasinya semakin rendah atau semakin tidak ada korelasi.
   - Jika nilai mendekati -1, maka korelasi antar fitur numerik semakin kuat negatif.
   - Korelasi antar fitur numerik yang memiliki korelasi kuat positif yaitu fitur price terhadap carat, x, y, dan z.
   - Korelasi antar fitur numerik yang tidak memiliki korelasi yaitu fitur price terhadap depth dan table.

8. **Menghapus Fitur dengan Korelasi Rendah**

   Penghapusan fitur atau kolom dataset yaitu fitur depth dan table karena fitur tersebut memiliki korelasi yang rendah terhadap fitur price.
   
   ![Screenshot 2022-10-10 at 00-27-21 Google Colaboratory](https://user-images.githubusercontent.com/77439245/194770865-f93c4c23-7d03-4f56-bf01-ed38d3574746.png)

## Data Preparation

Pada tahap persiapan data atau *data preparation* dilakukan beberapa proses, yaitu *encoding* pada fitur kategori, reduksi dimensi dengan menggunakan Principal Component Analysis (PCA), pembagian atau *split* dataset, dan proses standarisasi data.

1. **Encoding Fitur Kategori**

   Proses *encoding* fitur kategori yaitu cut, color, dan clarity dengan teknik *one-hot-encoding*, sehingga diperoleh fitur baru yang mewakili masing-masing variabel kategori.
   
   ![encoding](https://user-images.githubusercontent.com/77439245/194770925-20b916cb-d836-43fd-9c5f-e6df04473267.png)
   
2. **Reduksi Dimensi dengan PCA**

   Proses persiapan data atau *data preparation* dengan teknik reduksi dimensi atau *dimension reduction* merupakan teknik mengurangi jumlah fitur dengan tetap mempertahankan informasi pada data. Teknik pengurangan dimensi yang digunakan dalam kasus ini adalah Principal Component Analysis (PCA) untuk mereduksi dimensi, mengekstraksi fitur, dan mentransformasi data dari "n-dimensional space" ke dalam sistem berkoordinat baru dengan dimensi m, di mana m lebih kecil dari n.
   
   ![reduksi-dimensi](https://user-images.githubusercontent.com/77439245/194770934-17e5073b-71e1-475a-a459-9387d6df1486.jpg)
    
   Hasil proporsi informasi dari fitur x, y, dan z dengan menggunakan Principal Component Analysis (PCA), yaitu
   
   ![proporsi-reduksi](https://user-images.githubusercontent.com/77439245/194770977-2291dbea-1308-4300-9898-fe7120911f0a.png)
   
3. **Pembagian Dataset**

   Proses pembagian dataset menjadi data latih dan data uji dengan rasio perbandingan data latih dan data uji, yaitu 90 : 10. Total sampel dataset, data latih, dan data uji setelah membagi atau *split* dataset.
   
   ![split-dataset](https://user-images.githubusercontent.com/77439245/194771006-0bfe5434-dc6f-4c3a-a598-b3172647762c.png)
   
4. **Standarisasi**

   Proses standarisasi fitur numerik, yaitu carat dan dimension menggunakan StandardScaler sehingga fitur data menjadi bentuk yang lebih mudah diolah oleh model machine learning.
   
   ![standarisasi](https://user-images.githubusercontent.com/77439245/194771043-8e44db77-4ee8-456b-ac7a-c7b365d803b3.png)

## Modeling

Pada tahap modeling, dilakukan pemilihan algoritma yang akan digunakan dalam membuat model *machine learning*, serta pengembangan dan pelatihan model *machine learning* agar dapat digunakan untuk melakukan analisis prediksi.

Sebelum melakukan pengembangan model, dilakukan persiapan *dataframe* untuk menganalisis model dengan algoritma K-Nearest Neighbor (KNN), Random Forest, dan Boosting Algorithm.

```python
models = pd.DataFrame(index = ['train_mse', 'test_mse'],
                      columns = ['KNN', 'RandomForest', 'Boosting'])
```

1. **Algoritma K-Nearest Neighbor (KNN)**

   Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif).
   
   ```python
   knn = KNeighborsRegressor(n_neighbors=10)
   ```
   
2. **Algoritma Random Forest**

   Algoritma *random forest* adalah salah satu algoritma *supervised learning* yang digunakan untuk menyelesaikan masalah klasifikasi dan regresi. *Random forest* merupakan salah satu model *machine learning* yang termasuk ke dalam kategori *ensemble* (*group*) *learning*.
   
   ```python
   RF = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=55, n_jobs=-1)
   ```
   
3. **Boosting Algorithm**

   Algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi dengan menggabungkan beberapa model sederhana dan dianggap lemah (*weak learners*) sehingga membentuk suatu model yang kuat (*strong ensemble learner*).
   
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

![evaluation-mse](https://user-images.githubusercontent.com/77439245/194771114-8506e42e-a68d-4489-9dda-5392b8cc5697.png)

![evaluation-graph](https://user-images.githubusercontent.com/77439245/194771116-39e81269-f2ab-4cce-814f-fd5c37f850bc.jpg)

Berdasarkan grafik di atas, dapat disimpulkan yaitu:
- Model dengan algoritma Random Forest memberikan nilai *error* yang paling kecil yaitu *train* sebesar 108.567674 dan *test* sebesar 191.358713.
- Model dengan algoritma KKN memberikan nilai *error* *train* sebesar 202.514631 dan *test* sebesar 261.056355.
- Model dengan algoritma Booting memberikan nilai *error* yang paling besar yaitu *train* sebesar 924.027159 dan *test* sebesar 918.473432.

Kemudian dilakukan pengujian prediksi model menggunakan data uji.

![evaluation-test](https://user-images.githubusercontent.com/77439245/194771122-ba76cd34-4914-40c9-bdb2-c9834d1c81e3.png)

Berdasarkan *output* tabel di atas dapat dilihat bahwa urutan algoritma yang paling mendekati dengan nilai y_true adalah Random Forest. Nilai y_true sebesar 706 dan nilai prediksi Random Forest sebesar 715.7.

Kesimpulan yang diperoleh dari hasil analisis dan pemodelan *machine learning* untuk kasus ini adalah model yang digunakan untuk melakukan analisis prediksi harga batu permata menghasilkan tingkat *error* yang paling rendah menggunakan algoritma Random Forest dan memberikan hasil prediksi yang paling mendekati dengan data sebenarnya jika dibandingkan dengan algoritma lainnya, yaitu K-Nearest Neighbor dan Boosting Algorithm.

## Referensi

[1] I. Nofalia, "Bagaimana Prospek Investasi Berlian di Masa Depan? Apakah Menguntungkan?", *Finansialku.com*, 2018, Retrieved from: https://www.cermati.com/artikel/segera-beralih-dari-emas-ini-dia-alasan-kenapa-investasi-berlian-lebih-menguntungkan.

[2] S. Hadijah, "Keuntungan, Fakta, dan Tips Investasi Berlian yang Bisa Didapatkan", *Cermati.com*, 2022, Retrieved from: https://www.cermati.com/artikel/segera-beralih-dari-emas-ini-dia-alasan-kenapa-investasi-berlian-lebih-menguntungkan.

[3] L. Afifah, "Algoritma K-Nearest Neighbor (KNN) untuk Klasifikasi", *IlmudataPy*, Retreived from: [https://ilmudatapy.com/algoritma-k-nearest-neighbor-knn-untuk-klasifikasi](https://ilmudatapy.com/algoritma-k-nearest-neighbor-knn-untuk-klasifikasi).

[4] Trivusi, "Algoritma Random Forest: Pengertian dan Kegunaannya", *Trivusi*, 2022, Retrieved from: https://www.trivusi.web.id/2022/08/algoritma-random-forest.html.

[5] G. N. Kurniawati, "Algoritma Machine Learning yang Harus Kamu Pelajari di Tahun 2021", *DQLab*, 2021, Retrieved from: https://www.dqlab.id/algoritma-machine-learning-yang-perlu-dipelajari.
