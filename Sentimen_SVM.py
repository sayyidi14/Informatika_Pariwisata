from sklearn import svm
from sklearn.model_selection import train_test_split
import swifter
from nltk.corpus import stopwords
from nltk import word_tokenize
import seaborn as sns
import streamlit as st
import pandas as pd

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["Beranda", "Data Set", "Pre Processing", "TF-IDF", "SVM Linear", "SVM RBF", "Uji Sentimen Linier", "Uji Sentimen RBF"])

with tab1:
    st.markdown("""
    Nama\t : Sayyidi Makhdum\n
    NIM\t : 190411100159\n
    Kelas\t : Informatika Pariwisata B""")
    st.header("Analisis Sentimen Ulasan Google Maps Menggunakan Metode Support Vector Machine (SVM) Studi Kasus: Pantai Pasir Putih Tlangoh, Kabupaten Bangkalan")
with tab2:
    st.header("DATA SET")
    st.markdown(
        "Data Set yang digunakan Merupakan Hasil Crawling Ulasan Google Maps Pantai Pasir Putih Tlangong Dengan Jumlah 489 Ulasan")
    df = pd.read_excel("Data_Ulasan_Tlangoh_New.xlsx")
    df.columns = ['Ulasan Mentah', 'Ulasan', 'Sentimen']
    del df["Ulasan Mentah"]
    df = df.dropna()
    st.dataframe(df)
    st.text(f"Jumlah Baris Data = {df.shape[0]}")
    st.text(f"Jumlah Kolom Data = {df.shape[1]}")

    def labels(sentimen):
        if sentimen == "Negatif":
            return "-1"
        elif sentimen == "Netral":
            return "0"
        else:
            return "1"

    df["Label"] = df["Sentimen"].apply(labels)

    st.dataframe(df[:489][["Sentimen", "Label"]])

    import matplotlib.pyplot as plt

    label_count = df["Sentimen"].value_counts()
    label_count = label_count.sort_index()
    st.markdown("Kategori Sentimen Dibagi Menjadi 3 Kategori Yaitu")
    st.text(df.Sentimen.value_counts())

    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlabel('Sentimen')
    ax.set_ylabel('Jumlah Data')
    ax.bar(label_count.index, label_count.values)
    st.pyplot(fig)


with tab3:
    st.title("Pre Processing Data")
    st.header("Proses Cleaning")
    st.markdown("""
    1. Pembersihan Tanda Baca
    2. Pembersihan Link
    3. Pembersihan Angka
    4. Pembersihan Spasi Kosong
    5. Pembersihan Enter

    """)
# Cleaning Ulasan===========================================
    import re

    def cleaningulasan(ulasan):
        ulasan = re.sub(r'@[A-Za-a0-9]+', ' ', ulasan)
        ulasan = re.sub(r'#[A-Za-z0-9]+', ' ', ulasan)
        ulasan = re.sub(r"http\S+", ' ', ulasan)
        ulasan = re.sub(r'[0-9]+', ' ', ulasan)
        ulasan = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", ulasan)
        ulasan = ulasan.strip(' ')
        ulasan = ulasan.strip("\n")
        return ulasan

    df["Cleaning"] = df["Ulasan"].apply(cleaningulasan)
    st.dataframe(df[:489][["Ulasan", "Cleaning"]])

# Hilangkan Emoji===========================================
    st.header("Proses Penghilang Emoji/Emoticon")

    def clearEmoji(ulasan):
        return ulasan.encode('ascii', 'ignore').decode('ascii')
    df['HapusEmoji'] = df['Cleaning'].apply(clearEmoji)
    st.dataframe(df[:489][["Cleaning", "HapusEmoji"]])

    st.header("Proses Case Folding")
    st.markdown("Case Folding Merubah Seluruh Data Review Menjadi Huruf Kecil")

# Case Folding===========================================
    def casefoldingText(ulasan):
        ulasan = ulasan.lower()
        return ulasan
    df['CaseFolding'] = df['HapusEmoji'].apply(casefoldingText)
    st.dataframe(df[:489][["HapusEmoji", "CaseFolding"]])

# Tokenizing===========================================
    st.header("Proses Tokenizing")
    st.markdown(" Proses Tokenizing Untuk Memishkan Kalimat Menjadi Kata")
    import nltk
    nltk.download('punkt')

    def word_tokenize_wrapper(text):
        return word_tokenize(text)

    df['Tokenizing'] = df['CaseFolding'].str.lower().apply(word_tokenize_wrapper)
    st.dataframe(df[:489][["CaseFolding", "Tokenizing"]])

# Formalisasi===========================================
    st.header("Proses Formalisasi")
    st.markdown(
        "Proses Formalisasi Data Review Merubah Kata Gaul atau Bahasa Informal Menjadi Bahasa Formal")

    def convertToSlangword(ulasan):
        kamusSlang = eval(open(
            "slangwords.txt").read())
        pattern = re.compile(r'\b( ' + '|'.join(kamusSlang.keys())+r')\b')
        content = []
        for kata in ulasan:
            filterSlang = pattern.sub(lambda x: kamusSlang[x.group()], kata)
            content.append(filterSlang.lower())
        ulasan = content
        return ulasan

    df['Formalisasi'] = df['Tokenizing'].apply(convertToSlangword)
    st.dataframe(df[:489][["Tokenizing", "Formalisasi"]])

# Stopwords===========================================
    st.header("Proses Stopwords")
    st.markdown(
        "Proses Stopwords untuk Menghilangkan Kata yang Tidak Memiliki Nilai, Misal Konjungsi atau Kata Hubung")
    nltk.download('stopwords')

    daftar_stopword = stopwords.words('indonesian')
    # Masukan Kata dalam Stopwors Secara Manula
    # Tambahakan Data Stopwords Manual
    daftar_stopword.extend(
        ["nya", "es", "&", "x", "ok", "deh", "nya", "si", "a", "it", "r", "ahh"])
    daftar_stopword = set(daftar_stopword)

    def stopwordText(words):
        return [word for word in words if word not in daftar_stopword]
    # def datakosong(words):
    #   return [word for word in words if word ]

    df['Stopword Removal'] = df['Formalisasi'].apply(stopwordText)
    st.dataframe(df[:489][["Formalisasi", "Stopword Removal"]])

    df = df[df['Stopword Removal'].apply(lambda x: len(x) > 0)]
    st.dataframe(df[:489][["Formalisasi", "Stopword Removal"]])


# Stemming===========================================
    st.header("Proses Stemming")
    st.markdown(
        "Proses Stemming Data Mengembalikan Sebuah Kata Menjadi Kata Dasar, Misal Pantainya ==> Pantai Berburu ===> Buru")
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_dict = {}

    for document in df['Stopword Removal']:
        for term in document:
            if term not in term_dict:
                term_dict[term] = ' '

    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)
        print(term, ":", term_dict[term])

    def stemmingText(document):
        return [term_dict[term] for term in document]

    df['Stemming'] = df['Stopword Removal'].swifter.apply(stemmingText)
    st.dataframe(df[:489][["Stopword Removal", "Stemming"]])

# Join kata==========================================
    st.header("Proses Penggabungan Kembali Kata Menjadi Sebuah Kalimat")
    import string

    def remove_punct(text):
        text = " ".join(
            [char for char in text if char not in string.punctuation])
        return text

    df["Clean"] = df["Stemming"].apply(lambda x: remove_punct(x))
    st.dataframe(df[:489][["Stemming", "Clean"]])

    data_prepos = st.button("Simpan Data Prepos")

    if data_prepos:
        df.to_excel("Data_Prepos.xlsx")
        st.success("Data Berhasil Disimpan")

with tab4:
    st.title("TF-IDF")
    df = pd.read_excel(
        "Data_Prepos.xlsx")
    del df["Unnamed: 0"]
    st.subheader(
        "Data Hasil Stemming Akan Dilakukan Proses Perhitungan Bobot TF-IDF")
    st.markdown("Rumus Menghitung TF")
    st.latex(
        r"tf\left (t,d  \right )=\frac{f_t,_d}{{\sum}_{t^1 \in d}f_{t^1,d}}")
    st.markdown("Rumus Menghitung IDF")
    st.latex(r"""idf(t, D)= \log \frac{N}{ |\begin{Bmatrix}
                                        d\in D : t\in d
                                        \end{Bmatrix}|}""")
    st.markdown("Rumus Menghitung TF-IDF")
    st.latex(r"""
    tfidf(t,d,D)= tf(t,d) \times idf(t,D)
    """)
    st.dataframe(df[:489][["Stemming"]])

    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["Stemming"])
    st.markdown(
        "Data Array Bobot Tf-IDF")
    st.text(X)

    st.text(vectorizer.get_feature_names_out())

    # st.markdown(f"Banyaknya kosa kata = {len((cv.get_feature_names_out()))}")

    df_bobot = pd.DataFrame(X.todense().T,
                            index=vectorizer.get_feature_names_out(),
                            columns=[f'D{i+1}' for i in range(len(df["Stemming"]))])
    st.markdown(
        "Data Tabel Bobot Kata Dalam Setiap Dokumen")
    st.dataframe(df_bobot)

with tab5:
    st.title("SVM Linear")
    st.header("Proses Spliting Data Menjadi 20% Data Testing dan 70% Training")
    from sklearn import model_selection

    X = df['Clean']
    Y = df['Label']

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        X, Y, test_size=0.2)

    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    cv = CountVectorizer()
    cv_matrix = cv.fit_transform(X)

    code = """
    X = df['Clean']
    Y = df['Label']

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        X, Y, test_size=0.2)

    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)"""
    st.code(code, language="python")

    cv = CountVectorizer()
    cv_matrix = cv.fit_transform(X)

    # x_train, y_train

    from sklearn.model_selection import cross_val_score

    # SVM = svm.SVC(kernel='rbf')  # Jika dengan Kernel RBF
    SVM = svm.SVC(kernel='linear')  # Jika dengan Kernel Linear
    SVM.fit(x_train, y_train)

    acc_score = cross_val_score(
        SVM, x_train, y_train, cv=5, scoring='accuracy')
    pre_score = cross_val_score(SVM, x_train, y_train,
                                cv=5, scoring='precision_macro')
    rec_score = cross_val_score(SVM, x_train, y_train,
                                cv=5, scoring='recall_macro')
    f_score = cross_val_score(SVM, x_train, y_train, cv=5, scoring='f1_macro')

    st.header("Penerapan SVM Linear")
    code_acc_cv = """
    SVM = svm.SVC(kernel='linear')
    SVM.fit(x_train, y_train)

    acc_score = cross_val_score(
        SVM, x_train, y_train, cv=5, scoring='accuracy')
    pre_score = cross_val_score(SVM, x_train, y_train,
                                cv=5, scoring='precision_macro')
    rec_score = cross_val_score(SVM, x_train, y_train,
                                cv=5, scoring='recall_macro')
    f_score = cross_val_score(SVM, x_train, y_train, cv=5, scoring='f1_macro')"""

    st.code(code_acc_cv, language="python")

    st.header("Hasil Aklurasi Menggukan 5 Corss Valiadtion")

    st.text('Hasil Accuracy : %s' % (acc_score))
    st.text('Hasil Rata - Rata Accuracy : %s' % acc_score.mean())
    st.text('Hasil Precision : %s' % (pre_score))
    st.text('Hasil Rata - Rata Precision : %s' % pre_score.mean())
    st.text('Hasil Recall : %s' % (rec_score))
    st.text('Hasil Rata - Rata Recall : %s' % rec_score.mean())
    st.text('Hasil F-Measure : %s' % (f_score))
    st.text('Hasil Rata - Rata F-Measure : %s' % f_score.mean())

    y_pred = SVM.predict(x_test)

    from sklearn.metrics import classification_report, confusion_matrix

    st.header("Hasil Aklurasi Menggukan Tanpa Corss Valiadtion")
    st.text(confusion_matrix(y_test, y_pred))
    st.text(classification_report(y_test, y_pred))

    import joblib
    # joblib_models = joblib.load("modelSVMlinear")

    bt_simpan_Linear = st.button("Simpan Model Linear")

    if bt_simpan_Linear:
        st.success("Model Berhasil Disimpan")

with tab6:
    st.title("SVM RBF")
    st.header("Proses Spliting Data Menjadi 20% Data Testing dan 70% Training")
    from sklearn import model_selection

    X = df['Clean']
    Y = df['Label']

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        X, Y, test_size=0.2)

    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    cv = CountVectorizer()
    cv_matrix = cv.fit_transform(X)

    code = """
    X = df['Clean']
    Y = df['Label']

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        X, Y, test_size=0.2)

    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)"""
    st.code(code, language="python")

    cv = CountVectorizer()
    cv_matrix = cv.fit_transform(X)

    # x_train, y_train

    from sklearn.model_selection import cross_val_score

    SVM = svm.SVC(kernel='rbf')  # Jika dengan Kernel RBF
    # SVM = svm.SVC(kernel='linear')  # Jika dengan Kernel Linear
    SVM.fit(x_train, y_train)

    acc_score = cross_val_score(
        SVM, x_train, y_train, cv=5, scoring='accuracy')
    pre_score = cross_val_score(SVM, x_train, y_train,
                                cv=5, scoring='precision_macro')
    rec_score = cross_val_score(SVM, x_train, y_train,
                                cv=5, scoring='recall_macro')
    f_score = cross_val_score(SVM, x_train, y_train, cv=5, scoring='f1_macro')

    st.header("Penerapan SVM RBF")
    code_acc_cv = """
    SVM = svm.SVC(kernel='rbf')
    SVM.fit(x_train, y_train)

    acc_score = cross_val_score(
        SVM, x_train, y_train, cv=5, scoring='accuracy')
    pre_score = cross_val_score(SVM, x_train, y_train,
                                cv=5, scoring='precision_macro')
    rec_score = cross_val_score(SVM, x_train, y_train,
                                cv=5, scoring='recall_macro')
    f_score = cross_val_score(SVM, x_train, y_train, cv=5, scoring='f1_macro')"""

    st.code(code_acc_cv, language="python")

    st.header("Hasil Aklurasi Menggukan 5 Corss Valiadtion")

    st.text('Hasil Accuracy : %s' % (acc_score))
    st.text('Hasil Rata - Rata Accuracy : %s' % acc_score.mean())
    st.text('Hasil Precision : %s' % (pre_score))
    st.text('Hasil Rata - Rata Precision : %s' % pre_score.mean())
    st.text('Hasil Recall : %s' % (rec_score))
    st.text('Hasil Rata - Rata Recall : %s' % rec_score.mean())
    st.text('Hasil F-Measure : %s' % (f_score))
    st.text('Hasil Rata - Rata F-Measure : %s' % f_score.mean())

    y_pred = SVM.predict(x_test)

    from sklearn.metrics import classification_report, confusion_matrix

    st.header("Hasil Aklurasi Menggukan Tanpa Corss Valiadtion")
    st.text(confusion_matrix(y_test, y_pred))
    st.text(classification_report(y_test, y_pred))

    import joblib
    joblib.dump(
        SVM, "modelSVMRBF")
    # joblib_models = joblib.load("modelSVMRBF")

    bt_simpan_Linear = st.button("Simpan Model RBF")

    if bt_simpan_Linear:
        st.success("Model Berhasil Disimpan")

with tab7:
    st.title("Implementasi Sentimen Dengan Kernel Linier")
    joblib.dump(
        SVM, "modelSVMlinear")
    joblib_models = joblib.load(
        "modelSVMlinear")

    def prediksi(text):
        tfidf_vektor = vectorizer.transform([text])
        pred = joblib_models.predict(tfidf_vektor)
        if pred == 1:
            hate = "Positif"
        elif pred == 0:
            hate = "Netral"
        elif pred == -1:
            hate = "Negatif"
        else:
            hate = "Error"
        return hate

    # masukkan_kalimat = str(input("Masukan Review :"))
    masukkan_kalimat = st.text_input("Masukkan Review :")
    masukkan_kalimat = cleaningulasan(masukkan_kalimat)
    masukkan_kalimat = clearEmoji(masukkan_kalimat)
    masukkan_kalimat = casefoldingText(masukkan_kalimat)
    masukkan_kalimat = word_tokenize_wrapper(masukkan_kalimat)
    masukkan_kalimat = convertToSlangword(masukkan_kalimat)
    masukkan_kalimat = stopwordText(masukkan_kalimat)
    masukkan_kalimat = stemmingText(masukkan_kalimat)
    masukkan_kalimat = " ".join(masukkan_kalimat)

    uji_linier = st.button("Cek Sentimen")
    import time
    if uji_linier:
        with st.spinner('Wait for it...'):
            time.sleep(2)
            st.balloons()
        st.markdown(masukkan_kalimat)
        st.markdown(f"Sentimen Review : {prediksi(masukkan_kalimat)}")
        st.success("Review Berhasil Dilakukan Analisis", icon="✅")


with tab8:
    st.title("Implementasi Sentimen Dengan Kernel RBF")
    joblib.dump(
        SVM, "modelSVMRBF")
    joblib_models = joblib.load(
        "modelSVMRBF")

    def prediksi(text):
        tfidf_vektor = vectorizer.transform([text])
        pred = joblib_models.predict(tfidf_vektor)
        if pred == 1:
            hate = "Positif"
        elif pred == 0:
            hate = "Netral"
        elif pred == -1:
            hate = "Negatif"
        else:
            hate = "Error"
        return hate

    # masukkan_kalimat = str(input("Masukan Review :"))
    masukkan_kalimat = st.text_input("Masukkan Review RBF")
    masukkan_kalimat = cleaningulasan(masukkan_kalimat)
    masukkan_kalimat = clearEmoji(masukkan_kalimat)
    masukkan_kalimat = casefoldingText(masukkan_kalimat)
    masukkan_kalimat = word_tokenize_wrapper(masukkan_kalimat)
    masukkan_kalimat = convertToSlangword(masukkan_kalimat)
    masukkan_kalimat = stopwordText(masukkan_kalimat)
    masukkan_kalimat = stemmingText(masukkan_kalimat)
    masukkan_kalimat = " ".join(masukkan_kalimat)

    uji_linier = st.button("Cek Sentimen RBF")
    import time
    if uji_linier:
        with st.spinner('Wait for it...'):
            time.sleep(2)
            st.balloons()
        st.markdown(masukkan_kalimat)
        st.markdown(f"Sentimen Review : {prediksi(masukkan_kalimat)}")
        st.success("Review Berhasil Dilakukan Analisis", icon="✅")
