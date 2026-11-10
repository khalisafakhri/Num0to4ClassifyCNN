# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:55:56 2026

@author: fakhr

CARA PAKAI FILE INI
Jalankan cell satu per satu dengan Shift + Enter.
Jangan tekan F5 (Run file), karena F5 menjalankan semua cell dari atas.

ISI TIAP CELL
1. Data -> memuat dan menyiapkan dataset MNIST (angka 0-4)
2. Model dengan hidden layer
3. Model tanpa hidden layer -> sebagai pembanding
4. Simpan model dasar -> Base_model_0_4.keras
5. Model v1 -> MNIST saja (5 kelas, angka 0-4)                                             -> Latihan_CNN_0_4_v1.keras
6. Model v2 -> MNIST + angka 5-9 + QuickDraw/Google (6 kelas)                              -> Latihan_CNN_0_4_v3.keras
7. Model v3 -> MNIST + data buatan (4 dan 1 gaya ketikan) + QuickDraw + Dataset sendiri    -> Latihan_CNN_0_4_v7.keras

"""

# %% 1. Data

import numpy as np                    # untuk hitungan angka dan array
from tensorflow import keras          # untuk membuat model neural network
from PIL import Image                 # untuk membuka dan mengolah foto

# Mengambil dataset MNIST
# X = gambar, y = jawaban (angka yang benar)
# train = untuk belajar, test = untuk ujian
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Ambil angka 0 sampai 4
X_train = X_train[y_train < 5]
y_train = y_train[y_train < 5]
X_test = X_test[y_test < 5]
y_test = y_test[y_test < 5]

# Ubah nilai tiap piksel dari 0-255 menjadi 0-1 supaya model belajar lebih stabil
X_train = X_train / 255
X_test = X_test / 255

# Tampilkan jumlah gambar untuk data latih dan data uji
print("Jumlah data Latih:", len(X_train))
print("Jumlah data Uji:", len(X_test))

# %% 2. Membuat model dengan hidden layer

model = keras.Sequential([
    keras.Input(shape=(28, 28)),                    # gambar 28x28 piksel
    keras.layers.Flatten(),                         # gambar dijadikan satu baris yang totalnya ada 784 angka
    keras.layers.Dense(100, activation='relu'),     # hidden layer: 100 neuron yang belajar mengenali pola
    keras.layers.Dense(5, activation='softmax')     # lapisan hasil: 5 neuron untuk angka 0-4
])

# Mengatur cara model belajar
model.compile(optimizer='adam',                          # cara model memperbaiki dirinya
              loss='sparse_categorical_crossentropy',    # ukuran seberapa salah tebakan model
              metrics=['accuracy'])                      # hasil akurasi (brp persen benar)

model.fit(X_train, y_train, epochs=10)    # seluruh data dipelajari 10 epochs
model.evaluate(X_test, y_test)            # uji dengan data yang belum pernah dilihat

# %% 3. Model tanpa hidden layer (sebagai pembanding)

model_tanpa_hidden = keras.Sequential([
    keras.Input(shape=(28, 28)),                    # gambar 28x28 piksel
    keras.layers.Flatten(),                         # gambar dijadikan satu baris
    keras.layers.Dense(5, activation='softmax')     # lapisan hasil: 5 neuron untuk angka 0-4
])

model_tanpa_hidden.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

model_tanpa_hidden.fit(X_train, y_train, epochs=10)     # belajar 10 epochs
model_tanpa_hidden.evaluate(X_test, y_test)             # di uji, lalu bandingkan akurasinya dengan model di atas

# %% 4. Simpan model dasar
# Disimpan supaya tidak perlu belajar ulang dari awal
model.save(r'C:\Users\fakhr\Downloads\Base_model_0_4.keras')

# %% 5. Model v1 (MNIST)

import numpy as np                                  # untuk hitungan angka dan array
from tensorflow import keras                        # untuk membuat model neural network

# Memuat MNIST lagi supaya cell ini bisa dijalankan sendiri
(Xa, ya), (Xb, yb) = keras.datasets.mnist.load_data()
X1_train, y1_train = Xa[ya < 5] / 255, ya[ya < 5]   # data latih: hanya angka 0-4, nilai piksel dibuat 0-1
X1_test, y1_test = Xb[yb < 5] / 255, yb[yb < 5]     # data uji: hanya angka 0-4

cnn = keras.Sequential([
    keras.Input(shape=(28, 28)),                    # gambar 28x28 piksel
    keras.layers.Reshape((28, 28, 1)),              # menambahkan 1 dimensi "warna"

    # Augmentasi: gambar latihan diubah sedikit-sedikit supaya model tidak menghafal
    keras.layers.RandomRotation(0.08, fill_mode='constant', fill_value=0.0),        # diputar acak
    keras.layers.RandomTranslation(0.1, 0.1, fill_mode='constant', fill_value=0.0), # digeser acak
    keras.layers.RandomZoom(0.15, fill_mode='constant', fill_value=0.0),            # diperbesar/dikecilkan acak

    keras.layers.Conv2D(32, 3, activation='relu'),  # mencari 32 jenis pola kecil
    keras.layers.MaxPooling2D(),                    # merangkum hasilnya jadi lebih kecil
    keras.layers.Conv2D(64, 3, activation='relu'),  # mencari 64 pola yang lebih rumit
    keras.layers.MaxPooling2D(),                    # merangkum lagi
    keras.layers.Flatten(),                         # diratakan jadi satu baris
    keras.layers.Dropout(0.3),                      # mematikan 30% neuron secara acak supaya tidak menghafal
    keras.layers.Dense(64, activation='relu'),      # hidden layer memakai 64 neuron
    keras.layers.Dense(5, activation='softmax')     # layer hasil: 5 neuron untuk angka 0-4
])

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

cnn.fit(X1_train, y1_train, epochs=15)              # belajar 15 putaran
cnn.evaluate(X1_test, y1_test)                      # diuji dengan gambar yang belum pernah dilihat

cnn.save(r'C:\Users\fakhr\Downloads\Latihan_CNN_0_4_v1.keras')
print('Tersimpan: Latihan_CNN_0_4_v1.keras')

# %% 6. Model v3 (MNIST + angka 5-9 + QuickDraw)

import os
import urllib.request
import numpy as np
from tensorflow import keras
from PIL import Image

# Pengacak angka (seed tetap supaya hasil bisa diulang)
rng = np.random.default_rng(0)

# Folder tempat file QuickDraw disimpan
FOLDER = r'C:\Users\fakhr\Downloads\quickdraw'
os.makedirs(FOLDER, exist_ok=True)

# Jenis coretan yang diajarkan sebagai "Bukan angka"
KATEGORI = ['headphones', 'hand', 'pencil', 'cloud', 'lightning', 'star', 'fish', 'tree']
PER_KATEGORI = 2500             # gambar latihan per kategori
UJI_PER_KATEGORI = 300          # gambar uji per kategori


def pusatkan(img):
    """Potong pas di sekeliling coretan, jadikan persegi, kecilkan ke 20x20, taruh di tengah 28x28 (gaya MNIST)."""
    ys, xs = np.where(img > 40)
    if len(ys) == 0:
        return None                                         # gambar kosong dilewati
    potongan = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = potongan.shape
    sisi = max(h, w)
    persegi = np.zeros((sisi, sisi), dtype='uint8')
    persegi[(sisi - h) // 2:(sisi - h) // 2 + h, (sisi - w) // 2:(sisi - w) // 2 + w] = potongan
    kecil = np.array(Image.fromarray(persegi).resize((20, 20), Image.LANCZOS)).astype('float32')
    if kecil.max() == 0:
        return None
    kanvas = np.zeros((28, 28), dtype='float32')
    kanvas[4:24, 4:24] = kecil / kecil.max()                # nilai dibuat 0-1
    return kanvas


def muat_quickdraw(n_latih, n_uji):
    # Memuat gambar QuickDraw. File yang belum ada akan diunduh dulu dari Google.
    latih, uji = [], []
    for nama in KATEGORI:
        path = os.path.join(FOLDER, f'{nama}.npy')
        if not os.path.exists(path):
            print('Mengunduh', nama, '...')
            urllib.request.urlretrieve(
                f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{nama}.npy', path)
        data = np.load(path, mmap_mode='r')
        # bagian awal untuk latihan, bagian setelahnya untuk uji (tidak tercampur)
        for sebagian, tujuan, awal in ((n_latih, latih, 0), (n_uji, uji, n_latih)):
            for baris in data[awal:awal + sebagian]:
                hasil = pusatkan(np.array(baris).reshape(28, 28))
                if hasil is not None:
                    tujuan.append(hasil)
    return np.array(latih, dtype='float32'), np.array(uji, dtype='float32')


def susun(X, y, qd, n_lain):
    # Menggabungkan angka 0-4, angka 5-9, dan coretan QuickDraw jadi satu data, lalu diacak
    benar = np.where(y < 5)[0]                                  # angka 0-4
    lain = rng.permutation(np.where(y >= 5)[0])[:n_lain]        # angka 5-9, jawabannya "Bukan angka"
    Xs = np.concatenate([X[benar], X[lain], qd])
    ys = np.concatenate([y[benar], np.full(len(lain), 5), np.full(len(qd), 5)])   # label 5 = "Bukan angka"
    acak = rng.permutation(len(Xs))
    return Xs[acak], ys[acak]


(Xa, ya), (Xb, yb) = keras.datasets.mnist.load_data()
Xa, Xb = Xa / 255, Xb / 255

qd_latih, qd_uji = muat_quickdraw(PER_KATEGORI, UJI_PER_KATEGORI)
X_train, y_train = susun(Xa, ya, qd_latih, n_lain=12000)
X_test, y_test = susun(Xb, yb, qd_uji, n_lain=2000)
print('Data latihan:', X_train.shape, '| per kelas:', np.bincount(y_train))

cnn = keras.Sequential([
    keras.Input(shape=(28, 28)),
    keras.layers.Reshape((28, 28, 1)),
    keras.layers.RandomRotation(0.03, fill_mode='constant', fill_value=0.0),        # diputar sedikit
    keras.layers.RandomTranslation(0.1, 0.1, fill_mode='constant', fill_value=0.0), # digeser
    keras.layers.RandomZoom(0.15, fill_mode='constant', fill_value=0.0),            # diperbesar/dikecilkan
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(6, activation='softmax')     # 6 neuron: angka 0-4 dan "Bukan angka"
])

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=12)                # belajar 12 putaran
cnn.evaluate(X_test, y_test)

cnn.save(r'C:\Users\fakhr\Downloads\Latihan_CNN_0_4_v2.keras')
print('Tersimpan: Latihan_CNN_0_4_v2.keras')

# %% 7. SEBAGAI MODEL AKHIR (MNIST + Angka 4 buatan + QuickDraw + Dataset sendiri)

import os
import numpy as np
from tensorflow import keras
from PIL import Image, ImageDraw

# Pengacak angka
rng = np.random.default_rng(0)

# Angka 5-9 MNIST tidak dipakai sama sekali
ANGKA_BUKAN = []

# Jumlah gambar buatan
JUMLAH_EMPAT_TERTUTUP = 3000   # 4 gaya ketikan
JUMLAH_SATU_KETIKAN = 2000     # 1 gaya ketikan

FOLDER_QD = r'C:\Users\fakhr\Downloads\quickdraw'
FOLDER_FOTO = r'C:\Users\fakhr\Downloads\data_foto'


def ri(a, b):
    return int(rng.integers(a, b))


def pusatkan(arr):
    # Potong gambar tepat di sekitar goresan, jadikan persegi, kecilkan ke 20x20, lalu taruh di center 28x28
    ys, xs = np.where(arr > 0)
    potongan = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = potongan.shape
    sisi = max(h, w)
    persegi = np.zeros((sisi, sisi), dtype='uint8')
    persegi[(sisi - h) // 2:(sisi - h) // 2 + h, (sisi - w) // 2:(sisi - w) // 2 + w] = potongan
    kecil = np.array(Image.fromarray(persegi).resize((20, 20), Image.LANCZOS))
    kanvas = np.zeros((28, 28), dtype='uint8')
    kanvas[4:24, 4:24] = kecil
    return kanvas.astype('float32') / 255


def goyang(titik, besar=3):
    # Geser tiap titik sedikit supaya garis tidak terlalu lurus
    return [(x + ri(-besar, besar + 1), y + ri(-besar, besar + 1)) for x, y in titik]


def gambar_empat_tertutup():
    # Angka 4 yang bagian atasnya tersambung
    u = 112
    img = Image.new('L', (u, u), 0)
    d = ImageDraw.Draw(img)
    tebal = ri(6, 13)
    xv, ytop, ybot = ri(55, 75), ri(12, 26), ri(86, 102)
    xl, ymid = ri(20, 40), ri(55, 72)
    if ri(0, 2) == 0:
        xr = xv + ri(8, 26)
        d.line([(xv + ri(-4, 5), ytop), (xl, ymid), (xr, ymid)], fill=255, width=tebal, joint='curve')
    else:
        d.line([(xv, ytop), (xl, ymid), (xv, ymid)], fill=255, width=tebal, joint='curve')
    d.line([(xv, ytop), (xv + ri(-5, 6), ybot)], fill=255, width=tebal)
    return pusatkan(np.array(img))


def gambar_satu_ketikan():
    # Angka 1 dengan beragam gaya tulisan
    u = 112
    img = Image.new('L', (u, u), 0)
    d = ImageDraw.Draw(img)
    tebal = ri(5, 12)
    ytop, ybot = ri(12, 24), ri(88, 102)                       # ujung atas dan bawah batang
    miring = rng.uniform(-0.35, 0.35)                          # kemiringan batang
    xc = ri(48, 64)                                            # posisi tengah batang
    xtop = xc + int(miring * (ybot - ytop) / 2)
    xbot = xc - int(miring * (ybot - ytop) / 2)
    d.line(goyang([(xtop, ytop), (xbot, ybot)], 2), fill=255, width=tebal)
    if ri(0, 10) < 8:
        d.line(goyang([(xtop, ytop), (xtop - ri(10, 22), ytop + ri(8, 20))], 2), fill=255, width=tebal)
    if ri(0, 10) < 6:
        setengah = ri(14, 30)
        d.line(goyang([(xbot - setengah, ybot), (xbot + setengah, ybot)], 2), fill=255, width=tebal)
    return pusatkan(np.array(img))


def gambar_acak():
    # Bentuk buatan untuk opsi "Bukan angka"
    u = 112
    img = Image.new('L', (u, u), 0)
    d = ImageDraw.Draw(img)
    jenis = ri(0, 4)
    tebal = ri(6, 16)

    def titik():
        return (ri(10, u - 10), ri(10, u - 10))

    if jenis == 0:
        x1, dx, y1 = ri(10, 40), ri(40, 62), ri(30, 82)
        dy = ri(-dx // 2, dx // 2 + 1)
        d.line([(x1, y1), (x1 + dx, int(np.clip(y1 + dy, 10, 100)))], fill=255, width=tebal)
    elif jenis == 1:
        d.line([titik() for _ in range(ri(4, 9))], fill=255, width=tebal, joint='curve')
    elif jenis == 2:
        a, b = titik(), titik()
        d.ellipse([min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]) + 12, max(a[1], b[1]) + 12], fill=255)
    else:
        d.line([titik() for _ in range(ri(8, 16))], fill=255, width=tebal)
    return pusatkan(np.array(img))


# Data MNIST
(Xa, ya), (Xb, yb) = keras.datasets.mnist.load_data()
Xa, Xb = Xa / 255, Xb / 255


def susun_mnist(X, y):
    pilih = np.isin(y, [0, 1, 2, 3, 4] + ANGKA_BUKAN)
    return X[pilih].astype('float32'), np.where(y[pilih] < 5, y[pilih], 5)


X_m, y_m = susun_mnist(Xa, ya)
X_mt, y_mt = susun_mnist(Xb, yb)

# Data buatan
X_4 = np.array([gambar_empat_tertutup() for _ in range(JUMLAH_EMPAT_TERTUTUP)], dtype='float32')
X_4t = np.array([gambar_empat_tertutup() for _ in range(500)], dtype='float32')
X_1 = np.array([gambar_satu_ketikan() for _ in range(JUMLAH_SATU_KETIKAN)], dtype='float32')
X_1t = np.array([gambar_satu_ketikan() for _ in range(300)], dtype='float32')
X_j = np.array([gambar_acak() for _ in range(8000)], dtype='float32')
X_jt = np.array([gambar_acak() for _ in range(1500)], dtype='float32')

# Data QuickDraw
X_q, X_qt = np.zeros((0, 28, 28), dtype='float32'), np.zeros((0, 28, 28), dtype='float32')
kategori = ['headphones', 'hand', 'pencil', 'cloud', 'lightning', 'star', 'fish', 'tree']
daftar_q, daftar_qt = [], []

for nama in kategori:
    path = os.path.join(FOLDER_QD, nama + '.npy')
    if not os.path.exists(path):
        continue
    data = np.load(path, mmap_mode='r')
    for baris in data[:1500]:
        a = np.array(baris).reshape(28, 28)
        if (a > 40).any():
            daftar_q.append(pusatkan(np.where(a > 40, a, 0).astype('uint8')))
    for baris in data[2500:2700]:
        a = np.array(baris).reshape(28, 28)
        if (a > 40).any():
            daftar_qt.append(pusatkan(np.where(a > 40, a, 0).astype('uint8')))

if daftar_q:
    X_q, X_qt = np.array(daftar_q, dtype='float32'), np.array(daftar_qt, dtype='float32')

print('QuickDraw dipakai:', len(X_q), 'gambar')

# Pengolah foto
from scipy import ndimage as ndi                    # untuk mengolah gambar

try:
    from skimage.morphology import skeletonize      # untuk menipiskan goresan jadi 1 piksel
except ImportError:
    skeletonize = None
    print('PERINGATAN: scikit-image belum terpasang, goresan tidak ditipiskan (beda dengan web)')

# Ketebalan goresan target agar goresan yg di upload user mirip dgn data latih
TEBAL_TARGET = 0.12

# Angka hanya dianggap terbaca kalau keyakinan model di atas 75%
BATAS_YAKIN = 0.75

# Jumlah angka maksimal yang dibaca dalam satu foto
MAKS_ANGKA = 30

# Goresan yang jaraknya dekat dianggap satu angka
JARAK_GABUNG = 0.017

# Potongan goresan yang lebih kecil dari ini dibuang
BUANG_BINTIK = 0.00004

# Objek yang jumlah goresannya kurang dari ini dibuang
LUAS_REL = 0.1

# Periksa background
TOLERANSI_WARNA = 0.05    # selisih warna yang masih diterima
TERANG_MIN = 0.5          # batas bawah kecerahan bg dibanding kertas
TERANG_MAX = 1.8          # batas atas kecerahan bg dibanding kertas

# Menyeragamkan warna background
SEBARAN_LATAR = 0.2

# Seberapa jauh dari goresan area background di cek
RING_LOKAL = 0.025

# Objek yang terlalu dekat dengan tepi foto dibuang
MARGIN_TEPI = 0.015

# Dua goresan yang celahnya lebih kecil dari ini disambung garis
JARAK_SAMBUNG = 0.18

# Fungsi Pengolah Foto
def otsu(arr):
    # Mencari "garis batas" terang-gelap
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    jumlah_semua = np.dot(np.arange(256), hist)
    jumlah_b, bobot_b, terbaik, batas = 0.0, 0, 0.0, 0
    for t in range(256):
        bobot_b += hist[t]
        if bobot_b == 0:
            continue
        bobot_f = total - bobot_b
        if bobot_f == 0:
            break
        jumlah_b += t * hist[t]
        rata_b = jumlah_b / bobot_b
        rata_f = (jumlah_semua - jumlah_b) / bobot_f
        selisih = bobot_b * bobot_f * (rata_b - rata_f) ** 2
        if selisih > terbaik:
            terbaik, batas = selisih, t
    return batas


def latar_mirip_kertas(asli, label, i, mask, sl, kertas, sisi_foto):
    # Memeriksa background di sekeliling objek
    h0, w0 = asli.shape[:2]
    jarak_ring = max(8, int(sisi_foto * RING_LOKAL))
    p = jarak_ring + 4
    y0, y1 = max(sl[0].start - p, 0), min(sl[0].stop + p, h0)
    x0, x1 = max(sl[1].start - p, 0), min(sl[1].stop + p, w0)

    blob = label[y0:y1, x0:x1] == i
    goresan = mask[y0:y1, x0:x1] & blob
    if goresan.sum() == 0:
        return True

    semua_tinta = ndi.binary_dilation(mask[y0:y1, x0:x1], iterations=2)
    jarak = ndi.distance_transform_edt(~goresan)
    cincin = (jarak > 3) & (jarak <= jarak_ring) & ~semua_tinta
    if cincin.sum() < 30:
        return True

    piksel = asli[y0:y1, x0:x1][cincin].astype('float32')
    warna = np.median(piksel, axis=0)
    
    kroma_obj = warna / max(float(warna.sum()), 1.0)
    kroma_kertas = kertas / max(float(kertas.sum()), 1.0)
    selisih_warna = float(np.abs(kroma_obj - kroma_kertas).max())

    # Bandingkan brightness bg dengan brightness kertas utama
    rasio = (float(warna.mean()) + 1) / (float(kertas.mean()) + 1)

    # Kalau warna atau brightness jauh beda dari kertas, tolak
    if not (selisih_warna <= TOLERANSI_WARNA and TERANG_MIN <= rasio <= TERANG_MAX):
        return False

    # Kalau ada dua warna yang jauh berbeda di sekitar goresan, itu tepi benda
    p10 = np.percentile(piksel, 10, axis=0)
    p90 = np.percentile(piksel, 90, axis=0)
    sebaran = float(np.max((p90 - p10) / (piksel.mean(axis=0) + 1)))
    return sebaran <= SEBARAN_LATAR


def sambungkan(g):
    """Menyambung potongan goresan yang terpisah dengan garis di celahnya"""
    label, n = ndi.label(g, structure=np.ones((3, 3)))
    if n <= 1 or n > 40:
        return g

    titik = []
    for i in range(1, n + 1):
        ys, xs = np.where(label == i)
        pts = np.stack([ys, xs], axis=1)
        if len(pts) > 150:
            pts = pts[np.linspace(0, len(pts) - 1, 150).astype(int)]
        titik.append(pts)

    # Hitung jarak terdekat antara tiap potongan
    jarak = np.full((n, n), np.inf)
    pasangan = {}
    for a in range(n):
        for b in range(a + 1, n):
            d = np.sqrt(((titik[a][:, None, :] - titik[b][None, :, :]) ** 2).sum(-1))
            idx = np.unravel_index(np.argmin(d), d.shape)
            jarak[a, b] = jarak[b, a] = d[idx]
            pasangan[(a, b)] = (titik[a][idx[0]], titik[b][idx[1]])

    # Pastikan semua potongan tersambung
    sambung = set()
    terhubung = {0}
    while len(terhubung) < n:
        terbaik = None
        for a in terhubung:
            for b in range(n):
                if b not in terhubung and (terbaik is None or jarak[a, b] < terbaik[0]):
                    terbaik = (jarak[a, b], a, b)
        sambung.add((min(terbaik[1], terbaik[2]), max(terbaik[1], terbaik[2])))
        terhubung.add(terbaik[2])

    # Celah lain yang cukup dekat juga disambung
    ukuran = max(g.shape)
    for (a, b), d in ((k, jarak[k]) for k in pasangan):
        if d <= JARAK_SAMBUNG * ukuran:
            sambung.add((a, b))

    # Gambar garis lurus pendek di tiap celah yang harus disambung
    img = Image.fromarray(g.astype('uint8') * 255)
    gambar = ImageDraw.Draw(img)
    for a, b in sambung:
        p, q = pasangan[(a, b)]
        gambar.line([(int(p[1]), int(p[0])), (int(q[1]), int(q[0]))], fill=255, width=1)
    return np.array(img) > 0


def rapikan(g):
    """Menyambung goresan putus-putus, lalu menipiskannya jadi garis 1 piksel."""
    g = sambungkan(g)
    if skeletonize is not None:
        g = skeletonize(g)
    return g.astype(bool)


def normalisasi(g):
    """Mengubah satu goresan menjadi gambar 28x28px."""
    g = np.pad(g, 30)     # menambahkan ruang kosong di sekeliling supaya aman saat diproses
    ys, xs = np.where(g)
    if len(ys) == 0:
        return None
    sisi = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)

    # Samakan ketebalan goresan dengan MNIST
    luas_g = g.sum()
    keliling = (g & ~ndi.binary_erosion(g)).sum()
    lebar_goresan = 2 * luas_g / max(keliling, 1)
    target = TEBAL_TARGET * sisi
    tebal = g
    if lebar_goresan < target:
        tebal = ndi.binary_dilation(g, iterations=max(1, int(round((target - lebar_goresan) / 2))))
    elif lebar_goresan > target * 1.5:
        kurus = ndi.binary_erosion(g, iterations=max(1, int(round((lebar_goresan - target) / 2))))
        if kurus.sum() > 0:
            tebal = kurus

    # Potong pas di sekeliling angka, lalu jadikan persegi
    ys, xs = np.where(tebal)
    potongan = tebal[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype('float32') * 255
    h, w = potongan.shape
    sisi = max(h, w)
    kotak = np.zeros((sisi, sisi), dtype='float32')
    kotak[(sisi - h) // 2:(sisi - h) // 2 + h, (sisi - w) // 2:(sisi - w) // 2 + w] = potongan

    # Perkecil atau perbesar ke 20x20 piksel
    kecil20 = Image.fromarray(kotak.clip(0, 255).astype('uint8')).resize((20, 20), Image.LANCZOS)
    kecil20 = np.array(kecil20).astype('float32')
    if kecil20.max() == 0:
        return None
    kecil20 = kecil20 / kecil20.max() * 255

    # Taruh di tengah kanvas 28x28, lalu di jadikan center
    kanvas = np.zeros((28, 28), dtype='float32')
    kanvas[4:24, 4:24] = kecil20
    cy, cx = ndi.center_of_mass(kanvas)
    kanvas = ndi.shift(kanvas, (13.5 - cy, 13.5 - cx), order=1)
    return (kanvas / 255).clip(0, 1).astype('float32')


def urutkan(kandidat):
    """Mengurutkan objek, kalau baris dari atas ke bawah, di dalam baris dari kiri ke kanan."""
    tinggi_med = np.median([k['tinggi'] for k in kandidat])
    kandidat = sorted(kandidat, key=lambda k: k['cy'])
    baris = []
    for k in kandidat:
        # Kalau tingginya posisi hampir sama dengan baris sebelumnya, masuk baris yang sama
        if baris and abs(k['cy'] - np.mean([x['cy'] for x in baris[-1]])) < 0.6 * tinggi_med:
            baris[-1].append(k)
        else:
            baris.append([k])
    return [sorted(b, key=lambda k: k['cx']) for b in baris]


def siapkan(foto):
    """Mengembalikan."""

    # 1. Ambil foto dan perkecil kalau sangat besar
    img = Image.fromarray(foto).convert('RGB')
    img.thumbnail((800, 800))
    asli = np.array(img)
    arr = asli.astype('float32')
    h0, w0 = arr.shape[:2]
    sisi_foto = min(h0, w0)
    
    peta = asli.copy()

    # Warna kertas utama foto
    kertas = np.median(asli.reshape(-1, 3), axis=0).astype('float32')

    # 2. Haluskan bintik"/noise sedikit
    arr = ndi.gaussian_filter(arr, sigma=(0.7, 0.7, 0))

    # 3. Mencari goresan
    ukuran = max(9, int(sisi_foto * 0.08)) | 1
    gelap = np.zeros((h0, w0), dtype='float32')
    terang = np.zeros((h0, w0), dtype='float32')
    for c in range(3):
        kanal = arr[:, :, c]
        gelap += (ndi.grey_closing(kanal, size=(ukuran, ukuran)) - kanal) ** 2
        terang += (kanal - ndi.grey_opening(kanal, size=(ukuran, ukuran))) ** 2
    gelap = np.sqrt(gelap)
    terang = np.sqrt(terang)

    sinyal = terang if terang.sum() > 3 * gelap.sum() else gelap

    if sinyal.max() < 15:
        return [], peta, 'Tidak ada tulisan yang terdeteksi di foto.'
    arr = sinyal / sinyal.max() * 255

    # 4. Mencari nilai yang dianggap goresan
    batas = max(otsu(arr) * 0.5, 30)
    mask = arr > batas

    # 4b. Membuang bintik/noise
    label_m, n_m = ndi.label(mask, structure=np.ones((3, 3)))
    if n_m > 0:
        nomor = np.arange(1, n_m + 1)
        luas_potongan = np.array(ndi.sum(mask.astype('float32'), label_m, index=nomor))
        luas_min_potongan = max(6, int(sisi_foto ** 2 * BUANG_BINTIK))
        mask = np.isin(label_m, nomor[luas_potongan >= luas_min_potongan])
    if not mask.any():
        return [], peta, 'Tidak ada tulisan yang terdeteksi di foto (hanya bintik halus).'
    
    # 5. Mengumpulkan goresan yang berdekatan jadi satu objek
    r = max(3, int(sisi_foto * JARAK_GABUNG))
    gabung = ndi.distance_transform_edt(~mask) <= r
    label, jumlah = ndi.label(gabung)                 # memberi nomor untuk objek yang terpisah

    # 6. Memeriksa tiap objek
    margin = max(2, int(sisi_foto * MARGIN_TEPI))
    kandidat = []
    ditolak_latar = 0
    for i, sl in enumerate(ndi.find_objects(label), start=1):
        if sl is None:
            continue
        g = mask[sl] & (label[sl] == i)
        luas = int(g.sum())
        if luas < 12:
            continue   # terlalu kecil, dianggap noda

        # Terlalu dekat tepi foto
        if (sl[0].start < margin or sl[1].start < margin
                or sl[0].stop > h0 - margin or sl[1].stop > w0 - margin):
            continue

        ys, xs = np.where(g)
        tinggi = ys.max() - ys.min() + 1
        lebar = xs.max() - xs.min() + 1

        # Ukuran harus wajar
        besar = max(tinggi, lebar) / sisi_foto
        if besar < 0.04 or besar > 0.95:
            continue

        # Bentuk angka tidak boleh pipih dan lebar
        rasio = tinggi / lebar
        if rasio < 0.6 or rasio > 15:
            continue

        # Angka berupa goresan
        isi = luas / (tinggi * lebar)
        if rasio < 3 and isi > 0.65:
            continue

        # Background di sekeliling objek harus seragam di kedua sisi
        if not latar_mirip_kertas(asli, label, i, mask, sl, kertas, sisi_foto):
            ditolak_latar += 1
            continue

        kandidat.append(dict(
            sl=sl, g=g, luas=luas, tinggi=tinggi,
            ukuran=max(tinggi, lebar),
            cy=(sl[0].start + sl[0].stop) / 2,
            cx=(sl[1].start + sl[1].stop) / 2,
        ))

    # Kalau tidak ada objek yang terdeteksi, tampilkan alasannya
    if not kandidat:
        if ditolak_latar:
            return [], peta, 'Ada goresan di foto, tapi tidak tertulis di foto yang kamu kirim.'
        return [], peta, 'Ada objek di foto, tapi bentuknya tidak mirip angka.'

    # Buang titik/pengotor yang jauh lebih kecil dibanding objek terbesar
    terbesar = max(k['ukuran'] for k in kandidat)
    kandidat = [k for k in kandidat if k['ukuran'] >= 0.35 * terbesar]
    # Buang juga objek yang jumlah goresannya sangat sedikit
    luas_terbesar = max(k['luas'] for k in kandidat)
    kandidat = [k for k in kandidat if k['luas'] >= LUAS_REL * luas_terbesar]
    kandidat = sorted(kandidat, key=lambda k: k['luas'], reverse=True)[:MAKS_ANGKA]

    # 7. Urutkan, lalu ubah tiap objek jadi gambar 28x28px
    baris_hasil = []
    for baris in urutkan(kandidat):
        item = []
        for k in baris:
            k['arr'] = normalisasi(rapikan(k['g']))
            if k['arr'] is not None:
                item.append(k)
        if item:
            baris_hasil.append(item)

    return baris_hasil, peta, ''


# Dataset yang di foto sendiri
X_f, y_f = np.zeros((0, 28, 28), dtype='float32'), np.zeros(0, dtype=int)

try:
    NAMA = ['0', '1', '2', '3', '4', 'bukan']
    daftar_x, daftar_y = [], []

    for kelas, nama in enumerate(NAMA):
        folder = os.path.join(FOLDER_FOTO, 'latih', nama)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                foto = np.array(Image.open(os.path.join(folder, f)).convert('RGB'))
                baris, _, _ = siapkan(foto)          # foto dipotong jadi gambar 28x28 per angka
                for b in baris:
                    for k in b:
                        daftar_x.append(k['arr'])
                        daftar_y.append(kelas)

    if daftar_x:
        X_f = np.repeat(np.array(daftar_x, dtype='float32'), 15, axis=0)
        y_f = np.repeat(np.array(daftar_y), 15)
        X_f = np.clip(X_f + rng.normal(0, 0.03, X_f.shape), 0, 1).astype('float32')

except Exception as e:
    print('Foto sendiri dilewati:', type(e).__name__, e)

print('Foto sendiri dipakai:', len(X_f), 'gambar (setelah diulang)')

# Gabungkan dta
X_train = np.concatenate([X_m, X_4, X_1, X_j, X_q, X_f]).astype('float32')
y_train = np.concatenate([
    y_m,
    np.full(len(X_4), 4),
    np.full(len(X_1), 1),
    np.full(len(X_j), 5),
    np.full(len(X_q), 5),
    y_f
])

acak = rng.permutation(len(X_train))
X_train, y_train = X_train[acak], y_train[acak]

X_test = np.concatenate([X_mt, X_4t, X_1t, X_jt, X_qt]).astype('float32')
y_test = np.concatenate([
    y_mt,
    np.full(len(X_4t), 4),
    np.full(len(X_1t), 1),
    np.full(len(X_jt), 5),
    np.full(len(X_qt), 5)
])

print('Data latihan:', X_train.shape, '| per kelas:', np.bincount(y_train, minlength=6))

# Model (susunannya sama dengan model v3)
cnn = keras.Sequential([
    keras.Input(shape=(28, 28)),
    keras.layers.Reshape((28, 28, 1)),
    keras.layers.RandomRotation(0.03, fill_mode='constant', fill_value=0.0),
    keras.layers.RandomTranslation(0.1, 0.1, fill_mode='constant', fill_value=0.0),
    keras.layers.RandomZoom(0.15, fill_mode='constant', fill_value=0.0),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(6, activation='softmax')
])

cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cnn.fit(X_train, y_train, epochs=12)
cnn.evaluate(X_test, y_test)

# Disimpan sebagai model akhir (v3)
cnn.save(r'C:\Users\fakhr\Downloads\Latihan_CNN_0_4_v3.keras')
print('Tersimpan: Latihan_CNN_0_4_v3.keras')


# Pengecekan otomatis
def persen(X, kelas):
    tebakan = np.argmax(cnn.predict(X, verbose=0), axis=1)
    return 100 * np.mean(tebakan == kelas)


print('1 data MNIST dibaca 1       : %.1f%%' % persen(Xb[yb == 1], 1))
print('4 data MNIST dibaca 4       : %.1f%%' % persen(Xb[yb == 4], 4))
print('2 data MNIST dibaca 2       : %.1f%%' % persen(Xb[yb == 2], 2))
print('1 gaya ketikan dibaca 1     : %.1f%%' % persen(X_1t, 1))
print('4 gaya ketikan dibaca 4     : %.1f%%' % persen(X_4t, 4))