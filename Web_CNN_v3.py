# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 17:48:15 2026

@author: fakhr
"""
# WEBSITE CNN 0-4 VERSI 3

# Membuat Website

# Alur kerja:
# 1. Pengguna upload atau foto kertas berisi angka tulisan tangan.
# 2. Kode mencari goresan tulisan di foto (fungsi siapkan).
# 3. Tiap angka dipotong dan diubah jadi gambar kecil 28x28 seperti MNIST.
# 4. Model menebak tiap gambar: angka 0-4, ragu, atau "Bukan angka".
# 5. Hasilnya ditampilkan di web.


import traceback                                    # untuk menampilkan detail error
import numpy as np                                  # untuk hitung-hitungan angka dan tabel (array)
import gradio as gr                                 # untuk membuat halaman web
from PIL import Image, ImageDraw, ImageFont         # untuk membuka foto, menggambar, dan menulis teks di gambar
from scipy import ndimage as ndi                    # untuk mengolah gambar
from tensorflow import keras                        # untuk memanggil model yang sudah dilatih

try:
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None

# Memuat model
model = keras.models.load_model(r'C:\Users\fakhr\Downloads\proyek_ 0-4\Latihan_CNN_0_4_v3.keras')
JUMLAH_KELAS = model.output_shape[-1]

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

# Objek yang isinya terlalu jarang dibuang
ISI_MIN = 0.04

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

try:
    FONT = ImageFont.load_default(size=30)
except TypeError:
    FONT = ImageFont.load_default()

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
    """Mengurutkan objek: baris dari atas ke bawah, di dalam baris dari kiri ke kanan."""
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
    """Mengembalikan (daftar baris berisi objek, peta pengecekan, catatan)."""

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

    # 4b. Buang bintik
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

        # Bg di sekeliling objek harus seragam di kedua sisi
        if not latar_mirip_kertas(asli, label, i, mask, sl, kertas, sisi_foto):
            ditolak_latar += 1
            continue

        kandidat.append(dict(
            sl=sl, g=g, luas=luas, tinggi=tinggi,
            ukuran=max(tinggi, lebar),
            cy=(sl[0].start + sl[0].stop) / 2,        # posisi tengah (atas-bawah)
            cx=(sl[1].start + sl[1].stop) / 2,        # posisi tengah (kiri-kanan)
        ))

    # Kalau tidak ada objek yang lolos, tulis alasannya
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

    # 7. Urutkan, lalu ubah tiap objek jadi gambar 28x28
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


def susun_gambar(tiles):
    """Menyusun banyak gambar 28x28 menjadi satu gambar."""
    per_baris = 8
    kolom = min(per_baris, len(tiles))
    kosong = np.pad(np.zeros_like(tiles[0]), 4, constant_values=80)
    tiles = [np.pad(t, 4, constant_values=80) for t in tiles]
    baris_img = []
    for s in range(0, len(tiles), per_baris):
        row = tiles[s:s + per_baris]
        row = row + [kosong] * (kolom - len(row))
        baris_img.append(np.concatenate(row, axis=1))
    return np.concatenate(baris_img, axis=0)


# FUNGSI UTAMA YANG DIPANGGIL WEB

def tebak(foto):
    if foto is None:
        return '', None, None

    try:
        baris, peta, catatan = siapkan(foto)

        if not baris:
            return f'## Tidak ada angka yang diprediksi\n{catatan}', None, peta

        semua = [k for b in baris for k in b]
        batch = np.stack([k['arr'] for k in semua]).reshape(-1, 28, 28)
        hasil = model.predict(batch, verbose=0)
        
        for k, h in zip(semua, hasil):
            kelas = int(np.argmax(h))
            k['kelas'] = kelas
            k['yakin'] = float(h[kelas])
            if JUMLAH_KELAS == 6 and kelas == 5:
                k['status'] = 'bukan'
            elif k['yakin'] < BATAS_YAKIN:
                k['status'] = 'ragu'
            else:
                k['status'] = 'angka'

        warna = {'angka': (0, 200, 0), 'ragu': (255, 200, 0), 'bukan': (255, 0, 0)}

        rincian = []
        for no, k in enumerate(semua, start=1):
            sub = peta[k['sl']]
            sub[ndi.binary_dilation(k['g'], iterations=2)] = warna[k['status']]
            if k['status'] == 'angka':
                rincian.append(f"{no}. **{k['kelas']}** ({k['yakin'] * 100:.0f}%)")
            elif k['status'] == 'ragu':
                rincian.append(f"{no}. **Model ragu untuk menebaknya** (Tebakan terdekat {k['kelas']}, {k['yakin'] * 100:.0f}%)")
            else:
                rincian.append(f"{no}. Bukan angka 0-4 ({k['yakin'] * 100:.0f}%)")

        # Tulis nomor urut di dekat tiap objek pada gambar
        im = Image.fromarray(peta)
        d = ImageDraw.Draw(im)
        for no, k in enumerate(semua, start=1):
            d.text((k['sl'][1].start, max(k['sl'][0].start - 34, 0)), str(no),
                   fill=warna[k['status']], font=FONT)
        peta = np.array(im)

        # Gambar yang benar-benar dilihat model
        tiles = [np.array(Image.fromarray((k['arr'] * 255).astype('uint8')).resize((84, 84), Image.NEAREST))
                 for k in semua]
        dilihat = susun_gambar(tiles)

        # Teks hasil: hanya judul dan daftar bernomor
        md = '## Hasil\n\n' + '\n'.join(rincian)
        return md, dilihat, peta

    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f'{type(e).__name__}: {e}')
        
# INTERFACE WEB

web = gr.Interface(
    fn=tebak,
    inputs=gr.Image(
        type='numpy',
        label='Upload foto angka 0-4',
        sources=['upload', 'webcam', 'clipboard'],
        webcam_options=gr.WebcamOptions(mirror=False),
    ),
    outputs=[
        gr.Markdown(),
        gr.Image(label='Yang dilihat model (sesuai nomor)'),
        gr.Image(label='Pengecekan (Warna hijau = angka, warna kuning = ragu, warna merah = bukan angka)'),
    ],
    title='Sistem Klasifikasi Angka berbasis Neural Network V3',
    description='Upload atau foto angka 0-4, lalu model akan menebaknya!',
)

if __name__ == '__main__':
    web.launch(share=True)