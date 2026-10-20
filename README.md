# 01 — Make Sense of Data with Analysis and AI

## Kesesuaian dengan transkrip

Proyek latihan ini mengambil studi kasus **PT Maju Bersama**, distributor fiktif yang ingin memahami mengapa revenue dapat naik sementara gross profit dan arus kas belum tentu membaik. Struktur data sintetis mengikuti gambaran transkrip: 120 pelanggan, 24 produk, dan sekitar 4.000 invoice dengan 6 baris produk per invoice.

Workflow sengaja menunjukkan hal-hal yang dibahas narasumber:

1. **Business understanding** — mengubah pertanyaan bisnis menjadi pertanyaan analitik.
2. **Data preparation** — menangani baris duplikat, nilai kosong, dan kategori yang penulisannya tidak konsisten.
3. **Perbedaan granularitas** — mengagregasi invoice-line menjadi satu baris per invoice sebelum menggabungkannya dengan data pelanggan agar total tidak membengkak akibat join.
4. **Descriptive analytics** — melihat tren revenue, gross profit, piutang, dan tingkat invoice telat.
5. **Diagnostic analytics** — membandingkan piutang dan tingkat keterlambatan berdasarkan pelanggan/segmen/wilayah.
6. **Predictive analytics** — melatih model baseline untuk memperkirakan risiko telat bayar. Informasi hasil pembayaran tidak digunakan sebagai fitur model (menghindari target leakage).
7. **Prescriptive-style follow-up** — menghasilkan daftar invoice belum dibayar yang perlu diperiksa lebih dulu. Ini hanya contoh prioritas; data sintetis bukan dasar keputusan keuangan nyata.

Transkrip memperkenalkan KNIME Analytics Platform sebagai alat workflow. Versi ini memakai Python agar logika pembersihan, join, analisis, serta model mudah dibaca. Untuk mempraktikkan workflow visual di KNIME, lihat tautan referensi di bawah.

## Cara menjalankan

Butuh Python 3.10 atau lebih baru.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_analysis.py
```

Skrip membuat folder `data/` berisi data sintetis mentah dan folder `outputs/` berisi dataset bersih, KPI bulanan, hasil diagnostik, evaluasi model, daftar tindak lanjut, dan grafik PNG. Jalankan ulang untuk membuat hasil yang sama secara reproducible.

## Repo dan materi referensi

- [KNIME — Your first workflow (dokumentasi resmi)](https://docs.knime.com/get-started/first-workflow) — cocok untuk memahami workflow visual, membaca data, filter, join, dan visualisasi.
- [BscCanCode/KNIME_PROJECT](https://github.com/BscCanCode/KNIME_PROJECT) — contoh komunitas workflow KNIME end-to-end; bukan repo resmi narasumber dan belum diverifikasi berasal dari webinar.

**Catatan integritas:** data di sini dibuat oleh program untuk belajar, bukan data PT Maju Bersama sungguhan dan bukan kode yang diklaim milik pembicara.
