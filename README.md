# Smart Retail — Deteksi & Tracking

Template ini berisi skrip dan modul untuk deteksi dan tracking objek pada aplikasi smart retail (rak, tangan, dsb.). README ini disederhanakan dan tidak menyertakan referensi ke implementasi tracker yang tidak relevan.

Ringkasan singkat: install dengan `pip install -e .`, letakkan model di `smartfridge/checkpoint/`, lalu jalankan `python run_tracking.py`.

## Prerequisites

- Python 3.8+
- (Opsional, direkomendasikan) GPU dengan CUDA
- Conda atau `venv` untuk lingkungan terisolasi

## Instalasi

```bash
git clone <repository-url>
cd template
pip install -e .
```

## Struktur proyek (ringkas — level 2)
```
template/
├── smartfridge/
│   ├── checkpoint/               # Model checkpoints
│   │   ├── best (1).pt           # YOLO detection
│   │   └── hand_landmarker.task # MediaPipe hand
│   ├── trackers/
│   │   └── hybridsort/           # Hybridsort tracker
│   │   └── ocsort/               # OCsort tracker
│   │   └── bytetrack.py          # Bytetrack tracker
│   │   └── botsort.py            # Botsort tracker
│   └── models/yolo/detect/
│       ├── config.py             # Config file
│       ├── helper.py             # Utility functions
│       └── predict.py            # Predictor class
│       └── helper.py             # Helper func
│       └── config.py             # Config management
│       └── tracker.py            # Tracker class
├── run_tracking.py               # Main script
└── README.md
```
Keterangan singkat:
- `cfg/`: berisi `default.yaml` dan presets (mis. `bytetrack.yaml`, `botsort.yaml`, `hybridsort.yaml`) untuk memilih konfigurasi tracker dan model.
- `trackers/`: berisi implementasi tracker dan helper; contohnya `byte_tracker.py`, `bot_sort.py`, `hybrid_sort/` (kode reappearance berada di sini), `oc_sort/`.

## Trackers tersedia

- `bytetrack` — tracker berbasis ByteTrack
- `botsort` — BoTSORT implementation
- `hybridsort` — HybridSort (direkomendasikan untuk smart fridge)
- `ocsort` — OCSORT (opsional)

Catatan penting: `hybridsort` mengandung fitur reappearance yang membantu menjaga konsistensi ID ketika objek sementara hilang atau tertutup — berguna untuk skenario smart fridge di mana objek (produk, tangan) dapat terhalang atau bergerak singkat.

## Konfigurasi singkat (contoh)

```python
TRACKER = 'hybridsort'  # 'bytetrack' | 'botsort' | 'hybridsort' | 'ocsort'
CAMERA_FROM_TOP = True
HAND_LANDMARKER_PATH = 'smartfridge/checkpoint/hand_landmarker.task'
YOLO_PATH = 'smartfridge/checkpoint/best.pt'
```

## Contoh penggunaan singkat

```bash
python run_tracking.py
```

## Output

- Hasil video dan label biasanya tersimpan di `runs/detect/track*/`.

## Troubleshooting singkat

- Module not found: jalankan `pip install -e .` dari folder project
- Model file not found: pastikan model berada di `smartfridge/checkpoint/` dan path di konfigurasi cocok

## Development

- Setelah `pip install -e .`, edit kode di `smartfridge/` dan langsung jalankan tanpa reinstall
- Lokasi penting:
  - `smartfridge/engine/` — predictor, model wrapper
  - `smartfridge/data/` — loader, util
  - `smartfridge/cfg/` — konfigurasi (default.yaml, models/trackers presets)
  - `smartfridge/trackers/hybrid_sort/` — fungsi reappearance dan implementasi HybridSort

