# Smart Retail Detection & Tracking System

Sistem deteksi dan tracking objek untuk aplikasi smart retail dengan fitur hand landmark detection dan shelf detection.

## 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Conda atau virtual environment

## 🚀 Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd template
```

### 2. Install in Editable Mode

**PENTING:** Selalu install dalam mode editable agar perubahan kode langsung terdeteksi.

```bash
pip install -e .
```

## 📁 File Structure

```
template/
├── ultralytics/
│   ├── checkpoint/                          # Model checkpoints
│   │   ├── best (1).pt                     # YOLO detection model
│   │   └── hand_landmarker.task            # MediaPipe hand model
│   ├── trackers/
│   │   └── deep_sort_pytorch/              # DeepSort tracker
│   │       ├── configs/
│   │       │   └── deep_sort.yaml
│   │       └── deep_sort/deep/checkpoint/
│   │           └── ckpt.t7                 # DeepSort ReID model
│   └── models/yolo/detect/
│       ├── config.py                        # ⚙️ CONFIGURATION FILE
│       ├── helper.py                        # Utility functions
│       └── predict.py                       # Main predictor
├── run_tracking.py                          # Example usage script
└── README.md
```

## ⚙️ Configuration

### Edit File: `ultralytics/models/yolo/detect/config.py`

#### 1. **Pilih Tracker**

```python
# Set to True untuk DeepSort, False untuk ByteTrack/BoTSORT
USE_DEEPSORT = False  # Ubah sesuai kebutuhan
```

#### 2. **Konfigurasi Posisi Kamera**

```python
# True = kamera dari atas, False = kamera dari bawah
CAMERA_FROM_TOP = True  # Ubah sesuai instalasi kamera
```

#### 3. **Model Paths** (Auto-configured)

Path model otomatis disesuaikan berdasarkan struktur folder:

```python
# MediaPipe Hand Detection Model
HAND_LANDMARKER_MODEL_PATH = 'ultralytics/checkpoint/hand_landmarker.task'

# DeepSort ReID Checkpoint
DEEPSORT_REID_CKPT = 'ultralytics/trackers/deep_sort_pytorch/.../ckpt.t7'
```

#### 4. **Shelf Detection Lines**

Sesuaikan koordinat garis deteksi rak:

```python
SHELF_LINE_1_2 = ((330, 938), (1631, 798))  # Line 1-2
SHELF_LINE_3_4 = ((481, 816), (1500, 730))  # Line 3-4
SHELF_LINE_5_6 = ((585, 715), (1379, 651))  # Line 5-6
SHELF_LINE_7_8 = ((665, 634), (1282, 585))  # Line 7-8
```

#### 5. **Virtual Line Position**

```python
LINE_TOP_CAMERA = [(100, 500), (1800, 500)]     # Untuk kamera atas
LINE_BOTTOM_CAMERA = [(100, 700), (1800, 700)]  # Untuk kamera bawah
```

#### 6. **UI Settings**

```python
UI_LEFT_MARGIN = 20
UI_BOX_WIDTH = 500
UI_BOX_COLOR = [85, 45, 255]
UI_TEXT_COLOR = [225, 255, 255]
```

## 🎯 Usage

### Basic Example

```python
from ultralytics import YOLO

# Load model
model = YOLO("ultralytics/checkpoint/best (1).pt")

# Run tracking
results = model.track(
    source="path/to/video.mp4",
    stream=True,
    save=True,
    show=True,
    device=0,
    persist=True,
    tracker="bytetrack.yaml"  # atau None jika USE_DEEPSORT=True
)

# Process results
for result in results:
    # Your code here
    pass
```

### Running Example Script

```bash
python run_tracking.py
```

## 📦 Required Model Files

### 1. YOLO Detection Model

Letakkan model YOLO di:
```
ultralytics/checkpoint/best (1).pt
```

### 2. Hand Landmark Model

Download MediaPipe Hand Landmarker:
```bash
# Download dari: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
# Simpan di: ultralytics/checkpoint/hand_landmarker.task
```

### 3. DeepSort ReID Model (Opsional)

Jika menggunakan `USE_DEEPSORT=True`:
```
ultralytics/trackers/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7
```

## 🛠️ Development

### Editing Code

Setelah `pip install -e .`, edit file di:

```
ultralytics/models/yolo/detect/
├── config.py    # Edit konfigurasi di sini
├── helper.py    # Edit utility functions di sini
└── predict.py   # Edit main logic di sini
```

Tidak perlu install ulang, perubahan langsung terdeteksi!

### Adding Custom Functions

1. Tambahkan ke `helper.py` untuk utility functions
2. Import di `predict.py`:
   ```python
   from helper import your_new_function
   ```

### Changing Detection Lines

Edit koordinat di `config.py`:

```python
SHELF_LINE_1_2 = ((x1, y1), (x2, y2))
```

Warna garis:
```python
LINE_COLOR_1_2 = (B, G, R)  # Format BGR OpenCV
```

## 🎨 Output

- **Video Results:** `runs/detect/trackX/`
- **Shelf Coordinates:** `shelf_coordinates.txt` (di working directory)
- **Labels:** `runs/detect/trackX/labels/`

## 🐛 Troubleshooting

### Error: Module not found

```bash
# Reinstall in editable mode
pip install -e .
```

### Error: Model file not found

Pastikan model ada di:
- `ultralytics/checkpoint/best (1).pt`
- `ultralytics/checkpoint/hand_landmarker.task`

### Error: DeepSort checkpoint not found

Jika `USE_DEEPSORT=True`, pastikan:
- `ultralytics/trackers/deep_sort_pytorch/.../ckpt.t7` ada
- Atau set `USE_DEEPSORT=False` untuk menggunakan ByteTrack

### Camera position salah

Edit di `config.py`:
```python
CAMERA_FROM_TOP = True   # atau False
```

## 📝 Configuration Checklist

Sebelum run, pastikan sudah set di `config.py`:

- [ ] `USE_DEEPSORT` = True/False
- [ ] `CAMERA_FROM_TOP` = True/False  
- [ ] Model files ada di `ultralytics/checkpoint/`
- [ ] Shelf lines koordinat sudah sesuai
- [ ] Virtual line position sudah sesuai
- [ ] UI settings (opsional)

## 🔄 Workflow

1. **Install:** `pip install -e .`
2. **Configure:** Edit `ultralytics/models/yolo/detect/config.py`
3. **Add Models:** Copy ke `ultralytics/checkpoint/`
4. **Run:** `python run_tracking.py`
5. **Edit Code:** Langsung edit tanpa reinstall
6. **Test:** Run lagi untuk lihat perubahan

## 📚 Documentation

- [Ultralytics Docs](https://docs.ultralytics.com/)
- [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [DeepSort](https://github.com/nwojke/deep_sort)

## 🤝 Contributing

1. Edit code di `ultralytics/models/yolo/detect/`
2. Test dengan `python run_tracking.py`
3. Commit changes

---

**Note:** Selalu gunakan `pip install -e .` agar perubahan code langsung terdeteksi tanpa reinstall!
