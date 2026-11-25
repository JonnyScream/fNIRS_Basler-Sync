# Basler Camera & NIRx Aurora (fNIRS) Real-Time LSL Synchronization

A real-time multimodal data acquisition and alignment pipeline for **Basler industrial cameras** (via `pypylon`) and **NIRx NIRSport 2 fNIRS systems** (via **Aurora** and **Lab Streaming Layer — LSL**).

---

## Problem Statement

Training multimodal deep learning models (e.g., dual-stream neural networks combining facial video with prefrontal hemodynamic activity) requires strictly aligned time-series data from two independent hardware devices:

1. **Asynchronous Clocks & Start Times:** Aurora (fNIRS) and the Basler camera run on separate hardware clocks and are typically started at slightly different moments.
2. **Mismatched Sampling Rates:** A Basler camera captures frames at high rates (30 Hz up to >130 Hz), whereas NIRSport 2 samples brain hemodynamics at ~10.17 Hz.
3. **Memory & Preprocessing Bottlenecks:** Uncapped high-FPS video converted prematurely to `float32` quickly exhausts system RAM during windowing, and manual post-hoc trimming introduces human error.

### How This Pipeline Solves It
* **Zero-Config LSL Stream Discovery:** Automatically detects the live `type='NIRS'` stream broadcast by Aurora over LSL without requiring external recording software.
* **Real-Time 1-to-1 Alignment:** Pairs every grabbed camera frame (`pypylon`) with the latest incoming fNIRS sample and logs synchronized `lsl.local_clock()` timestamps for both modalities.
* **Hardware Rate Capping & Memory Efficiency:** Caps the Basler camera at `30.0 FPS`, resizes frames on the fly (`224x224`), and stores video tensors in `uint8` (saving 75% RAM compared to `float32`).
* **Automated Auditing & Windowing:** Outputs a synchronization/jitter quality report and slices the aligned data into Z-score normalized sliding windows formatted directly for **PyTorch** `(B, T, C, H, W)` or **TensorFlow** `(B, T, H, W, C)`.

---

## Prerequisites & Installation

### Hardware & Software
* **NIRx NIRSport 2** connected to PC with **NIRx Aurora** (v2025.2+ or any LSL-enabled version) installed.
* **Basler Camera** connected via USB3/GigE with **Basler pylon Camera Software Suite** installed.
* **Python 3.9+**

### Python Dependencies
Install the required packages in your virtual environment:

```bash
pip install numpy opencv-python pypylon pylsl
```

---

## Step-by-Step User Guide

### Step 1: Prepare the Basler Camera
1. Connect the Basler camera to your computer.
2. **Close Basler Pylon Viewer** if it is open (only one application can access the camera stream at a time).

### Step 2: Configure & Start NIRx Aurora
1. Power on the **NIRSport 2** device and launch **NIRx Aurora**.
2. Select your montage/configuration (e.g., `Standard_Prefrontal_8x8`) and connect to the device.
   * *Note:* Aurora broadcasts LSL automatically under `Data out stream name: Aurora` (`type='NIRS'`). No manual LSL checkbox is needed in modern Aurora versions.
3. Place the cap on the participant and click **Optimize** (Signal Optimization) until channels turn **green**.
4. Click **Start** (or **Preview** / **Record** if you also want a `.snirf` backup) so live fNIRS signals are actively streaming on the Aurora screen.

### Step 3: Run the Synchronized Recording Script
1. Open a terminal in the project directory and run:
   ```bash
   python basler_aurora_synched_recordings.py
   ```
2. **Enter Session ID:** When prompted (`Enter Subject / Session ID:`), type a unique identifier (e.g., `sub01_trial1`). A timestamped folder will be created automatically under `recordings/`.
3. **Verify LSL Connection:** Confirm the console prints:
   ```text
   Successfully connected to fNIRS stream!
   Camera hardware frame rate capped at 30.0 FPS.
   ```
4. **Start Recording:** Press **ENTER** in the terminal to open the live camera preview and begin synchronized capture.

### Step 4: Stop Recording Safely
1. Click inside the **`Synced Recording`** OpenCV video window so it has keyboard focus.
2. Press **`Q`** or **`ESC`** (works in both English and Hebrew keyboard layouts).
   * *Avoid closing the window with the red `X` button or `Ctrl+C`, as pressing `Q`/`ESC` ensures clean camera shutdown and tensor generation.*

---

## Output Files & Directory Structure

Each session saves all raw and processed files into `recordings/<subject_id>_<YYYYMMDD_HHMMSS>/`:

| File Name | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `X_video_aligned.npy` | `(N, H, W, 3)` | `uint8` | Raw BGR frames aligned 1-to-1 with fNIRS samples |
| `X_fnirs_aligned.npy` | `(N, Channels)` | `float32` | Raw fNIRS sample paired with each video frame |
| `frame_timestamps.npy` | `(N,)` | `float64` | LSL local clock timestamp for each video frame |
| `fnirs_timestamps.npy` | `(N,)` | `float64` | LSL timestamp of the paired fNIRS sample |
| `X_video_windows.npy` | `(B, T, C, H, W)`* | `uint8` | RGB sliding windows ready for model training |
| `X_fnirs_windows.npy` | `(B, Channels, T)`* | `float32` | Per-channel Z-score normalized fNIRS windows |

*\*Shapes shown are for `FRAMEWORK = "pytorch"`. Setting `FRAMEWORK = "tensorflow"` outputs `(B, T, H, W, C)` and `(B, T, Channels)`.*

---

## Configuration Parameters

Edit the top of `basler_aurora_synched_recordings.py` to match your neural network requirements:

```python
FRAMEWORK = "pytorch"       # "pytorch" or "tensorflow"
TARGET_FPS = 30.0           # Camera hardware FPS limit
RESIZE_DIM = (224, 224)     # Frame resolution (width, height); set None for native resolution
WINDOW_SECONDS = 2.0        # Duration of each training window in seconds
OVERLAP_RATIO = 0.5         # Sliding window overlap (0.5 = 50% overlap)
BASE_OUTPUT_DIR = "recordings"
```

---

## Troubleshooting

* **`RuntimeError: No NIRS stream found`:** Ensure Aurora is open, connected to the NIRSport 2, and actively running a measurement/preview (`Start` or `Record` clicked).
* **Camera fails to open (`pypylon` error):** Close Pylon Viewer or any other software accessing the camera, then reconnect the USB cable.
* **`Q` key does not stop recording:** Click once inside the OpenCV video window to focus it, or press **`ESC`**.
* **Training Normalization Reminder:** `X_video_windows.npy` is saved as `uint8` (`0–255`) to conserve memory. Divide by `255.0` (and apply ImageNet mean/std if needed) inside your PyTorch `Dataset` or TensorFlow `tf.data` pipeline.
