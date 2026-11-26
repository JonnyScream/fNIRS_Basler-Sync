# Basler Camera, NIRx Aurora (fNIRS) & Unicorn EEG Real-Time LSL Synchronization

A real-time multimodal data acquisition, alignment, and quality-control visualization pipeline for **Basler industrial cameras** (via `pypylon`), **NIRx NIRSport 2 fNIRS systems** (via **Aurora** and **Lab Streaming Layer — LSL**), and **g.tec Unicorn Hybrid Black EEG systems** over LSL, configured via a central **`config.yml`** file.

---

## Problem Statement

Training multimodal deep learning models (e.g., neural networks combining facial video with prefrontal hemodynamic activity and electrophysiology) requires strictly aligned time-series data from independent hardware devices:

1. **Asynchronous Clocks & Start Times:** Aurora (fNIRS), Unicorn Recorder (EEG), and the Basler camera run on separate hardware clocks and are typically started at slightly different moments.
2. **Mismatched Sampling Rates:** A Basler camera captures frames at 30 Hz (up to >130 Hz uncapped), NIRSport 2 samples brain hemodynamics at ~10.17 Hz, and Unicorn EEG samples at 250 Hz.
3. **Channel & Memory Bottlenecks:** Uncapped high-FPS video converted prematurely to `float32` quickly exhausts system RAM during windowing, and Unicorn LSL streams include 9 auxiliary channels (accelerometer, gyroscope, battery, counters) alongside the 8 brain channels.

### How This Pipeline Solves It
* **YAML-Driven Experiment Configuration (`config.yml`):** Centralizes subject IDs, output directories, target sampling rates, LSL stream filters, EEG channel slicing, and windowing hyperparameters—and saves a snapshot (`session_config.yml`) inside every recording folder for full reproducibility.
* **Zero-Config LSL Stream Discovery:** Automatically detects the live `type='NIRS'` stream broadcast by Aurora and the `type='EEG'` stream broadcast by Unicorn Recorder over LSL.
* **Real-Time Multi-Rate Alignment:** Pairs every grabbed camera frame (`pypylon`) 1-to-1 with the latest incoming fNIRS sample, pulls high-rate EEG data in non-blocking chunks (`pull_chunk`), extracts only the 8 EEG brain channels (`num_eeg_channels: 8`), and logs synchronized `lsl.local_clock()` timestamps across all modalities.
* **Hardware Rate Capping & Memory Efficiency:** Caps the Basler camera at `30.0 FPS`, resizes frames on the fly (`224x224`), and stores video tensors in `uint8` (saving 75% RAM compared to `float32`).
* **Automated Auditing, Windowing & Visualization:** Outputs a synchronization/jitter quality report, slices aligned data into Z-score normalized sliding windows formatted for **PyTorch** or **TensorFlow**, and generates diagnostic PNG plots via `visualize_session.py`.

---

## Prerequisites & Installation

### Hardware & Software
* **NIRx NIRSport 2** connected to PC with **NIRx Aurora** (v2025.2+ or any LSL-enabled version) installed.
* **Basler Camera** connected via USB3/GigE with **Basler pylon Camera Software Suite** installed.
* *(Optional)* **g.tec Unicorn Hybrid Black** paired via Bluetooth with **Unicorn Suite / Unicorn Recorder** installed.
* **Python 3.9+**

### Python Dependencies
Install the required packages in your virtual environment:

```bash
pip install numpy opencv-python pypylon pylsl pyyaml matplotlib
```

---

## Step-by-Step User Guide

### Step 1: Prepare & Focus the Basler Camera
1. Connect the Basler camera to your computer.
2. *(Recommended)* Open **Basler Pylon Viewer** briefly to verify lens focus and exposure/aperture on the participant's face.
3. **Close Basler Pylon Viewer completely** before running the Python script (only one application can access the camera stream at a time).

### Step 2: Configure & Start NIRx Aurora (fNIRS)
1. Power on the **NIRSport 2** device and launch **NIRx Aurora**.
2. Select your montage/configuration (e.g., `Standard_Prefrontal_8x8`) and connect to the device.
   * *Note:* Aurora broadcasts LSL automatically under `Data out stream name: Aurora` (`type='NIRS'`). No manual LSL checkbox is needed in modern Aurora versions.
3. Place the cap on the participant and click **Optimize** (Signal Optimization) until channels turn **green**.
4. Click **Start** (or **Preview** / **Record** if you also want a `.snirf` backup) so live fNIRS signals are actively streaming on the Aurora screen.

### Step 3: Configure & Start Unicorn Recorder (EEG)
1. Power on the **Unicorn** unit and connect it via Bluetooth.
2. Open **Unicorn Recorder**:
   * In the **`Device`** tab, leave all eight `Digital Out 0–7` switches set to **`OFF`** (these are hardware TTL trigger outputs, not EEG channels).
   * In the **`Network`** tab, enable **`Processed Data LSL Output`** (and ensure the **50 Hz Notch filter** is enabled in signal processing settings to prevent power-line hum).
3. Click the **Play** button (top-left triangle) or open the **`Acquisition`** tab and verify that all 8 EEG channels (`Fz, C3, Cz, C4, Pz, PO7, Oz, PO8`) are streaming cleanly.

### Step 4: Review `config.yml` & Run the Recording Script
1. Verify `config.yml` settings (`subject_id`, `base_output_dir`, `eeg.enabled: true`, `num_eeg_channels: 8`).
2. Open a terminal in the `Code/` directory and run:
   ```bash
   python basler_aurora_synched_recordings.py
   ```
   *(Or pass a custom YAML file via `python basler_aurora_synched_recordings.py --config path/to/custom_config.yml`.)*
3. **Verify Connections:** Confirm the console connects to fNIRS, Unicorn EEG (`250.00 Hz`), and caps the camera at `30.0 FPS`.
4. **Start Recording:** Press **ENTER** in the terminal to open the live camera preview and begin synchronized capture.

### Step 5: Stop Recording Safely
1. Click inside the **`Synced Recording`** OpenCV video window so it has keyboard focus.
2. Press **`ESC`** or **`Q`** (works in both English and Hebrew keyboard layouts).
   * *Avoid closing the window with the red `X` button or `Ctrl+C`, as pressing `ESC`/`Q` ensures clean camera shutdown and tensor generation.*

### Step 6: Visualize & Audit the Session (`visualize_session.py`)
Run the visualization script to generate diagnostic plots inside `<session_dir>/outputs/`:
```bash
python visualize_session.py
```
*(Or specify an older recording folder via `python visualize_session.py --session_dir <path_to_session>`.)*

This saves four PNG reports inside `<session_dir>/outputs/`:
* `01_sync_and_jitter.png` — Camera FPS stability and fNIRS sample age/lag.
* `02_fnirs_signals.png` — Z-score normalized fNIRS channels over time.
* `03_eeg_signals_and_psd.png` — All 8 Unicorn EEG channels and Power Spectral Density (PSD) to verify Alpha band (`8–12 Hz`) and `50 Hz` line-noise filtering.
* `04_multimodal_dashboard.png` — Unified timeline aligning 5 sampled video frames with fNIRS and EEG activity.

---

## Output Files & Directory Structure

Each session saves all configuration, raw, windowed, and visual output files into `<base_output_dir>/<subject_id>_<YYYYMMDD_HHMMSS>/`:

| File / Folder | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `session_config.yml` | — | YAML | Exact copy of the `config.yml` parameters used for the session |
| `X_video_aligned.npy` | `(N, H, W, 3)` | `uint8` | Raw BGR frames aligned 1-to-1 with fNIRS samples |
| `X_fnirs_aligned.npy` | `(N, Channels)` | `float32` | Raw fNIRS sample paired with each video frame |
| `frame_timestamps.npy` | `(N,)` | `float64` | LSL local clock timestamp for each video frame |
| `fnirs_timestamps.npy` | `(N,)` | `float64` | LSL timestamp of the paired fNIRS sample |
| `X_video_windows.npy` | `(B, T, C, H, W)`* | `uint8` | RGB sliding windows ready for model training |
| `X_fnirs_windows.npy` | `(B, Channels, T)`* | `float32` | Per-channel Z-score normalized fNIRS windows |
| `X_eeg_raw.npy` *(if enabled)* | `(M, 8)` | `float32` | Continuous 250 Hz EEG brain samples pulled via LSL chunks |
| `eeg_timestamps.npy` *(if enabled)* | `(M,)` | `float64` | LSL timestamps for each EEG sample |
| `X_eeg_windows.npy` *(if enabled)* | `(B, 8, T_eeg)`* | `float32` | Z-score normalized EEG windows aligned to each video/fNIRS window |
| `outputs/*.png` | — | PNG | Diagnostic synchronization, fNIRS, EEG PSD, and dashboard plots |

*\*Shapes shown are for `framework: "pytorch"`. Setting `framework: "tensorflow"` outputs `(B, T, H, W, C)`, `(B, T, Channels)`, and `(B, T_eeg, 8)`.*

---

## Configuration Reference (`config.yml`)

```yaml
experiment:
  subject_id: ""                                        # Leave empty ("") to prompt in terminal at runtime
  base_output_dir: "../../fNIRS_Basler_recordings/recordings"
  framework: "pytorch"                                  # "pytorch" -> (B, T, C, H, W) | "tensorflow" -> (B, T, H, W, C)

camera:
  target_fps: 30.0                                      # Hardware camera frame rate cap (Hz)
  resize_dim: [224, 224]                                # [width, height] for reduced RAM usage (null for native)

fnirs:
  enabled: true
  stream_type: "NIRS"                                   # LSL stream type broadcast by NIRx Aurora
  stream_name: ""                                       # Optional exact stream name (e.g., "Aurora")
  expected_srate: 10.17                                 # Nominal NIRSport 2 sampling rate in Hz
  zscore_normalize: true                                # Apply per-channel Z-score normalization

eeg:
  enabled: true                                         # Set to true when Unicorn LSL stream is active
  stream_type: "EEG"                                    # LSL stream type broadcast by Unicorn Recorder
  stream_name: ""                                       # Optional exact stream name; leave empty to match by type
  expected_srate: 250.0                                 # g.tec Unicorn Hybrid Black sampling rate (Hz)
  num_eeg_channels: 8                                   # Keep first 8 EEG brain channels (excludes aux/battery)
  zscore_normalize: true                                # Apply per-channel Z-score normalization

windowing:
  window_seconds: 2.0                                   # Duration of each training window in seconds
  overlap_ratio: 0.5                                    # 0.5 = 50% overlap between consecutive windows
```

---

## Troubleshooting

* **`ModuleNotFoundError: No module named 'yaml'` or `'matplotlib'`:** Install dependencies via `pip install pyyaml matplotlib`.
* **`RuntimeError: No NIRS stream found`:** Ensure Aurora is open, connected to the NIRSport 2, and actively running a measurement/preview (`Start` or `Record` clicked).
* **EEG stream not detected when `eeg.enabled: true`:** Verify that **Unicorn Recorder** has `Processed Data LSL Output` enabled under `Network` and acquisition is running (`Play` clicked).
* **Strong 50 Hz peak in EEG PSD plot:** Enable the **50 Hz Notch filter** in Unicorn Recorder (or apply a 50 Hz notch + 1–40 Hz bandpass filter in preprocessing) before training.
* **Camera fails to open (`pypylon` error):** Close Pylon Viewer or any other software accessing the camera, then reconnect the USB cable.
* **`Q` key does not stop recording:** Click once inside the OpenCV video window to focus it, or press **`ESC`**.
* **Training Normalization Reminder:** `X_video_windows.npy` is saved as `uint8` (`0–255`) to conserve memory. Divide by `255.0` inside your PyTorch `Dataset` or TensorFlow `tf.data` pipeline.
