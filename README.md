# Basler Camera, NIRx Aurora (fNIRS) & EEG Real-Time LSL Synchronization

A real-time multimodal data acquisition and alignment pipeline for **Basler industrial cameras** (via `pypylon`), **NIRx NIRSport 2 fNIRS systems** (via **Aurora** and **Lab Streaming Layer — LSL**), and optional **EEG amplifiers** over LSL, configured via a central **`config.yml`** file.

---

## Problem Statement

Training multimodal deep learning models (e.g., neural networks combining facial video with prefrontal hemodynamic activity and electrophysiology) requires strictly aligned time-series data from independent hardware devices:

1. **Asynchronous Clocks & Start Times:** Aurora (fNIRS), EEG amplifiers, and the Basler camera run on separate hardware clocks and are typically started at slightly different moments.
2. **Mismatched Sampling Rates:** A Basler camera captures frames at 30 Hz (up to >130 Hz uncapped), NIRSport 2 samples brain hemodynamics at ~10.17 Hz, and EEG systems typically sample at 250–1,000 Hz.
3. **Memory & Configuration Bottlenecks:** Uncapped high-FPS video converted prematurely to `float32` quickly exhausts system RAM during windowing, while hardcoded experiment parameters make multi-subject sessions error-prone.

### How This Pipeline Solves It
* **YAML-Driven Experiment Configuration (`config.yml`):** Centralizes subject IDs, output directories, target sampling rates, LSL stream filters, and windowing hyperparameters—and saves a snapshot (`session_config.yml`) inside every recording folder for full reproducibility.
* **Zero-Config LSL Stream Discovery:** Automatically detects the live `type='NIRS'` stream broadcast by Aurora (and optional `type='EEG'` stream) over LSL.
* **Real-Time Multi-Rate Alignment:** Pairs every grabbed camera frame (`pypylon`) 1-to-1 with the latest incoming fNIRS sample, pulls high-rate EEG data in non-blocking chunks (`pull_chunk`), and logs synchronized `lsl.local_clock()` timestamps across all modalities.
* **Hardware Rate Capping & Memory Efficiency:** Caps the Basler camera at `30.0 FPS`, resizes frames on the fly (`224x224`), and stores video tensors in `uint8` (saving 75% RAM compared to `float32`).
* **Automated Auditing & Windowing:** Outputs a synchronization/jitter quality report and slices aligned data into Z-score normalized sliding windows formatted directly for **PyTorch** or **TensorFlow**.

---

## Prerequisites & Installation

### Hardware & Software
* **NIRx NIRSport 2** connected to PC with **NIRx Aurora** (v2025.2+ or any LSL-enabled version) installed.
* **Basler Camera** connected via USB3/GigE with **Basler pylon Camera Software Suite** installed.
* *(Optional)* **EEG Amplifier & Acquisition Software** with LSL streaming enabled (e.g., BrainProducts LiveAmp, g.tec, OpenBCI).
* **Python 3.9+**

### Python Dependencies
Install the required packages in your virtual environment:

```bash
pip install numpy opencv-python pypylon pylsl pyyaml
```

---

## Step-by-Step User Guide

### Step 1: Prepare the Basler Camera
1. Connect the Basler camera to your computer.
2. **Close Basler Pylon Viewer** if it is open (only one application can access the camera stream at a time).

### Step 2: Configure & Start NIRx Aurora (and Optional EEG)
1. Power on the **NIRSport 2** device and launch **NIRx Aurora**.
2. Select your montage/configuration (e.g., `Standard_Prefrontal_8x8`) and connect to the device.
   * *Note:* Aurora broadcasts LSL automatically under `Data out stream name: Aurora` (`type='NIRS'`). No manual LSL checkbox is needed in modern Aurora versions.
3. Place the cap on the participant and click **Optimize** (Signal Optimization) until channels turn **green**.
4. Click **Start** (or **Preview** / **Record** if you also want a `.snirf` backup) so live fNIRS signals are actively streaming on the Aurora screen.
5. *(Optional)* If recording EEG, start your EEG acquisition software and enable its LSL output (`type='EEG'`).

### Step 3: Edit `config.yml`
Open `config.yml` and set your session parameters before running the script:
* Set `experiment.subject_id` (e.g., `"sub01_trial1"`). If left blank (`""`), the terminal will prompt you interactively at runtime.
* Verify `experiment.base_output_dir` (e.g., `"../fNIRS_Basler_recordings/recordings"`).
* Adjust `camera.target_fps`, `windowing.window_seconds`, and `framework` (`"pytorch"` or `"tensorflow"`) as needed.
* Set `eeg.enabled: true` if acquiring EEG alongside fNIRS and video (otherwise keep `false`).

### Step 4: Run the Synchronized Recording Script
1. Open a terminal in the project directory and run:
   ```bash
   python basler_aurora_synched_recordings.py
   ```
   *(You can also pass a custom YAML file via `python basler_aurora_synched_recordings.py --config path/to/custom_config.yml`.)*
2. **Verify LSL Connection:** Confirm the console prints:
   ```text
   Successfully connected to fNIRS stream!
   Camera hardware frame rate capped at 30.0 FPS.
   ```
3. **Start Recording:** Press **ENTER** in the terminal to open the live camera preview and begin synchronized capture.

### Step 5: Stop Recording Safely
1. Click inside the **`Synced Recording`** OpenCV video window so it has keyboard focus.
2. Press **`Q`** or **`ESC`** (works in both English and Hebrew keyboard layouts).
   * *Avoid closing the window with the red `X` button or `Ctrl+C`, as pressing `Q`/`ESC` ensures clean camera shutdown and tensor generation.*

---

## Output Files & Directory Structure

Each session saves all configuration, raw, and windowed files into `<base_output_dir>/<subject_id>_<YYYYMMDD_HHMMSS>/`:

| File Name | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `session_config.yml` | — | YAML | Exact copy of the configuration parameters used for the session |
| `X_video_aligned.npy` | `(N, H, W, 3)` | `uint8` | Raw BGR frames aligned 1-to-1 with fNIRS samples |
| `X_fnirs_aligned.npy` | `(N, Channels)` | `float32` | Raw fNIRS sample paired with each video frame |
| `frame_timestamps.npy` | `(N,)` | `float64` | LSL local clock timestamp for each video frame |
| `fnirs_timestamps.npy` | `(N,)` | `float64` | LSL timestamp of the paired fNIRS sample |
| `X_video_windows.npy` | `(B, T, C, H, W)`* | `uint8` | RGB sliding windows ready for model training |
| `X_fnirs_windows.npy` | `(B, Channels, T)`* | `float32` | Per-channel Z-score normalized fNIRS windows |
| `X_eeg_raw.npy` *(if enabled)* | `(M, EEG_Ch)` | `float32` | Continuous high-rate EEG samples pulled via LSL chunks |
| `eeg_timestamps.npy` *(if enabled)* | `(M,)` | `float64` | LSL timestamps for each EEG sample |
| `X_eeg_windows.npy` *(if enabled)* | `(B, EEG_Ch, T_eeg)`* | `float32` | Z-score normalized EEG windows aligned to each video/fNIRS window |

*\*Shapes shown are for `framework: "pytorch"`. Setting `framework: "tensorflow"` outputs `(B, T, H, W, C)`, `(B, T, Channels)`, and `(B, T_eeg, EEG_Ch)`.*

---

## Configuration Reference (`config.yml`)

```yaml
experiment:
  subject_id: "sub01_trial1"                         # Leave empty ("") to be prompted in terminal
  base_output_dir: "../fNIRS_Basler_recordings/recordings"
  framework: "pytorch"                               # "pytorch" or "tensorflow"

camera:
  target_fps: 30.0                                   # Hardware camera frame rate cap (Hz)
  resize_dim: [224, 224]                             # [width, height]; set null for native resolution

fnirs:
  lsl_type: "NIRS"                                   # LSL stream type broadcast by Aurora
  lsl_name: ""                                       # Optional exact stream name (e.g., "Aurora")
  expected_srate: 10.17                              # Reference NIRSport 2 sampling rate (Hz)
  connection_timeout: 10.0                           # Seconds to wait for LSL stream discovery

eeg:
  enabled: false                                     # Set to true when an EEG LSL stream is active
  lsl_type: "EEG"                                    # LSL stream type broadcast by EEG software
  lsl_name: ""                                       # Optional exact EEG stream name
  expected_srate: 500.0                              # Reference EEG sampling rate (Hz)
  connection_timeout: 10.0                           # Seconds to wait for EEG LSL stream discovery

windowing:
  window_seconds: 2.0                                # Duration of each training window in seconds
  overlap_ratio: 0.5                                 # 0.5 = 50% overlap between consecutive windows
```

---

## Troubleshooting

* **`ModuleNotFoundError: No module named 'yaml'`:** Install PyYAML using `pip install pyyaml`.
* **`RuntimeError: No NIRS stream found`:** Ensure Aurora is open, connected to the NIRSport 2, and actively running a measurement/preview (`Start` or `Record` clicked).
* **EEG stream not detected when `eeg.enabled: true`:** Verify that your EEG acquisition software is actively streaming over LSL with `type='EEG'` (or specify `eeg.lsl_name` in `config.yml`).
* **Camera fails to open (`pypylon` error):** Close Pylon Viewer or any other software accessing the camera, then reconnect the USB cable.
* **`Q` key does not stop recording:** Click once inside the OpenCV video window to focus it, or press **`ESC`**.
* **Training Normalization Reminder:** `X_video_windows.npy` is saved as `uint8` (`0–255`) to conserve memory. Divide by `255.0` (and apply ImageNet mean/std if needed) inside your PyTorch `Dataset` or TensorFlow `tf.data` pipeline.
