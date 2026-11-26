import argparse
import os
from datetime import datetime
import numpy as np
import cv2
try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required to load config.yml. Install it via: pip install pyyaml") from exc
from pypylon import pylon
from pylsl import resolve_byprop, StreamInlet, local_clock

DEFAULT_CONFIG = {
    "experiment": {
        "subject_id": "",
        "base_output_dir": "../fNIRS_Basler_recordings/recordings",
        "framework": "pytorch",
    },
    "camera": {
        "target_fps": 30.0,
        "resize_dim": [224, 224],
    },
    "fnirs": {
        "enabled": True,
        "stream_type": "NIRS",
        "stream_name": "",
        "expected_srate": 10.17,
        "zscore_normalize": True,
    },
    "eeg": {
        "enabled": False,
        "stream_type": "EEG",
        "stream_name": "",
        "expected_srate": 250.0,
        "zscore_normalize": True,
    },
    "windowing": {
        "window_seconds": 2.0,
        "overlap_ratio": 0.5,
    },
}


def load_config(config_path):
    """Loads parameters from a YAML file and merges them with defaults."""
    cfg = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        for section, values in user_cfg.items():
            if section in cfg and isinstance(values, dict):
                cfg[section].update(values)
            else:
                cfg[section] = values
        print(f"Loaded configuration from: {config_path}")
    else:
        print(f"Warning: Config file '{config_path}' not found. Using built-in defaults.")
    return cfg


def create_session_dir(base_dir, subject_id):
    """Creates a unique directory per subject and timestamp to prevent overwriting."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_id = subject_id.strip() if subject_id and subject_id.strip() else "unknown_subject"
    session_dir = os.path.join(base_dir, f"{clean_id}_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def connect_lsl_stream(stream_type, stream_name="", timeout=10.0, label="LSL"):
    """Resolves and connects to an LSL stream by name (if provided) or type."""
    if stream_name:
        print(f"Searching for {label} LSL stream (name='{stream_name}')...")
        streams = resolve_byprop("name", stream_name, timeout=timeout)
    else:
        print(f"Searching for {label} LSL stream (type='{stream_type}')...")
        streams = resolve_byprop("type", stream_type, timeout=timeout)

    if not streams:
        raise RuntimeError(
            f"No {label} stream found (type='{stream_type}', name='{stream_name}'). "
            f"Ensure the acquisition software is open and actively streaming."
        )

    inlet = StreamInlet(streams[0])
    info = inlet.info()
    print(
        f"Connected to {label} stream: name='{info.name()}', "
        f"channels={info.channel_count()}, nominal_srate={info.nominal_srate():.2f} Hz"
    )
    return inlet


def audit_sync_quality(frame_ts, fnirs_ts, eeg_ts=None, target_fps=30.0):
    """Audits camera frame rate stability, fNIRS update rate, optional EEG rate, and alignment lag."""
    print("\n--- Synchronization & Jitter Audit ---")
    if len(frame_ts) < 2:
        print("Not enough samples collected to compute timing statistics.")
        return target_fps

    frame_diffs = np.diff(frame_ts)
    mean_fps = 1.0 / np.mean(frame_diffs)
    jitter_ms = np.std(frame_diffs) * 1000.0
    max_gap_ms = np.max(frame_diffs) * 1000.0
    print(f"Camera FPS:        mean={mean_fps:.2f} Hz | jitter(std)={jitter_ms:.2f} ms | max_gap={max_gap_ms:.2f} ms")

    if fnirs_ts is not None and len(fnirs_ts) > 0:
        unique_fnirs_ts = np.unique(fnirs_ts)
        fnirs_diffs = np.diff(unique_fnirs_ts)
        mean_fnirs_hz = 1.0 / np.mean(fnirs_diffs) if len(fnirs_diffs) > 0 else 0.0
        sync_lag_ms = (frame_ts - fnirs_ts) * 1000.0
        print(f"fNIRS Stream Rate: mean={mean_fnirs_hz:.2f} Hz ({len(unique_fnirs_ts)} unique samples across {len(frame_ts)} frames)")
        print(f"fNIRS Sync Lag:    mean={np.mean(sync_lag_ms):.2f} ms | max={np.max(sync_lag_ms):.2f} ms")

    if eeg_ts is not None and len(eeg_ts) > 1:
        eeg_diffs = np.diff(eeg_ts)
        mean_eeg_hz = 1.0 / np.mean(eeg_diffs)
        print(f"EEG Stream Rate:   mean={mean_eeg_hz:.2f} Hz ({len(eeg_ts)} total high-rate samples collected)")

    return mean_fps


def zscore_channels(arr):
    """Applies per-channel Z-score normalization."""
    mean = np.mean(arr, axis=0, keepdims=True)
    std = np.std(arr, axis=0, keepdims=True) + 1e-8
    return (arr - mean) / std


def prepare_training_windows(
    video_arr,
    fnirs_arr,
    frame_ts_arr,
    fps,
    window_sec,
    overlap_ratio,
    framework,
    fnirs_zscore=True,
    eeg_arr=None,
    eeg_ts_arr=None,
    eeg_zscore=True,
    eeg_expected_srate=250.0,
):
    """Normalizes modalities, slices into time-aligned windows, and formats tensor dimensions."""
    fnirs_proc = zscore_channels(fnirs_arr) if fnirs_zscore else fnirs_arr.copy()
    video_rgb = video_arr[..., ::-1]  # BGR -> RGB in uint8 to save 75% RAM

    window_size = max(1, int(round(window_sec * fps)))
    step_size = max(1, int(round(window_size * (1.0 - overlap_ratio))))

    n_samples = len(video_rgb)
    if n_samples < window_size:
        print(f"\nWarning: Recorded {n_samples} frames, fewer than {window_size} frames ({window_sec}s). Skipping windowing.")
        return None, None, None

    has_eeg = eeg_arr is not None and eeg_ts_arr is not None and len(eeg_arr) > 0
    if has_eeg:
        eeg_proc = zscore_channels(eeg_arr) if eeg_zscore else eeg_arr.copy()
        eeg_diffs = np.diff(eeg_ts_arr)
        measured_eeg_hz = (1.0 / np.mean(eeg_diffs)) if len(eeg_diffs) > 0 else eeg_expected_srate
        eeg_target_len = max(1, int(round(window_sec * measured_eeg_hz)))

    video_windows = []
    fnirs_windows = []
    eeg_windows = []

    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        video_windows.append(video_rgb[start:end])
        fnirs_windows.append(fnirs_proc[start:end])

        if has_eeg:
            t_start = frame_ts_arr[start]
            idx_start = int(np.searchsorted(eeg_ts_arr, t_start, side="left"))
            idx_end = min(len(eeg_proc), idx_start + eeg_target_len)
            chunk = eeg_proc[idx_start:idx_end]
            if len(chunk) < eeg_target_len:
                pad_rows = eeg_target_len - len(chunk)
                pad_val = chunk[-1:] if len(chunk) > 0 else np.zeros((1, eeg_proc.shape[1]), dtype=np.float32)
                chunk = np.vstack([chunk, np.repeat(pad_val, pad_rows, axis=0)])
            eeg_windows.append(chunk)

    X_vid_win = np.array(video_windows, dtype=np.uint8)       # Default: (B, T, H, W, C)
    X_fnirs_win = np.array(fnirs_windows, dtype=np.float32)   # Default: (B, T, Channels)
    X_eeg_win = np.array(eeg_windows, dtype=np.float32) if has_eeg else None

    if framework.lower() == "pytorch":
        X_vid_win = np.transpose(X_vid_win, (0, 1, 4, 2, 3))    # (B, T, C, H, W)
        X_fnirs_win = np.transpose(X_fnirs_win, (0, 2, 1))      # (B, Channels, T)
        if X_eeg_win is not None:
            X_eeg_win = np.transpose(X_eeg_win, (0, 2, 1))      # (B, EEG_Channels, T_eeg)

    return X_vid_win, X_fnirs_win, X_eeg_win


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synchronized Basler + fNIRS (+ optional EEG) Recorder")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to YAML configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    exp_cfg = cfg["experiment"]
    cam_cfg = cfg["camera"]
    fnirs_cfg = cfg["fnirs"]
    eeg_cfg = cfg["eeg"]
    win_cfg = cfg["windowing"]

    framework = exp_cfg["framework"]
    base_output_dir = exp_cfg["base_output_dir"]
    target_fps = float(cam_cfg["target_fps"])
    resize_dim = tuple(cam_cfg["resize_dim"]) if cam_cfg["resize_dim"] else None
    window_seconds = float(win_cfg["window_seconds"])
    overlap_ratio = float(win_cfg["overlap_ratio"])

    subject_id = str(exp_cfg.get("subject_id", "")).strip()
    if not subject_id:
        subject_id = input("Enter Subject / Session ID (e.g., sub01_trial1): ")

    session_dir = create_session_dir(base_output_dir, subject_id)
    print(f"Session files will be saved to: {session_dir}")

    # Save a snapshot of the exact config used for this session
    with open(os.path.join(session_dir, "session_config.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # 1. Connect to fNIRS LSL stream from Aurora
    fnirs_inlet = connect_lsl_stream(
        stream_type=fnirs_cfg["stream_type"],
        stream_name=fnirs_cfg.get("stream_name", ""),
        label="fNIRS (Aurora)",
    )

    # 2. Optionally connect to EEG LSL stream (runs in separate EEG acquisition software)
    eeg_inlet = None
    if eeg_cfg.get("enabled", False):
        eeg_inlet = connect_lsl_stream(
            stream_type=eeg_cfg["stream_type"],
            stream_name=eeg_cfg.get("stream_name", ""),
            label="EEG",
        )

    # 3. Connect to Basler camera and cap acquisition frame rate
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    try:
        camera.AcquisitionFrameRateEnable.Value = True
        camera.AcquisitionFrameRate.Value = target_fps
        print(f"Camera hardware frame rate capped at {target_fps} FPS.")
    except Exception as e:
        print(f"Warning: Could not set hardware FPS cap ({e}). Running at camera default rate.")

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment.Value = pylon.OutputBitAlignment_MsbAligned

    active_modalities = "Video + fNIRS" + (" + EEG" if eeg_inlet is not None else "")
    input(f"Press ENTER to start synchronized recording ({active_modalities})...")
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    aligned_frames = []
    aligned_fnirs = []
    frame_timestamps = []
    fnirs_timestamps = []

    eeg_samples_all = []
    eeg_timestamps_all = []

    latest_fnirs_sample = None
    latest_fnirs_ts = None

    print("Recording and aligning in real time... Click the video window and press 'Q' or 'ESC' to stop.")
    try:
        while camera.IsGrabbing():
            # Pull most recent fNIRS sample (non-blocking)
            while True:
                sample, ts = fnirs_inlet.pull_sample(timeout=0.0)
                if sample is None:
                    break
                latest_fnirs_sample = sample
                latest_fnirs_ts = ts

            # Pull all available high-rate EEG samples in chunks (non-blocking)
            if eeg_inlet is not None:
                samples, timestamps = eeg_inlet.pull_chunk(timeout=0.0)
                if timestamps and latest_fnirs_sample is not None:
                    eeg_samples_all.extend(samples)
                    eeg_timestamps_all.extend(timestamps)

            # Grab frame from Basler camera
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded() and latest_fnirs_sample is not None:
                frame_ts = local_clock()
                img = converter.Convert(grabResult).GetArray()

                if resize_dim is not None:
                    img = cv2.resize(img, resize_dim, interpolation=cv2.INTER_AREA)

                aligned_frames.append(img)
                aligned_fnirs.append(latest_fnirs_sample)
                frame_timestamps.append(frame_ts)
                fnirs_timestamps.append(latest_fnirs_ts)

                cv2.imshow("Synced Recording (Press Q or ESC to stop)", img)
                key = cv2.waitKey(5) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 27 = ESC (works in any keyboard language)
                    break
            grabResult.Release()
    finally:
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()

        X_video = np.array(aligned_frames, dtype=np.uint8)
        X_fnirs = np.array(aligned_fnirs, dtype=np.float32)
        frame_ts_arr = np.array(frame_timestamps, dtype=np.float64)
        fnirs_ts_arr = np.array(fnirs_timestamps, dtype=np.float64)

        np.save(os.path.join(session_dir, "X_video_aligned.npy"), X_video)
        np.save(os.path.join(session_dir, "X_fnirs_aligned.npy"), X_fnirs)
        np.save(os.path.join(session_dir, "frame_timestamps.npy"), frame_ts_arr)
        np.save(os.path.join(session_dir, "fnirs_timestamps.npy"), fnirs_ts_arr)
        print(f"\nSaved {len(X_video)} raw aligned Video+fNIRS pairs to {session_dir}")

        X_eeg_raw = None
        eeg_ts_arr = None
        if eeg_inlet is not None and len(eeg_samples_all) > 0:
            X_eeg_raw = np.array(eeg_samples_all, dtype=np.float32)
            eeg_ts_arr = np.array(eeg_timestamps_all, dtype=np.float64)
            np.save(os.path.join(session_dir, "X_eeg_raw.npy"), X_eeg_raw)
            np.save(os.path.join(session_dir, "eeg_timestamps.npy"), eeg_ts_arr)
            print(f"Saved {len(X_eeg_raw)} raw high-rate EEG samples to {session_dir}")

        measured_fps = audit_sync_quality(frame_ts_arr, fnirs_ts_arr, eeg_ts_arr, target_fps=target_fps)

        X_vid_win, X_fnirs_win, X_eeg_win = prepare_training_windows(
            video_arr=X_video,
            fnirs_arr=X_fnirs,
            frame_ts_arr=frame_ts_arr,
            fps=measured_fps,
            window_sec=window_seconds,
            overlap_ratio=overlap_ratio,
            framework=framework,
            fnirs_zscore=fnirs_cfg.get("zscore_normalize", True),
            eeg_arr=X_eeg_raw,
            eeg_ts_arr=eeg_ts_arr,
            eeg_zscore=eeg_cfg.get("zscore_normalize", True),
            eeg_expected_srate=float(eeg_cfg.get("expected_srate", 250.0)),
        )

        if X_vid_win is not None:
            np.save(os.path.join(session_dir, "X_video_windows.npy"), X_vid_win)
            np.save(os.path.join(session_dir, "X_fnirs_windows.npy"), X_fnirs_win)
            print(f"\nSaved windowed tensors ({framework.upper()} layout):")
            print(f"X_video_windows.npy shape: {X_vid_win.shape} (dtype={X_vid_win.dtype})")
            print(f"X_fnirs_windows.npy shape: {X_fnirs_win.shape} (dtype={X_fnirs_win.dtype})")
            if X_eeg_win is not None:
                np.save(os.path.join(session_dir, "X_eeg_windows.npy"), X_eeg_win)
                print(f"X_eeg_windows.npy   shape: {X_eeg_win.shape} (dtype={X_eeg_win.dtype})")
