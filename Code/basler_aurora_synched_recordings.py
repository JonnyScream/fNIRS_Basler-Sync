import os
from datetime import datetime
import numpy as np
import cv2
from pypylon import pylon
from pylsl import resolve_byprop, StreamInlet, local_clock

# --- Configuration ---
FRAMEWORK = "pytorch"       # Options: "pytorch" -> (B, T, C, H, W) | "tensorflow" -> (B, T, H, W, C)
RESIZE_DIM = (224, 224)     # (width, height) for reduced RAM usage
WINDOW_SECONDS = 2.0        # Window duration in seconds
OVERLAP_RATIO = 0.5         # 0.5 = 50% overlap between consecutive windows
BASE_OUTPUT_DIR = "recordings"


def create_session_dir(base_dir, subject_id):
    """Creates a unique directory per subject and timestamp to prevent overwriting."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_id = subject_id.strip() if subject_id.strip() else "unknown_subject"
    session_dir = os.path.join(base_dir, f"{clean_id}_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def audit_sync_quality(frame_ts, fnirs_ts):
    """Audits camera frame rate stability, fNIRS update rate, and alignment lag."""
    print("\n--- Synchronization & Jitter Audit ---")
    if len(frame_ts) < 2:
        print("Not enough samples collected to compute timing statistics.")
        return 30.0

    frame_diffs = np.diff(frame_ts)
    mean_fps = 1.0 / np.mean(frame_diffs)
    jitter_ms = np.std(frame_diffs) * 1000.0
    max_gap_ms = np.max(frame_diffs) * 1000.0

    unique_fnirs_ts = np.unique(fnirs_ts)
    fnirs_diffs = np.diff(unique_fnirs_ts)
    mean_fnirs_hz = 1.0 / np.mean(fnirs_diffs) if len(fnirs_diffs) > 0 else 0.0

    sync_lag_ms = (frame_ts - fnirs_ts) * 1000.0

    print(f"Camera FPS:        mean={mean_fps:.2f} Hz | jitter(std)={jitter_ms:.2f} ms | max_gap={max_gap_ms:.2f} ms")
    print(f"fNIRS Stream Rate: mean={mean_fnirs_hz:.2f} Hz ({len(unique_fnirs_ts)} unique samples across {len(frame_ts)} frames)")
    print(f"Sync Lag (Age):    mean={np.mean(sync_lag_ms):.2f} ms | max={np.max(sync_lag_ms):.2f} ms")
    return mean_fps


def prepare_training_windows(video_arr, fnirs_arr, fps, window_sec, overlap_ratio, framework):
    """Normalizes data, slices into time-based windows, and formats tensor dimensions."""
    # 1. Per-channel Z-score normalization for fNIRS
    mean = np.mean(fnirs_arr, axis=0, keepdims=True)
    std = np.std(fnirs_arr, axis=0, keepdims=True) + 1e-8
    fnirs_norm = (fnirs_arr - mean) / std

    # 2. Scale video pixels to [0, 1] float32 and convert BGR -> RGB
    video_rgb = video_arr[..., ::-1].astype(np.float32) / 255.0

    # 3. Compute window and step sizes in frames from measured FPS
    window_size = max(1, int(round(window_sec * fps)))
    step_size = max(1, int(round(window_size * (1.0 - overlap_ratio))))

    n_samples = len(video_rgb)
    if n_samples < window_size:
        print(f"\nWarning: Recorded {n_samples} frames, fewer than {window_size} frames ({window_sec}s). Skipping windowing.")
        return None, None

    video_windows = []
    fnirs_windows = []
    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        video_windows.append(video_rgb[start:end])
        fnirs_windows.append(fnirs_norm[start:end])

    X_vid_win = np.array(video_windows)    # Default: (B, T, H, W, C)
    X_fnirs_win = np.array(fnirs_windows)  # Default: (B, T, Channels)

    # 4. Adjust layout for target deep learning framework
    if framework.lower() == "pytorch":
        # Transpose video to (B, T, C, H, W) and fNIRS to (B, Channels, T)
        X_vid_win = np.transpose(X_vid_win, (0, 1, 4, 2, 3))
        X_fnirs_win = np.transpose(X_fnirs_win, (0, 2, 1))

    return X_vid_win, X_fnirs_win


# --- Main Execution ---
subject_id = input("Enter Subject / Session ID (e.g., sub01_trial1): ")
session_dir = create_session_dir(BASE_OUTPUT_DIR, subject_id)
print(f"Session files will be saved to: {session_dir}")

# 1. Connect to fNIRS LSL stream from Aurora
print("Searching for fNIRS LSL stream from Aurora...")
streams = resolve_byprop('type', 'NIRS', timeout=10)
if not streams:
    raise RuntimeError("No NIRS stream found. Ensure Aurora is open and actively streaming.")

fnirs_inlet = StreamInlet(streams)
print("Successfully connected to fNIRS stream!")

# 2. Connect to Basler camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

input("Press ENTER to start synchronized recording (Video + fNIRS)...")
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

aligned_frames = []
aligned_fnirs = []
frame_timestamps = []
fnirs_timestamps = []

latest_fnirs_sample = None
latest_fnirs_ts = None

print("Recording and aligning in real time... Press 'q' in the video window to stop.")
try:
    while camera.IsGrabbing():
        # Pull most recent fNIRS sample (non-blocking)
        while True:
            sample, ts = fnirs_inlet.pull_sample(timeout=0.0)
            if sample is None:
                break
            latest_fnirs_sample = sample
            latest_fnirs_ts = ts

        # Grab frame from Basler camera
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded() and latest_fnirs_sample is not None:
            frame_ts = local_clock()
            img = converter.Convert(grabResult).GetArray()

            if RESIZE_DIM is not None:
                img = cv2.resize(img, RESIZE_DIM, interpolation=cv2.INTER_AREA)

            aligned_frames.append(img)
            aligned_fnirs.append(latest_fnirs_sample)
            frame_timestamps.append(frame_ts)
            fnirs_timestamps.append(latest_fnirs_ts)

            cv2.imshow('Synced Recording (Press Q to stop)', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        grabResult.Release()
finally:
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()

    X_video = np.array(aligned_frames)
    X_fnirs = np.array(aligned_fnirs)
    frame_ts_arr = np.array(frame_timestamps)
    fnirs_ts_arr = np.array(fnirs_timestamps)

    # Save raw aligned arrays and timestamps inside the session directory
    np.save(os.path.join(session_dir, "X_video_aligned.npy"), X_video)
    np.save(os.path.join(session_dir, "X_fnirs_aligned.npy"), X_fnirs)
    np.save(os.path.join(session_dir, "frame_timestamps.npy"), frame_ts_arr)
    np.save(os.path.join(session_dir, "fnirs_timestamps.npy"), fnirs_ts_arr)
    print(f"\nSaved {len(X_video)} raw aligned pairs to {session_dir}")

    # Run timing audit and get measured FPS
    measured_fps = audit_sync_quality(frame_ts_arr, fnirs_ts_arr)

    # Generate framework-ready sliding windows
    X_vid_win, X_fnirs_win = prepare_training_windows(
        X_video, X_fnirs, measured_fps, WINDOW_SECONDS, OVERLAP_RATIO, FRAMEWORK
    )

    if X_vid_win is not None:
        np.save(os.path.join(session_dir, "X_video_windows.npy"), X_vid_win)
        np.save(os.path.join(session_dir, "X_fnirs_windows.npy"), X_fnirs_win)
        print(f"\nSaved windowed tensors ({FRAMEWORK.upper()} layout):")
        print(f"X_video_windows.npy shape: {X_vid_win.shape}")
        print(f"X_fnirs_windows.npy shape: {X_fnirs_win.shape}")
