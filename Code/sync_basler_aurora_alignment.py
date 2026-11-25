import mne
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# --- 1. File Paths (Update these to your actual files) ---
snirf_path = r"C:\Users\jomul\Documents\NIRx\Data\your_subject_folder\your_file.snirf"
video_path = r"face_video_1234567890.avi"
csv_path = r"face_timestamps_1234567890.csv"

# --- 2. Load .snirf and Find START / STOP Trigger Times ---
raw = mne.io.read_raw_snirf(snirf_path, preload=True)

# Optional: Convert raw optical intensity -> optical density -> HbO/HbR
# raw_od = mne.preprocessing.nirs.optical_density(raw)
# raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
# data_source = raw_haemo
data_source = raw  # Keep raw intensity (or swap with raw_haemo above)

# Extract START and STOP onset times (in seconds relative to Aurora recording start)
annotations = data_source.annotations
start_time = None
stop_time = None

for onset, desc in zip(annotations.onset, annotations.description):
    if 'START' in str(desc):
        start_time = onset
    elif 'STOP' in str(desc):
        stop_time = onset

if start_time is None or stop_time is None:
    raise ValueError(f"Could not find both START and STOP triggers. Found annotations: {set(annotations.description)}")

print(f"Found triggers -> START: {start_time:.3f}s, STOP: {stop_time:.3f}s (Duration: {stop_time - start_time:.3f}s)")

# Crop fNIRS data strictly between START and STOP
raw_cropped = data_source.copy().crop(tmin=start_time, tmax=stop_time)
fnirs_data = raw_cropped.get_data().T  # Shape: (n_fnirs_samples, n_channels)
fnirs_rel_times = np.linspace(0, stop_time - start_time, len(fnirs_data))

# --- 3. Load Video Frames and Relative Timestamps ---
df_ts = pd.read_csv(csv_path)
video_rel_times = df_ts['lsl_timestamp'].values - df_ts['lsl_timestamp'].values[0]

cap = cv2.VideoCapture(video_path)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Optional: resize frame here for your neural network (e.g., 224x224)
    # frame = cv2.resize(frame, (224, 224))
    frames.append(frame)
cap.release()

video_frames = np.array(frames)  # Shape: (n_frames, H, W, 3)

# Ensure frame count matches CSV row count
min_len = min(len(video_frames), len(video_rel_times))
video_frames = video_frames[:min_len]
video_rel_times = video_rel_times[:min_len]

# --- 4. Align Both Modalities 1-to-1 (Frame-by-Frame) ---
# Interpolates the lower-rate fNIRS signal onto each exact video frame timestamp
interp_func = interp1d(fnirs_rel_times, fnirs_data, axis=0, kind='linear', fill_value="extrapolate")
fnirs_aligned = interp_func(video_rel_times)  # Shape: (n_frames, n_channels)

# --- 5. Save Final Aligned NumPy Arrays ---
np.save("X_video_aligned.npy", video_frames)
np.save("X_fnirs_aligned.npy", fnirs_aligned)

print(f"Alignment complete!")
print(f"X_video_aligned.npy shape: {video_frames.shape}")
print(f"X_fnirs_aligned.npy shape: {fnirs_aligned.shape}")
