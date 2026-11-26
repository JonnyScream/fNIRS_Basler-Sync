import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


def find_latest_session(search_roots):
    """Finds the most recently modified recording session directory."""
    candidates = []
    for root in search_roots:
        if os.path.isdir(root):
            for entry in os.scandir(root):
                if entry.is_dir() and os.path.exists(os.path.join(entry.path, "frame_timestamps.npy")):
                    candidates.append(entry.path)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def plot_sync_audit(frame_ts, fnirs_ts, output_dir):
    """Plots instantaneous camera FPS and fNIRS synchronization lag over time."""
    t_rel = frame_ts - frame_ts[0]
    frame_diffs_ms = np.diff(frame_ts) * 1000.0
    inst_fps = 1000.0 / (frame_diffs_ms + 1e-9)
    sync_lag_ms = (frame_ts - fnirs_ts) * 1000.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # 1. Camera Frame Rate Stability
    axes[0].plot(t_rel[1:], inst_fps, color="#1f77b4", linewidth=1.0, label="Instantaneous FPS")
    axes[0].axhline(np.mean(inst_fps), color="#d62728", linestyle="--", label=f"Mean: {np.mean(inst_fps):.2f} Hz")
    axes[0].set_ylabel("Camera FPS (Hz)")
    axes[0].set_title("Hardware Synchronization & Timing Stability")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # 2. fNIRS Sample Age / Sync Lag
    axes[1].plot(t_rel, sync_lag_ms, color="#2ca02c", linewidth=1.0, label="fNIRS Sample Age (ms)")
    axes[1].axhline(np.mean(sync_lag_ms), color="#ff7f0e", linestyle="--", label=f"Mean Lag: {np.mean(sync_lag_ms):.1f} ms")
    axes[1].set_ylabel("Sync Lag (ms)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "01_sync_and_jitter.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_fnirs_signals(frame_ts, fnirs_data, output_dir, max_channels=8):
    """Plots Z-score normalized fNIRS channels over time."""
    t_rel = frame_ts - frame_ts[0]
    n_total_ch = fnirs_data.shape[1]
    # Skip channel 0 if it's a sample counter from Aurora, plot active signal channels
    start_ch = 1 if n_total_ch > 8 else 0
    end_ch = min(n_total_ch, start_ch + max_channels)
    subset = fnirs_data[:, start_ch:end_ch]

    # Z-score normalize for clear visual comparison
    subset_norm = (subset - np.mean(subset, axis=0)) / (np.std(subset, axis=0) + 1e-8)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(subset_norm.shape[1]):
        ax.plot(t_rel, subset_norm[:, i] + i * 3.5, linewidth=1.0, label=f"Ch {start_ch + i}")

    ax.set_title(f"fNIRS Signals (Channels {start_ch}-{end_ch - 1}, Z-Scored & Offset)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Normalized Amplitude (offset per channel)")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "02_fnirs_signals.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_eeg_signals(eeg_ts, eeg_data, t0, output_dir):
    """Plots all 8 Unicorn EEG channels and their Power Spectral Density (PSD)."""
    t_rel = eeg_ts - t0
    n_ch = eeg_data.shape[1]
    ch_names = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"][:n_ch]

    # Z-score normalize for clean stacked visualization
    eeg_norm = (eeg_data - np.mean(eeg_data, axis=0)) / (np.std(eeg_data, axis=0) + 1e-8)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1.4, 1.0]})

    # 1. Stacked Time-Series
    for i in range(n_ch):
        label = ch_names[i] if i < len(ch_names) else f"EEG {i}"
        axes[0].plot(t_rel, eeg_norm[:, i] + i * 4.0, linewidth=0.7, label=label)

    axes[0].set_title(f"Unicorn EEG Signals ({n_ch} Channels, Z-Scored & Offset)")
    axes[0].set_ylabel("Normalized Amplitude")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 2. Power Spectral Density (PSD) to check Alpha (8-12Hz) & Mains Hum (50Hz)
    srate = 1.0 / np.mean(np.diff(eeg_ts))
    freqs = np.fft.rfftfreq(len(eeg_data), d=1.0 / srate)
    psd = np.mean(np.abs(np.fft.rfft(eeg_data - np.mean(eeg_data, axis=0), axis=0)) ** 2, axis=1)

    mask = (freqs >= 1.0) & (freqs <= 60.0)
    axes[1].semilogy(freqs[mask], psd[mask], color="#9467bd", linewidth=1.2)
    axes[1].axvline(50.0, color="red", linestyle="--", alpha=0.6, label="50 Hz Power Line")
    axes[1].axvspan(8.0, 12.0, color="green", alpha=0.15, label="Alpha Band (8-12 Hz)")
    axes[1].set_title(f"Mean EEG Power Spectral Density (Measured Rate: {srate:.1f} Hz)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power (log scale)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "03_eeg_signals_and_psd.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_multimodal_dashboard(video_data, frame_ts, fnirs_data, eeg_data, eeg_ts, output_dir):
    """Creates a unified dashboard aligning video snapshots with fNIRS and EEG timelines."""
    t0 = frame_ts[0]
    t_rel = frame_ts - t0
    n_frames = len(video_data)

    # Pick 5 evenly spaced frames across the recording
    sample_indices = np.linspace(0, n_frames - 1, 5, dtype=int)

    has_eeg = eeg_data is not None and eeg_ts is not None and len(eeg_data) > 0
    n_rows = 3 if has_eeg else 2

    fig = plt.figure(figsize=(14, 3.2 * n_rows))
    gs = fig.add_gridspec(n_rows, 5, height_ratios=[1.2] + [1.0] * (n_rows - 1))

    # Top Row: 5 Sample Video Frames
    for col, idx in enumerate(sample_indices):
        ax_img = fig.add_subplot(gs[0, col])
        rgb_frame = cv2.cvtColor(video_data[idx], cv2.COLOR_BGR2RGB)
        ax_img.imshow(rgb_frame)
        ax_img.set_title(f"t = {t_rel[idx]:.1f}s (Frame {idx})", fontsize=9)
        ax_img.axis("off")

    # Middle Row: fNIRS Signal Trace with Frame Markers
    ax_fnirs = fig.add_subplot(gs[1, :])
    n_total_fnirs = fnirs_data.shape[1]
    start_ch = 1 if n_total_fnirs > 8 else 0
    end_ch = min(n_total_fnirs, start_ch + 4)
    fnirs_sub = fnirs_data[:, start_ch:end_ch]
    fnirs_norm = (fnirs_sub - np.mean(fnirs_sub, axis=0)) / (np.std(fnirs_sub, axis=0) + 1e-8)

    for i in range(fnirs_norm.shape[1]):
        ax_fnirs.plot(t_rel, fnirs_norm[:, i], linewidth=1.0, label=f"fNIRS Ch {start_ch + i}")
    for idx in sample_indices:
        ax_fnirs.axvline(t_rel[idx], color="black", linestyle=":", alpha=0.6)
    ax_fnirs.set_title("Aligned fNIRS Activity")
    ax_fnirs.set_ylabel("Z-Score")
    ax_fnirs.legend(loc="upper right", ncol=4, fontsize=8)
    ax_fnirs.grid(True, alpha=0.3)

    # Bottom Row (Optional): EEG Trace with Frame Markers
    if has_eeg:
        ax_eeg = fig.add_subplot(gs[2, :], sharex=ax_fnirs)
        t_eeg = eeg_ts - t0
        n_plot_eeg = min(4, eeg_data.shape[1])
        eeg_sub = eeg_data[:, :n_plot_eeg]
        eeg_norm = (eeg_sub - np.mean(eeg_sub, axis=0)) / (np.std(eeg_sub, axis=0) + 1e-8)
        ch_names = ["Fz", "C3", "Cz", "C4"][:n_plot_eeg]
        for i in range(n_plot_eeg):
            ax_eeg.plot(t_eeg, eeg_norm[:, i] + i * 3.0, linewidth=0.6, label=ch_names[i])
        for idx in sample_indices:
            ax_eeg.axvline(t_rel[idx], color="black", linestyle=":", alpha=0.6)
        ax_eeg.set_title("Aligned Unicorn EEG Activity (First 4 Channels)")
        ax_eeg.set_ylabel("Z-Score (offset)")
        ax_eeg.set_xlabel("Time (seconds)")
        ax_eeg.legend(loc="upper right", ncol=4, fontsize=8)
        ax_eeg.grid(True, alpha=0.3)
    else:
        ax_fnirs.set_xlabel("Time (seconds)")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "04_multimodal_dashboard.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize recorded multimodal session data and save PNG plots.")
    parser.add_argument(
        "--session_dir",
        type=str,
        default="",
        help="Path to a specific session folder (leave empty to auto-select the latest recording)",
    )
    args = parser.parse_args()

    session_dir = args.session_dir.strip()
    if not session_dir:
        default_roots = [
            "../../fNIRS_Basler_recordings/recordings",
            "../fNIRS_Basler_recordings/recordings",
            "../recordings",
            "recordings",
        ]
        session_dir = find_latest_session(default_roots)

    if not session_dir or not os.path.isdir(session_dir):
        raise FileNotFoundError(
            "Could not find a valid session directory. Pass it explicitly via: "
            "python visualize_session.py --session_dir <path_to_folder>"
        )

    print(f"Loading data from: {session_dir}")
    output_dir = os.path.join(session_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Load core aligned modalities
    video_data = np.load(os.path.join(session_dir, "X_video_aligned.npy"))
    fnirs_data = np.load(os.path.join(session_dir, "X_fnirs_aligned.npy"))
    frame_ts = np.load(os.path.join(session_dir, "frame_timestamps.npy"))
    fnirs_ts = np.load(os.path.join(session_dir, "fnirs_timestamps.npy"))

    # Load optional EEG data if present
    eeg_path = os.path.join(session_dir, "X_eeg_raw.npy")
    eeg_ts_path = os.path.join(session_dir, "eeg_timestamps.npy")
    eeg_data = np.load(eeg_path) if os.path.exists(eeg_path) else None
    eeg_ts = np.load(eeg_ts_path) if os.path.exists(eeg_ts_path) else None

    # Generate and save all plots
    plot_sync_audit(frame_ts, fnirs_ts, output_dir)
    plot_fnirs_signals(frame_ts, fnirs_data, output_dir)

    if eeg_data is not None and eeg_ts is not None:
        plot_eeg_signals(eeg_ts, eeg_data, frame_ts[0], output_dir)

    plot_multimodal_dashboard(video_data, frame_ts, fnirs_data, eeg_data, eeg_ts, output_dir)
    print(f"\nAll plots saved successfully inside: {output_dir}")
