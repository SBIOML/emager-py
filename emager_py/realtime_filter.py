import numpy as np
from scipy.signal import butter, iirnotch, tf2sos, sosfilt_zi, sosfilt
import matplotlib.pyplot as plt
from collections import deque


# --- Filter design (do this once) ---
def design_emg_filter(fs, band=(20, 350), notch_freq=60, notch_Q=30, order=2):
    nyq = 0.5 * fs

    # Bandpass filter
    low, high = band[0] / nyq, band[1] / nyq
    sos_bandpass = butter(order, [low, high], btype="band", output="sos")

    # Notch filter
    w0 = notch_freq / nyq
    b_notch, a_notch = iirnotch(w0, notch_Q)
    sos_notch = tf2sos(b_notch, a_notch)

    # Combine filters
    sos_combined = np.vstack((sos_bandpass, sos_notch))

    # Initialize filter state
    zi = sosfilt_zi(sos_combined)

    return sos_combined, zi


# --- Real-time filter class ---
class RealTimeEMGFilter:
    def __init__(self, fs, n_channels=64):
        self.sos, zi = design_emg_filter(fs)
        # Initialize filter state for each channel
        self.zi = np.tile(zi[:, :, np.newaxis], (1, 1, n_channels))
        self.n_channels = n_channels

        # For visualization
        self.buffer_raw = deque(maxlen=1000)
        self.buffer_filtered = deque(maxlen=1000)

    def process_chunk(self, chunk):
        """Process a chunk of EMG data with shape (N, C)"""
        # Apply filter and update state
        filtered, self.zi = sosfilt(self.sos, chunk, axis=-2, zi=self.zi)
        return filtered

    def process_sample(self, sample):
        """Process a single sample (1D array with n_channels)"""
        # Reshape to 2D (1 sample × channels)
        sample_2d = sample.reshape(1, -1)
        filtered = self.process_chunk(sample_2d)

        # Store for visualization
        self.buffer_raw.append(sample[0])  # Store first channel
        self.buffer_filtered.append(filtered[0, 0])

        return filtered[0]  # Return as 1D array


# --- Simulate real-time data acquisition ---
def simulate_realtime():
    fs = 1000  # Hz
    n_channels = 64

    # Create filter
    emg_filter = RealTimeEMGFilter(fs, n_channels)

    # Setup plot
    plt.figure(figsize=(12, 6))
    (line_raw,) = plt.plot([], [], "gray", alpha=0.7, label="Raw EMG")
    (line_filtered,) = plt.plot([], [], "blue", label="Filtered EMG")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.title("Real-time EMG Filtering")
    plt.legend()
    plt.grid(True)

    buffer_raw = deque(maxlen=1000)

    for i in range(1, 5 * fs + 1):
        # Generate noisy sample with 60Hz interference
        t = i / fs
        sample = 0.5 * np.random.randn(n_channels) + 0.2 * np.sin(2 * np.pi * 60 * t)

        buffer_raw.append(sample)

        # Process sample
        # filtered_sample = emg_filter.process_sample(sample)

        # Update plot every 100 samples
        if i % 100 == 0:
            raw = np.array(buffer_raw)
            filtered = emg_filter.process_chunk(raw)
            line_raw.set_data(range(len(raw)), raw[:, 0])
            line_filtered.set_data(range(len(filtered)), filtered[:, 0])

            plt.xlim(0, len(raw))
            plt.ylim(-1, 1)
            plt.pause(0.1)
            # Legend in upper-right
            plt.legend(loc="upper right")
            buffer_raw.clear()

    plt.show()


# Run simulation
if __name__ == "__main__":
    simulate_realtime()
