import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import butter, filtfilt
import utilities.intan_utilities as intan_utils
from sklearn.decomposition import FastICA
import time

# === Bandpass Filter Function ===
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


# === Load HDEMG Data ===
#file_path = r'G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\intan_HDEMG_sleeve\Jonathan\raw\2024_10_22\IsometricFlexionRamp_1_241022_150834\IsometricFlexionRamp_1_241022_150834.rhd'
file_path = r'C:\Users\HP\Desktop\Temp\HDEMG_comparison_042325\forceRamp3_250423_164802\forceRamp3_250423_164802.rhd'
result = intan_utils.load_rhd_file(file_path)

emg_data = result['amplifier_data']          # Shape: (channels, samples)
fs = result['frequency_parameters']['amplifier_sample_rate']
t_s = result['t_amplifier']

#start_t = int(16*fs)  # Start at 5 seconds
#end_t = int(26*fs)  # int(20*fs)    # End at 10 seconds
start_t = 0  # Start at 5 seconds
end_t = emg_data.shape[1]  # int(20*fs)    # End at 10 seconds

channel_idx = 65

emg_data = emg_data[:, start_t:end_t]  # Crop the data to the desired time window
t_s = t_s[start_t:end_t]  # Crop the time vector to match the data

# === Step 1: Filter ALL channels ===
filtered_emg = np.array([bandpass_filter(chan, 10, 500, fs) for chan in emg_data])

# === Step 2: PCA decomposition and reconstruction ===
n_components=15
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(filtered_emg.T)         # Shape: (samples, components)
X_reconstructed = pca.inverse_transform(X_pca).T  # Shape: (channels, samples)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Find how many components reach 95% variance
n_pca_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components to retain 95% variance: {n_pca_95}")

plt.plot(cumulative_variance * 100)
plt.axhline(y=95, color='r', linestyle='--', label='95% threshold')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained (%)")
plt.title("Screen Plot")
plt.legend()
plt.grid(True)
plt.show()

tic = time.time()
pca_k = PCA(n_components=n_pca_95)
X_pca_k = pca_k.fit_transform(filtered_emg.T)
X_reconstructed_k = pca_k.inverse_transform(X_pca_k).T  # Shape: [channels, samples]
print(f"PCA reconstruction time: {time.time() - tic:.2f} seconds")

# === Step 3: Plot filtered signal vs PCA reconstruction on one channel ===
plt.figure(figsize=(14, 8))


# --- Plot 1: Filtered vs PCA-reconstructed signal ---
plt.subplot(2, 1, 1)
plt.plot(t_s, filtered_emg[channel_idx], label='Filtered', alpha=0.8)
plt.plot(t_s, X_reconstructed_k[channel_idx], label='PCA Reconstructed', alpha=0.6, linewidth=0.5)
plt.title(f"Channel {channel_idx} - Filtered vs PCA Reconstruction")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# --- Plot 2: Variance explained ---
plt.subplot(2, 1, 2)
plt.bar(np.arange(1, n_components+1), pca.explained_variance_ratio_ * 100)
plt.title("Explained Variance by Principal Components")
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained (%)")
plt.tight_layout()
plt.show()

# Now using FastICA
# === ICA Analysis ===
tic = time.time()
X = filtered_emg.T
n_ica_components = 20
ica = FastICA(n_components=n_ica_components, random_state=42)
X_ica = ica.fit_transform(X)
print(f"ICA decomposition time: {time.time() - tic:.2f} seconds")

# Visualize ICA Components
plt.figure(figsize=(15, 12))
for i in range(n_ica_components):
    plt.subplot(5, 4, i + 1)
    plt.plot(X_ica[:80000, i])
    plt.title(f"ICA Comp {i+1}")
    plt.xticks([]); plt.yticks([])
plt.tight_layout()
plt.show()

# === ICA Reconstruction ===
#good_ica_components = [0, 4, 7, 8, 14, 16, 18]  # Adjust as needed
#good_ica_components = [0, 1, 3, 6, 7, 9, 12]  # Adjust as needed
good_ica_components = [0, 2, 3, 5, 8, 12, 13, 15, 18]  # Adjust as needed
X_ica_filtered = np.zeros_like(X_ica)
X_ica_filtered[:, good_ica_components] = X_ica[:, good_ica_components]
X_ica_recon_all = ica.inverse_transform(X_ica).T
X_ica_recon_good = ica.inverse_transform(X_ica_filtered).T

# === ICA Comparison Plot ===
plt.figure(figsize=(14, 6))
plt.plot(t_s, filtered_emg[channel_idx], label='Filtered EMG', alpha=0.6)
plt.plot(t_s, X_ica_recon_all[channel_idx], label='ICA Full Recon', linestyle='--')
plt.plot(t_s, X_ica_recon_good[channel_idx], label='ICA Denoised', linewidth=2)
plt.title(f"Channel {channel_idx} - ICA Reconstruction Comparison")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()