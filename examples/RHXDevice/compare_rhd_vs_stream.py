import matplotlib.pyplot as plt
from intan.io import load_rhd_file
from intan.rhx_interface import IntanRHXDevice

# === Config ===
SAMPLES_TO_COMPARE = 8000  # Assume 4kHz sampling rate
CHANNEL_INDEX = 15


# ================= TCP Section ==================
# === 1) Initialize and configure RHX device ===
print("Connecting to RHX TCP client...")
device = IntanRHXDevice()
device.enable_wide_channel(CHANNEL_INDEX)

# === 2) Create streamer and get data ===
print("Streaming data...")
timestamps, voltages = device.record(duration_sec=2)
voltages = voltages[0]  # Use the single channel directly
print("✅ Client data collection complete.")
device.close()
# ================================================

# ============== .rhd File Section ==============
# === 1) Load RHD file ===
print("Loading RHD file...")
RHD_PATH = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\2024_10_22\raw\RingFlexion_241022_144153\RingFlexion_241022_144153.rhd"
rhd_result = load_rhd_file(RHD_PATH)
print(f"Shape of emg data: {rhd_result['amplifier_data'].shape}")

# === 2) Parse out 2 seconds of data ===
rhd_data = rhd_result["amplifier_data"][:, :SAMPLES_TO_COMPARE]
fs_rhd = rhd_result["frequency_parameters"]["amplifier_sample_rate"]
ts_rhd = rhd_result["t_amplifier"][:SAMPLES_TO_COMPARE]
print(f"RHD file sampling rate: {fs_rhd} Hz")

# ================================================
# # ================= Comparison Section ==================

# Plot both signals
plt.figure(figsize=(12, 4))
plt.plot(ts_rhd, rhd_data[CHANNEL_INDEX], label="RHD File", linewidth=1)
plt.plot(timestamps, voltages, '--', label="Live TCP", linewidth=1)
plt.title(f"Channel {CHANNEL_INDEX} — First {SAMPLES_TO_COMPARE} Samples")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

