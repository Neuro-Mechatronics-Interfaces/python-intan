Comparing RHD File Data to Live TCP Stream
==============================================

This example demonstrates how to compare the data recorded to an Intan `.rhd` file with data streamed in real time over TCP from the same device and channel.
This is useful for verifying that your live streaming pipeline matches the saved data and for troubleshooting.

**Requirements:**
- A recent `.rhd` data file recorded with the Intan RHX system
- The same Intan RHX Controller device available for TCP streaming
- `matplotlib` for plotting

**You’ll see how to:**
- Connect to the RHX device and stream from a selected channel
- Load a `.rhd` file and extract a matching data segment
- Plot and visually compare both traces

----

**Example code:**

.. code-block:: python

    """
    Example: Compare saved .rhd file data to live TCP streamed data from Intan RHX.
    """

    import matplotlib.pyplot as plt
    from intan.io import load_rhd_file
    from intan.rhx_interface import IntanRHXDevice, config_options

    # === Config ===
    SAMPLES_TO_COMPARE = 8000    # Number of samples to compare (e.g., 2 seconds at 4 kHz)
    CHANNEL_INDEX = 15           # Channel to compare

    # --- 1. Stream data live from RHX device ---
    print("Connecting to RHX TCP client...")
    device = IntanRHXDevice()
    config_options.set_channel(device, 'a', channel=CHANNEL_INDEX, enable_wide=True)

    print("Streaming data...")
    timestamps, voltages = device.stream(duration_sec=2)
    voltages = voltages[0]  # Use the single channel directly
    print("✅ Client data collection complete.")
    device.close()

    # --- 2. Load same channel from RHD file ---
    print("Loading RHD file...")
    RHD_PATH = r"PATH/TO/YOUR_FILE.rhd"  # <-- Update this path as needed
    rhd_result = load_rhd_file(RHD_PATH)
    print(f"Shape of emg data: {rhd_result['amplifier_data'].shape}")

    rhd_data = rhd_result["amplifier_data"][CHANNEL_INDEX, :SAMPLES_TO_COMPARE]
    fs_rhd = rhd_result["frequency_parameters"]["amplifier_sample_rate"]
    ts_rhd = rhd_result["t_amplifier"][:SAMPLES_TO_COMPARE]
    print(f"RHD file sampling rate: {fs_rhd} Hz")

    # --- 3. Plot both signals ---
    plt.figure(figsize=(12, 4))
    plt.plot(ts_rhd, rhd_data, label="RHD File", linewidth=1)
    plt.plot(timestamps, voltages, '--', label="Live TCP", linewidth=1)
    plt.title(f"Channel {CHANNEL_INDEX} — First {SAMPLES_TO_COMPARE} Samples")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

----

**Expected outcome:**

- Both traces should overlap closely (after any expected delays or offsets), demonstrating that your live data stream matches what is saved to file.
- Terminal output will confirm data shape and sampling rate.

**Tips:**
- Use the same channel and sampling rate for both streaming and file loading.
- Update the `RHD_PATH` variable with the path to your `.rhd` file.
- Adjust `SAMPLES_TO_COMPARE` and `CHANNEL_INDEX` as needed for your test.

**Troubleshooting:**
- Ensure the RHX TCP server is running and enabled.
- Check for any timing mismatches, which may arise if the file and stream are not started simultaneously.

