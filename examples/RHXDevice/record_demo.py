import numpy as np

from intan.interface import IntanRHXDevice
#from intan.plotting import plot_figure

if __name__ == "__main__":
    SAMPLE_RATE = 4000.0
    DURATION_SEC = 10
    NUM_CHANNELS = 128

    # Initialize the RHX device
    device = IntanRHXDevice(sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS)

    # Configure the device channels
    device.enable_wide_channel(range(NUM_CHANNELS))
    device.set_blocks_per_write(8)

    try:
        # Start the RHX device and collect data for the specified duration
        emg = device.record(duration_sec=DURATION_SEC)

    except KeyboardInterrupt:
        print("\n[✋] Streaming interrupted by user.")
        emg = None

    finally:
        # Stop the RHX device and close the connection
        device.set_run_mode("stop")
        device.close()

        # Plot the EMG data
        if emg is not None:
            t = np.arange(emg.shape[1]) / SAMPLE_RATE
            #plot_figure(emg[5], t, title="EMG Signal", x_label="Time (s)", y_label="Amplitude (uV)", legend=False)

        import matplotlib.pyplot as plt
        plt.plot(t, emg[5])
        plt.title("EMG Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (uV)")
        plt.show()
