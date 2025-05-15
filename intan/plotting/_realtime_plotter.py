import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from scipy.signal import iirnotch, butter, filtfilt


class RealtimePlotter:
    def __init__(self, client, sampling_rate=2000.0, plotting_interval=1.0,
                 samples_per_fetch=50, channels_to_plot=None):
        self.client = client
        self.sampling_rate = sampling_rate
        self.plotting_interval = plotting_interval
        self.samples_per_fetch = samples_per_fetch

        self.buffer_size = int(self.plotting_interval * self.sampling_rate)
        self.x = np.linspace(0, self.plotting_interval, self.buffer_size)

        # Define which channels to plot
        if channels_to_plot is None:
            self.channels_to_plot = [0]  # default to channel 0
        else:
            self.channels_to_plot = channels_to_plot

        self.n_channels = len(self.channels_to_plot)
        self.ydata = {
            ch: np.zeros(self.buffer_size, dtype=np.float32)
            for ch in self.channels_to_plot
        }

        self.init_filters()
        self.init_plot()

    def init_filters(self):
        notch_freq = 60.0
        Q = 30.0
        self.notch_b, self.notch_a = iirnotch(notch_freq, Q, fs=self.sampling_rate)

        lowcut = 10.0
        highcut = 500.0
        nyq = 0.5 * self.sampling_rate
        self.bp_b, self.bp_a = butter(4, [lowcut / nyq, highcut / nyq], btype='band')

    def apply_filters(self, signal):
        signal = filtfilt(self.notch_b, self.notch_a, signal)
        signal = filtfilt(self.bp_b, self.bp_a, signal)
        return signal

    def init_plot(self):
        self.figure, self.axes = plt.subplots(self.n_channels, 1, figsize=(10, 2 * self.n_channels), sharex=True)
        if self.n_channels == 1:
            self.axes = [self.axes]

        self.hl = []
        for ax, ch in zip(self.axes, self.channels_to_plot):
            ax.set_xlim(0, self.plotting_interval)
            ax.set_ylim(-200, 200)
            ax.set_ylabel(f"Ch {ch}")
            ax.set_facecolor('#001230')
            line, = ax.plot(self.x, self.ydata[ch], lw=0.5, color='#d92eab')
            self.hl.append(line)

        self.axes[-1].set_xlabel("Time (s)")
        self.figure.suptitle("Realtime EMG", fontsize=14)
        plt.subplots_adjust(hspace=0.3)

        # Slider to control y-axis range
        axcolor = 'lightgoldenrodyellow'
        axylim = plt.axes([0.1, 0.02, 0.65, 0.03], facecolor=axcolor)
        self.sylim = Slider(axylim, 'Ylim', 10, 1000, valinit=200)

        def update_slider(val):
            for ax in self.axes:
                ax.set_ylim(-val, val)
            self.figure.canvas.draw_idle()

        self.sylim.on_changed(update_slider)

        self.anim = FuncAnimation(self.figure, self.update, interval=20)

    def update(self, frame):
        for i, ch in enumerate(self.channels_to_plot):
            new_samples = self.client.get_samples(channel=ch, n_samples=self.samples_per_fetch)
            if new_samples:
                new_arr = np.array(new_samples, dtype=np.float32)
                if len(new_arr) >= 5:
                    new_arr = self.apply_filters(new_arr)
                self.ydata[ch] = np.roll(self.ydata[ch], -len(new_arr))
                self.ydata[ch][-len(new_arr):] = new_arr
                self.hl[i].set_ydata(self.ydata[ch])
        return self.hl

    def run(self):
        print("Launching real-time plotter...")
        plt.show()

def run_realtime_plot(client, channels=[0], sampling_rate=2000):
    plotter = RealtimePlotter(client, sampling_rate=sampling_rate, channels_to_plot=channels)
    plotter.run()
