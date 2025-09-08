"""
intan.processing._emg_trial_selector

Graphical tool for interactively labeling trial events on EMG recordings.

This GUI allows researchers to:
- Load and visualize EMG signals from `.rhd` files
- Select individual channels
- Click to mark trial onset points
- Assign labels to each indexed event
- Append new recordings for multi-session review
- Export trial events to a timestamped CSV or TXT file

Clicking the signal while "Set Trial Index" is enabled will add a labeled marker.
This tool is useful for supervised training of gesture classifiers, post-hoc annotation,
or protocol validation in EMG experiments.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Scrollbar, VERTICAL
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class EMGTrialSelector:
    """
    Tkinter-based application for manual EMG trial indexing.

    Attributes:
        emg_data (np.ndarray): EMG signal matrix (channels × samples)
        time_vector (np.ndarray): Time vector aligned with EMG samples
        sampling_rate (float): Sampling rate of amplifier
        current_channel (int): Channel index currently displayed
        indexing_enabled (bool): If True, allows user to click to insert marker
    """
    def __init__(self, root):
        self.root = root
        self.root.title("EMG Trial Selector")

        self.emg_data = None
        self.time_vector = None
        self.sampling_rate = None
        self.current_channel = 0
        self.indexing_enabled = False

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- Top Controls Frame ---
        control_frame = tk.Frame(root)
        control_frame.pack(side="top", fill="x", pady=5)

        tk.Button(control_frame, text="Load EMG File", command=self.load_file).pack(side="left", padx=5)
        tk.Button(control_frame, text="Set Trial Index", command=self.enable_indexing).pack(side="left", padx=5)
        tk.Button(control_frame, text="Append EMG File", command=self.append_file).pack(side="left", padx=5)

        # --- Main Frame (Canvas + Sidebar) ---
        main_frame = tk.Frame(root)
        main_frame.pack(side="top", fill="both", expand=True)

        # === Plot Area ===
        self.figure, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side="left", fill="both", expand=True)

        # === Sidebar Frame ===
        sidebar_frame = tk.Frame(main_frame)
        sidebar_frame.pack(side="right", fill="y", padx=10)

        # --- Channel Selector ---
        tk.Label(sidebar_frame, text="Channel:").pack(anchor="w")
        self.channel_selector = ttk.Combobox(sidebar_frame, state="readonly")
        self.channel_selector.bind("<<ComboboxSelected>>", self.update_channel)
        self.channel_selector.pack(fill="x", pady=5)

        # --- Label Entry Field ---
        tk.Label(sidebar_frame, text="Custom Label:").pack(anchor="w")
        self.label_entry = tk.Entry(sidebar_frame)
        self.label_entry.pack(fill="x", pady=5)
        self.label_entry.insert(0, "Label")  # Default text

        # --- Table ---
        self.table = ttk.Treeview(sidebar_frame, columns=("Sample Index", "Label"), show="headings", height=20)
        self.table.heading("Sample Index", text="Sample Index")
        self.table.heading("Label", text="Label")
        self.table.column("Sample Index", width=100)
        self.table.column("Label", width=100)
        self.table.pack(side="top", fill="y")

        scrollbar = Scrollbar(sidebar_frame, orient=VERTICAL, command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        # --- Save & Delete Buttons ---
        button_frame = tk.Frame(sidebar_frame)
        button_frame.pack(side="bottom", pady=10)
        tk.Button(button_frame, text="Save", command=self.save_table).pack(side="left", padx=5)
        tk.Button(button_frame, text="Delete", command=self.delete_selected).pack(side="left", padx=5)

        self.canvas.mpl_connect("button_press_event", self.on_click)

    def load_file(self):
        """
        Load EMG data from a .rhd file and initialize the GUI with the first channel.
        """
        from intan.io import load_rhd_file

        path = filedialog.askopenfilename(filetypes=[
            ("RHD files", "*.rhd"),
            ("CSV Files", "*.csv"),
            ("All files", "*.*")])
        if not path:
            return

        if path.endswith('.csv'):
            # Load CSV file. first column has timestamp data in milliseconds elapsed, the rest are EMG channels if
            # they have "EMG" in the name. The first row has only header information
            data = np.loadtxt(path, delimiter=',', skiprows=1)
            self.time_vector = data[:, 0] / 1000.0

            # Find the columns that contain "EMG" in their header
            with open(path, 'r') as f:
                header = f.readline().strip().split(',')
            emg_columns = [i for i, col in enumerate(header) if "EMG" in col]
            self.emg_data = data[:, emg_columns].T

            self.sampling_rate = 1000.0 / (self.time_vector[1] - self.time_vector[0])  # Assuming uniform sampling
        else:
            # Load RHD file. Should have dedicated structure
            result = load_rhd_file(path)
            self.emg_data = result["amplifier_data"]
            self.time_vector = result["t_amplifier"]
            self.sampling_rate = result["frequency_parameters"]["amplifier_sample_rate"]

        self.channel_selector['values'] = [f"Channel {i}" for i in range(self.emg_data.shape[0])]
        self.channel_selector.current(0)
        self.current_channel = 0
        self.plot_channel()

    def append_file(self):
        """
        Append EMG data from another .rhd file to the current data.
        """
        from intan.io import load_rhd_file

        path = filedialog.askopenfilename(filetypes=[("RHD files", "*.rhd"), ("All files", "*.*")])
        if not path:
            return

        result = load_rhd_file(path)
        new_emg = result["amplifier_data"]
        new_time = result["t_amplifier"]
        new_rate = result["frequency_parameters"]["amplifier_sample_rate"]

        if self.emg_data is None:
            # If no prior data, treat this like a fresh load
            self.emg_data = new_emg
            self.time_vector = new_time
            self.sampling_rate = new_rate
            self.channel_selector['values'] = [f"Channel {i}" for i in range(self.emg_data.shape[0])]
            self.channel_selector.current(0)
            self.current_channel = 0
            self.plot_channel()
            return

        # Sanity check: channel count and sampling rate must match
        if new_emg.shape[0] != self.emg_data.shape[0] or new_rate != self.sampling_rate:
            messagebox.showerror("Error", "Appended file must have same channel count and sampling rate.")
            return

        # Offset time vector based on last timestamp
        last_time = self.time_vector[-1]
        offset_time = new_time

        self.emg_data = np.concatenate((self.emg_data, new_emg), axis=1)
        self.time_vector = np.concatenate((self.time_vector, offset_time))
        self.plot_channel()

    def sample_index_to_timestamp(self, index):
        """
        Convert a sample index to a timestamp string.

        Parameters:
            index (int): Sample index to convert.

        Returns:
            str: Formatted timestamp string (HH:MM:SS).
        """
        seconds = index / self.sampling_rate
        return str(datetime.timedelta(seconds=int(seconds)))

    def save_table(self):
        """
        Save the trial markers to a text file with sample index and timestamp.
        """
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")],
            title="Save Trial Markers"
        )
        if not path:
            return

        # Collect and sort table data by sample index
        rows = []
        for row in self.table.get_children():
            sample_index, label = self.table.item(row)["values"]
            sample_index = int(sample_index)
            timestamp = self.sample_index_to_timestamp(sample_index)
            rows.append((sample_index, timestamp, label))

        rows.sort(key=lambda x: x[0])  # Sort by sample index

        # Write to text file
        with open(path, "w") as f:
            f.write("Sample Index,Timestamp,Label\n")
            for sample_index, timestamp, label in rows:
                f.write(f"{sample_index},{timestamp},{label}\n")

        messagebox.showinfo("Saved", f"Trial markers saved to:\n{path}")

    def delete_selected(self):
        """
        Delete selected rows from the table.
        """
        selected = self.table.selection()
        for item in selected:
            self.table.delete(item)

    def update_channel(self, event=None):
        """
        Update the current channel based on the selection from the dropdown.
        """
        if self.emg_data is None:
            return
        self.current_channel = self.channel_selector.current()
        self.plot_channel()

    def enable_indexing(self):
        """
        Enable the indexing mode to allow trial marking on the plot.
        """
        self.indexing_enabled = True

    def on_click(self, event):
        """
        Handle mouse click events on the plot to mark trial onset points.

        Parameters:
            event (matplotlib.backend_bases.Event): The mouse event.
        """
        if not self.indexing_enabled or event.inaxes != self.ax:
            return

        time_clicked = event.xdata
        amp_clicked = event.ydata
        if time_clicked is None:
            return

        sample_index = max(0, int(time_clicked * self.sampling_rate))
        self.ax.axvline(x=time_clicked, color='blue', linestyle='--')
        self.ax.axhline(y=amp_clicked, color='red', linestyle='--')
        self.canvas.draw()

        # Insert editable row into the table
        label = self.label_entry.get()
        self.table.insert("", "end", values=(sample_index, label))
        self.indexing_enabled = False

    def plot_channel(self):
        """
        Plot the currently selected EMG channel.
        """
        self.ax.clear()
        self.ax.plot(self.time_vector, self.emg_data[self.current_channel], label=f"Channel {self.current_channel}")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude (µV)")
        self.ax.set_title("EMG Signal")
        self.ax.legend()
        self.canvas.draw()

    def on_closing(self):
        self.root.quit()

