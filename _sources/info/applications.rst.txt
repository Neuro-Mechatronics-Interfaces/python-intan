Applications
===============

This section highlights advanced GUI applications provided with the `intan` package.


EMG Viewer GUI
------------------

The `EMGViewer` application is a full-featured GUI for visualizing, filtering, and segmenting EMG data from Intan `.rhd` recordings.

**Features:**
- Interactive time-domain and frequency-domain plots
- Channel and segment selection
- Filtering (bandpass, notch)
- Manual trial labeling and segmentation
- Feature extraction for ML
- Integrated training data and model-building tools

**How to launch:**

.. code-block:: python

    from intan.applications import EMGViewer
    import tkinter as tk

    root = tk.Tk()
    app = EMGViewer(root)
    root.mainloop()

Or, launch with the splash screen:

.. code-block:: python

    from intan.applications import launch_emg_viewer
    launch_emg_viewer()

**Screenshots:**
*(Insert screenshots or animated GIFs here if available)*

**Typical workflow:**
1. Click "Load File" to open a `.rhd` EMG data file.
2. Explore channels, apply filters, segment trials, and export results.
3. Use tabs for feature extraction and model building.


EMG Trial Selector
----------------------

The `EMGSelector` application provides an intuitive GUI to mark, label, and export trial events in EMG recordings.

**Features:**
- Load `.rhd` EMG files and visualize data by channel
- Click on the plot to mark trial onset points
- Label each event and export to a CSV or TXT file
- Append multiple sessions for review
- Use for supervised labeling, protocol annotation, or classifier training

**How to launch:**

.. code-block:: python

    from intan.applications import EMGSelector, launch_emg_selector
    import tkinter as tk

    # Manual launch:
    root = tk.Tk()
    app = EMGSelector(root)
    root.mainloop()

    # One-line launcher:
    launch_emg_selector()

**Screenshots:**
*(Insert screenshots or GIFs showing labeling workflow if available)*

**Typical workflow:**
1. Load an `.rhd` file.
2. Select channel and label mode.
3. Click to mark trials, assign labels, and export.
