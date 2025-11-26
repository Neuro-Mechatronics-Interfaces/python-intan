GUI Applications
=================

The `intan` package includes several PyQt5-based GUI applications for real-time EMG visualization, data annotation, and gesture control. These provide user-friendly interfaces for common workflows.

**Requirements:**

- PyQt5 (``pip install PyQt5``)
- PyQtGraph (``pip install pyqtgraph``)
- Intan RHX device with TCP streaming (for real-time apps)

----

EMG Viewer Application
-----------------------

A comprehensive real-time EMG visualization tool with multiple display modes and filtering options.

.. image:: ../../assets/emg_viewer_screenshot.png
    :alt: EMG Viewer Application
    :width: 800
    :align: center

**Launch the application:**

.. code-block:: bash

    # From examples directory
    python examples/applications/run_emg_viewer.py

**Programmatic usage:**

.. code-block:: python

    """
    Launch EMG Viewer from Python script
    Script: examples/applications/run_emg_viewer.py
    """
    from intan.applications import EMGViewer
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    # Initialize viewer
    viewer = EMGViewer(
        num_channels=128,
        sample_rate=4000,
        window_duration=3.0,  # 3-second display window
    )

    # Optional: Configure channels to display
    viewer.set_visible_channels(range(0, 64))  # Show first 64 channels

    viewer.show()
    sys.exit(app.exec_())

**Features:**

- **Multiple display modes**:

  - Stacked waveforms
  - Waterfall plot
  - Heatmap view
  - RMS grid

- **Real-time filtering**:

  - Notch filter (50/60 Hz)
  - Bandpass (10-500 Hz)
  - High-pass DC removal

- **Channel selection**:

  - Select/deselect individual channels
  - Group selection by port
  - Custom channel groupings

- **Recording**:

  - Record to `.npz` format
  - Add time-stamped markers
  - Export to CSV

**Keyboard shortcuts:**

- ``Space``: Pause/Resume streaming
- ``R``: Start/Stop recording
- ``M``: Add marker at current time
- ``F``: Toggle filtering
- ``+/-``: Adjust amplitude scale

----

Trial Selector Application
----------------------------

Interactive tool for annotating and segmenting EMG trials from recorded data.

.. code-block:: bash

    # Launch trial selector
    python examples/applications/run_trial_selector.py

**Usage:**

.. code-block:: python

    """
    Script: examples/applications/run_trial_selector.py
    """
    from intan.applications import TrialSelector
    from intan.io import load_rhd_file
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    # Load data
    result = load_rhd_file()
    emg_data = result['amplifier_data']
    fs = result['frequency_parameters']['amplifier_sample_rate']
    t = result['t_amplifier']

    # Launch selector
    selector = TrialSelector(
        data=emg_data,
        time_vector=t,
        sample_rate=fs,
        channel_names=[f"Ch{i}" for i in range(emg_data.shape[0])]
    )

    selector.show()
    sys.exit(app.exec_())

**Features:**

- Visual trial boundary selection
- Multi-label annotation
- Trial quality rating
- Export selected trials
- Gesture timing verification
- Automatic trial detection based on amplitude threshold

**Workflow:**

1. Load recorded EMG file
2. Zoom/pan to find trial boundaries
3. Click to mark trial start/end
4. Label trial with gesture name
5. Rate trial quality (good/bad/uncertain)
6. Export annotated trials for training

**Output format:**

Saves annotations as CSV:

::

    trial_id,start_time,end_time,label,quality,notes
    1,5.234,7.891,flex,good,clean signal
    2,12.456,15.123,extend,good,
    3,20.001,22.567,rest,bad,motion artifact

----

Gesture Control Pipeline GUI
------------------------------

Real-time gesture recognition interface with live feedback and control outputs.

.. code-block:: bash

    # Launch gesture pipeline
    python examples/applications/gesture_pipeline_gui.py

**Features:**

- **Live prediction display**: Shows current gesture with confidence
- **Prediction history**: Scrolling timeline of past predictions
- **Performance metrics**: Accuracy, latency, prediction rate
- **Model management**: Load/switch between models on-the-fly
- **Control outputs**: Send commands to external devices
- **Training mode**: Collect labeled data for model improvement

**Configuration:**

Uses a profile JSON file for settings:

.. code-block:: json

    {
        "model_path": "/path/to/model.keras",
        "pca_path": "/path/to/pca_model.pkl",
        "norm_path": "/path/to/norm_params.npz",
        "device": {
            "num_channels": 128,
            "sample_rate": 4000,
            "buffer_duration": 1.0
        },
        "prediction": {
            "window_ms": 250,
            "step_sec": 0.1,
            "confidence_threshold": 0.7
        },
        "control": {
            "enabled": true,
            "output_type": "serial",
            "port": "COM3",
            "baud_rate": 115200
        }
    }

**Launch with profile:**

.. code-block:: bash

    python gesture_pipeline_gui.py --profile gesture_pipeline_profile.json

**Programmatic usage:**

.. code-block:: python

    """
    Custom gesture pipeline with GUI
    """
    from intan.applications import GesturePipelineGUI
    from intan.interface import IntanRHXDevice
    from intan.ml import ModelManager
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Initialize components
    device = IntanRHXDevice(num_channels=128)
    model_manager = ModelManager.load_from_files(
        model_path='model.keras',
        pca_path='pca_model.pkl',
        norm_path='norm_params.npz'
    )

    # Create GUI
    gui = GesturePipelineGUI(
        device=device,
        model_manager=model_manager,
        label_names=['rest', 'flex', 'extend', 'pinch'],
        update_interval_ms=100
    )

    # Optional: Add control callback
    def on_gesture_detected(gesture, confidence):
        print(f"Detected: {gesture} ({confidence:.2%})")
        # Send command to robot, game, etc.

    gui.gesture_detected.connect(on_gesture_detected)

    gui.show()
    device.start_streaming()

    sys.exit(app.exec_())

----

Application Launcher
---------------------

Unified launcher for all GUI applications.

.. code-block:: python

    """
    Launch application selector
    Script: examples/applications/_launcher.py
    """
    from intan.applications import launch_application_selector
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    launch_application_selector()
    sys.exit(app.exec_())

Presents a menu to select which application to run:

- EMG Viewer
- Trial Selector
- Gesture Pipeline
- Signal Quality Monitor
- Impedance Checker

----

Customizing Applications
--------------------------

All GUI applications are built with extensibility in mind:

**Custom widget integration:**

.. code-block:: python

    from intan.applications import EMGViewer
    from PyQt5.QtWidgets import QPushButton

    class CustomEMGViewer(EMGViewer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Add custom button
            self.custom_button = QPushButton("Custom Action")
            self.custom_button.clicked.connect(self.custom_action)
            self.toolbar.addWidget(self.custom_button)

        def custom_action(self):
            # Implement custom functionality
            current_data = self.get_current_data()
            # Process data...

**Custom plot overlays:**

.. code-block:: python

    from intan.plotting import StackedPlot
    import pyqtgraph as pg

    class AnnotatedPlot(StackedPlot):
        def add_event_marker(self, time, label):
            line = pg.InfiniteLine(
                pos=time,
                angle=90,
                pen=pg.mkPen('r', width=2),
                label=label
            )
            self.plot_widget.addItem(line)

----

Performance Optimization
-------------------------

For smooth real-time performance:

1. **Use downsampling for display:**

   .. code-block:: python

       viewer.set_display_downsample_factor(2)  # Display every 2nd sample

2. **Limit visible channels:**

   .. code-block:: python

       viewer.set_visible_channels(range(32))  # Show only 32 channels

3. **Adjust update rate:**

   .. code-block:: python

       viewer.set_update_interval(50)  # Update every 50ms

4. **Use OpenGL rendering:**

   .. code-block:: python

       import pyqtgraph as pg
       pg.setConfigOption('useOpenGL', True)
       pg.setConfigOption('enableExperimental', True)

----

See Also
---------

- :doc:`live_plotting` - Real-time plotting techniques
- :doc:`gesture_classification` - ML pipeline for gesture control
- :doc:`../info/applications` - Application architecture details
- API Reference: :mod:`intan.applications`
