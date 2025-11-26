Frequently Asked Questions (FAQs)
==================================

.. contents:: Quick Links
   :local:
   :depth: 2

----

Installation & Setup
--------------------

Where can I find Anaconda?
~~~~~~~~~~~~~~~~~~~~~~~~~~

Anaconda can be downloaded from the official website at https://www.anaconda.com/products/distribution. It is available for Windows, macOS, and Linux.

How do I install Anaconda?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Anaconda can be installed by downloading the installer from the official website and following the installation instructions provided there. The installation process is straightforward and typically involves running the downloaded installer and following the prompts.

Alternatively, you can use Miniconda (a minimal version): https://docs.conda.io/en/latest/miniconda.html

How do I install the python-intan package?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**From PyPI (recommended):**

.. code-block:: bash

    pip install python-intan

**From source (for development):**

.. code-block:: bash

    git clone https://github.com/Neuro-Mechatronics-Interfaces/python-intan.git
    cd python-intan
    pip install -e .

What Python version is required?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python 3.10 or higher is required. We recommend Python 3.10 or 3.11 for best compatibility with TensorFlow and other dependencies.

How do I enable GPU support for TensorFlow?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**For NVIDIA GPUs on Windows/WSL2:**

1. Install NVIDIA drivers for your GPU
2. Install CUDA toolkit:

   .. code-block:: bash

       # WSL2
       sudo apt-get install cuda

       # Or via pip
       pip install tensorflow[and-cuda] nvidia-cudnn-cu12

3. Verify GPU availability:

   .. code-block:: python

       import tensorflow as tf
       print(tf.config.list_physical_devices('GPU'))

See the :doc:`../examples/gesture_classification` guide for detailed GPU setup instructions.

----

File Loading & Data Access
---------------------------

What file formats are supported?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package supports:

- ``.rhd`` - Intan's native recording format (recommended)
- ``.rhs`` - Intan stimulation recording format
- ``.dat`` - Legacy Intan binary format
- ``.csv`` - Comma-separated values (for preprocessed data)
- ``.npz`` - NumPy compressed arrays (for processed datasets)

See :doc:`../examples/loading_data` for examples.

How do I load a .rhd file?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from intan.io import load_rhd_file

    # With file picker dialog
    result = load_rhd_file()

    # Or specify path directly
    result = load_rhd_file('/path/to/file.rhd')

    # Access data
    emg_data = result['amplifier_data']  # Shape: (channels, samples)
    fs = result['frequency_parameters']['amplifier_sample_rate']
    t = result['t_amplifier']  # Time vector

Can I load multiple files at once?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! Use ``load_rhd_files()`` with the ``concatenate`` option:

.. code-block:: python

    from intan.io import load_rhd_files

    # Load as separate files
    results_list = load_rhd_files(concatenate=False)

    # Or concatenate into single recording
    combined = load_rhd_files(concatenate=True)

How do I access specific channels?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from intan.io import load_rhd_file, get_channel_names

    result = load_rhd_file('data.rhd')

    # Get all channel names
    channel_names = get_channel_names(result)
    print(channel_names)  # ['A-000', 'A-001', ...]

    # Access by index
    channel_5_data = result['amplifier_data'][5, :]

    # Access by name
    channel_idx = channel_names.index('A-015')
    channel_data = result['amplifier_data'][channel_idx, :]

Why is my file loading slowly?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Large files (>1GB) can take time to load. Try:

1. **Memory-mapped loading** (doesn't load entire file into RAM):

   .. code-block:: python

       result = load_rhd_file('large_file.rhd', mmap_mode='r')

2. **Load only specific channels**:

   .. code-block:: python

       result = load_rhd_file('data.rhd', channels=[0, 1, 2, 3])

3. **Use a faster disk** (SSD vs HDD makes a big difference)

----

Device & Hardware
-----------------

Is LSL Supported?
~~~~~~~~~~~~~~~~~

**Yes!** Lab Streaming Layer (LSL) is fully supported as of version 0.0.2. You can:

- Publish Intan EMG data to LSL streams
- Subscribe to external LSL marker streams
- Synchronize recording with other LSL-compatible applications

See :doc:`../examples/lsl_streaming` for complete examples.

How do I connect to an Intan RHX device?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from intan.interface import IntanRHXDevice

    # Connect (defaults to localhost:5000/5001)
    device = IntanRHXDevice()

    # Enable channels
    device.enable_wide_channel(range(64))  # Enable first 64 channels

    # Start streaming
    device.start_streaming()

    # Get data
    timestamps, data = device.stream(duration_sec=1.0)

    # Cleanup
    device.close()

**Important:** Make sure the RHX software is running with TCP servers enabled (Remote Control → Network).

My device won't connect. What should I check?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Is RHX software running?** The Intan RHX software must be open
2. **Are TCP servers enabled?** In RHX: Network → Enable TCP Servers
3. **Check ports:** Default is 5000 (command) and 5001 (data). Verify in RHX settings
4. **Firewall:** Ensure Windows Firewall isn't blocking the connection
5. **Try manual connection:**

   .. code-block:: python

       device = IntanRHXDevice(
           host='127.0.0.1',
           command_port=5000,
           data_port=5001
       )
       print(f"Connected: {device.connected}")

What hardware is compatible?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package works with:

- **Intan RHX Recording Controllers** (RHD USB Interface Board, RHD Recording Controller)
- **RHD2000 Series Amplifiers** (RHD2132, RHD2164, RHD2216, etc.)
- **RHS2000 Series Stimulation/Recording Amplifiers**
- **Raspberry Pi Pico** (for peripheral control and IMU integration)
- **Lab Streaming Layer** compatible devices

Can I use this without Intan hardware?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! You can:

- Load and analyze existing ``.rhd`` files without hardware
- Process EMG data from other sources (load via ``.csv`` or ``.npz``)
- Use the signal processing and ML modules independently
- Develop and test with sample data

----

Data Processing
---------------

How do I filter my EMG data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from intan.processing import filter_emg, notch_filter

    # Remove 60Hz powerline noise
    emg_notched = notch_filter(emg_data, fs, f0=60)

    # Bandpass filter (10-500 Hz typical for EMG)
    emg_filtered = filter_emg(emg_notched, 'bandpass', fs,
                               lowcut=10, highcut=500)

    # High-pass to remove DC offset
    emg_hp = filter_emg(emg_data, 'highpass', fs, lowcut=20)

What preprocessing steps are recommended for EMG?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Typical pipeline:

1. **Notch filter** (remove 50/60 Hz powerline)
2. **Bandpass filter** (10-500 Hz for EMG)
3. **Rectification** (optional, for envelope)
4. **Normalization** (for ML applications)

.. code-block:: python

    from intan.processing import (
        notch_filter, filter_emg, rectify, normalize
    )

    # 1. Remove powerline noise
    emg = notch_filter(raw_emg, fs, f0=60)

    # 2. Bandpass filter
    emg = filter_emg(emg, 'bandpass', fs, lowcut=20, highcut=500)

    # 3. Rectify (for envelope/RMS)
    emg_rect = rectify(emg)

    # 4. Normalize (for ML)
    emg_norm = normalize(emg, method='zscore')

How do I calculate RMS?
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from intan.processing import window_rms

    # 100ms RMS window
    window_samples = int(0.1 * fs)
    rms_emg = window_rms(emg_data, window_size=window_samples)

How do I downsample my data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from intan.processing import downsample

    # Downsample by factor of 4 (e.g., 20kHz → 5kHz)
    emg_downsampled = downsample(emg_data, factor=4)

----

Machine Learning
----------------

How do I train a gesture classifier?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the complete guide: :doc:`../examples/gesture_classification`

Quick summary:

1. **Prepare training data** from ``.rhd`` files
2. **Train model** with PCA + neural network
3. **Real-time prediction** from streaming device

.. code-block:: bash

    # 1. Build dataset
    python examples/gesture_classifier/1a_build_training_dataset_rhd.py

    # 2. Train model
    python examples/gesture_classifier/2_train_model.py

    # 3. Predict in real-time
    python examples/gesture_classifier/3d_predict_from_device_realtime.py

What ML models are supported?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **CNN** (Convolutional Neural Network) - Default, best for EMG
- **LSTM** (Long Short-Term Memory) - For temporal patterns
- **Dense** (Fully-connected) - Simple baseline
- **Custom models** - Bring your own Keras model

How much training data do I need?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Minimum:** 5-10 trials per gesture (not recommended)

**Good:** 20-30 trials per gesture

**Better:** 50+ trials per gesture across multiple sessions

**Best practices:**

- Collect data across different arm positions
- Include multiple subjects if building a general model
- Balance classes (equal trials per gesture)
- Include "rest" as a class

My model accuracy is low. How can I improve it?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Data quality:**
1. Check signal quality - are channels noisy?
2. Ensure consistent gesture execution
3. Add more training trials
4. Balance your dataset

**Preprocessing:**
1. Verify filtering parameters (10-500 Hz bandpass)
2. Try different normalization methods
3. Remove bad channels (high impedance/noise)

**Model tuning:**
1. Increase PCA components (try 20-50)
2. Add more training epochs
3. Try data augmentation
4. Use transfer learning

**Feature engineering:**
1. Add time-domain features (RMS, MAV)
2. Include frequency features
3. Use different window sizes (200-300ms)

----

Visualization
-------------

How do I plot my data?
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from intan.plotting import waterfall, plot_channel_by_index

    # Multi-channel waterfall plot
    waterfall(emg_data, channels=range(64), time=t,
              plot_title='EMG Activity')

    # Single channel
    plot_channel_by_index(5, result)

How do I create real-time plots?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :doc:`../examples/live_plotting` for detailed examples.

.. code-block:: python

    from intan.interface import IntanRHXDevice
    from intan.plotting import RealtimePlotter

    device = IntanRHXDevice(num_channels=64)
    device.start_streaming()

    plotter = RealtimePlotter(n_channels=64, sample_rate=4000)
    plotter.start()

    # Update loop
    while True:
        _, data = device.stream(n_frames=200)
        plotter.update(data)

Can I use the GUI applications?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! See :doc:`../examples/gui_applications`:

.. code-block:: bash

    # EMG Viewer
    python examples/applications/run_emg_viewer.py

    # Trial Selector
    python examples/applications/run_trial_selector.py

    # Gesture Pipeline
    python examples/applications/gesture_pipeline_gui.py

----

Troubleshooting
---------------

ImportError: No module named 'intan'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution:**

1. Ensure package is installed:

   .. code-block:: bash

       pip install python-intan

2. Check you're in the correct environment:

   .. code-block:: bash

       conda activate intan  # or your env name
       python -c "import intan; print(intan.__version__)"

3. If developing, install in editable mode:

   .. code-block:: bash

       pip install -e .

ModuleNotFoundError: No module named 'tensorflow'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install TensorFlow:

.. code-block:: bash

    pip install tensorflow==2.19.0

For GPU support:

.. code-block:: bash

    pip install tensorflow[and-cuda] nvidia-cudnn-cu12

"Could not connect to RHX TCP server" error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Checklist:**

1. ✅ Is Intan RHX software running?
2. ✅ Are TCP servers enabled? (Network → Enable TCP Servers)
3. ✅ Are you using correct ports? (default: 5000, 5001)
4. ✅ Is firewall blocking connection?
5. ✅ Try connecting from RHX software's "Remote Control" panel first

Memory error when loading large files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solutions:**

1. Use memory-mapped mode:

   .. code-block:: python

       result = load_rhd_file('file.rhd', mmap_mode='r')

2. Load fewer channels
3. Process file in chunks
4. Increase system RAM or use machine with more memory

PyQt5 errors in GUI applications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install PyQt5:

.. code-block:: bash

    pip install PyQt5 pyqtgraph

If issues persist on Linux:

.. code-block:: bash

    sudo apt-get install python3-pyqt5

Sampling rate mismatch warnings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This usually means:

1. Expected sample rate doesn't match file/device
2. Solution: Let the package auto-detect:

   .. code-block:: python

       device = IntanRHXDevice()  # Auto-detects sample rate
       fs = device.get_sample_rate()

Channels appear empty or zero
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Check:**

1. Are channels enabled in RHX software?
2. Are amplifiers connected and powered?
3. Did you enable wide/high/low channels?

   .. code-block:: python

       device.enable_wide_channel(range(64))

4. Check impedance values (should be <1 MΩ)

----

Performance Optimization
------------------------

How can I speed up real-time processing?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use fewer channels** - Only enable what you need
2. **Downsample display** - Update plot every 50ms, not every sample
3. **Use PyQtGraph** instead of Matplotlib for GUI
4. **Enable OpenGL**:

   .. code-block:: python

       import pyqtgraph as pg
       pg.setConfigOption('useOpenGL', True)

5. **Process in separate thread** - Don't block data acquisition

How do I optimize for low latency?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Reduce buffer size**:

   .. code-block:: python

       device = IntanRHXDevice(buffer_duration_sec=0.5)

2. **Use smaller prediction windows** (100-200ms)
3. **Optimize model** - Use smaller networks
4. **GPU acceleration** for inference
5. **Adjust TCP block size**:

   .. code-block:: python

       device.set_blocks_per_write(1)

----

Platform-Specific
-----------------

Does this work on macOS?
~~~~~~~~~~~~~~~~~~~~~~~~~

**Partial support.** The package works on macOS for:

- ✅ File loading and processing
- ✅ Machine learning
- ✅ LSL integration
- ⚠️ RHX TCP streaming (if RHX software runs on Mac)
- ❌ Some hardware interfaces (Pico serial)

Best tested on Windows and Linux (WSL2).

Does this work on Linux?
~~~~~~~~~~~~~~~~~~~~~~~~~

**Yes!** Fully supported, especially Ubuntu/Debian-based distributions.

For GUI applications, install Qt:

.. code-block:: bash

    sudo apt-get install python3-pyqt5 python3-pyqt5.qtsvg

Can I use this in WSL2?
~~~~~~~~~~~~~~~~~~~~~~~

**Yes!** WSL2 is recommended for GPU support on Windows.

**Benefits:**

- Native Linux environment
- Better CUDA support
- Faster file I/O in some cases

**Limitations:**

- Serial port access requires special configuration
- GUI applications need X server (WSLg or VcXsrv)

----

Contributing & Support
----------------------

How can I contribute?
~~~~~~~~~~~~~~~~~~~~~

We welcome contributions! See the `contribution guide <https://github.com/Neuro-Mechatronics-Interfaces/python-intan/blob/main/CONTRIBUTING.md>`_.

You can contribute by:

- Reporting bugs
- Submitting pull requests
- Adding examples
- Improving documentation
- Sharing your use cases

Where can I get help?
~~~~~~~~~~~~~~~~~~~~~

1. **Documentation:** https://neuro-mechatronics-interfaces.github.io/python-intan/
2. **GitHub Issues:** https://github.com/Neuro-Mechatronics-Interfaces/python-intan/issues
3. **Examples:** Check ``examples/`` directory in the repository

How do I report a bug?
~~~~~~~~~~~~~~~~~~~~~~

Open an issue on GitHub with:

1. Python version and OS
2. Package version (``import intan; print(intan.__version__)``)
3. Minimal code to reproduce the issue
4. Full error traceback
5. Expected vs actual behavior

Is this package maintained?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! Active development by the Neuromechatronics Lab at Carnegie Mellon University.

**Version history:**

- v0.0.3 (2025-01) - Expanded examples, improved ML pipeline, LSL support
- v0.0.2 (2024-12) - Added GUI applications, real-time prediction
- v0.0.1 (2024-05) - Initial release

----

Citation & License
------------------

How do I cite this package?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bibtex

    @software{Shulgach_Python_Intan_2025,
      author = {Shulgach, Jonathan, Murphy, Max and Foy, Adrian},
      title = {{Python Intan Package}},
      year = {2025},
      month = {05},
      version = {0.0.3},
      url = {https://github.com/Neuro-Mechatronics-Interfaces/python-intan},
      note = "{\tt jshulgac@andrew.cmu.edu}"
    }

What license is this under?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MIT License - Free for academic and commercial use. See LICENSE file for details.

----

Still have questions?
---------------------

Check the :doc:`index` or open an issue on `GitHub <https://github.com/Neuro-Mechatronics-Interfaces/python-intan/issues>`_.
