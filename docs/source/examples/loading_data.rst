Loading and Parsing Data Files
===============================

This section covers various methods for loading EMG and electrophysiology data from Intan files. The `intan.io` module supports multiple file formats including `.rhd`, `.dat`, `.csv`, and `.npz` files.

----

Loading .rhd Files
-------------------

Intan's native recording format. Contains EMG/neural data along with metadata.

**Basic loading (with file dialog):**

.. code-block:: python

    """
    Example: Load a single .rhd file using file picker
    Script: examples/Read_Files/load_rhd_demo.py
    """
    from intan.io import load_rhd_file, print_all_channel_names

    # Opens file dialog to select file
    result = load_rhd_file()

    # Extract data
    emg_data = result['amplifier_data']          # Shape: (channels, samples)
    fs = result['frequency_parameters']['amplifier_sample_rate']
    t_amplifier = result['t_amplifier']          # Time vector
    channels = result['amplifier_channels']      # Channel metadata

    # Print available channels
    print_all_channel_names(result)

**Loading with specific path:**

.. code-block:: python

    from intan.io import load_rhd_file

    # Direct path (no dialog)
    result = load_rhd_file('/path/to/data.rhd')

**Loading multiple .rhd files:**

.. code-block:: python

    """
    Load multiple .rhd files and optionally concatenate them
    """
    from intan.io import load_rhd_files

    # Option 1: Load as list of separate recordings
    results = load_rhd_files(concatenate=False)  # Opens dialog for multi-select

    # Option 2: Concatenate into single recording
    combined_result = load_rhd_files(concatenate=True)

**Parsing raw .rhd data:**

.. code-block:: python

    """
    Low-level parsing of .rhd file structure
    Script: examples/Read_Files/parse_rhd_data.py
    """
    from intan.io import parse_rhd_header, read_rhd_blocks

    # Parse header information
    with open('data.rhd', 'rb') as fid:
        header = parse_rhd_header(fid)
        print(f"Sampling rate: {header['sample_rate']} Hz")
        print(f"Number of channels: {len(header['amplifier_channels'])}")

        # Read data blocks
        blocks = read_rhd_blocks(fid, header)

----

Loading .dat Files
-------------------

Intan's legacy binary format. Typically used with older RHD2000 systems.

.. code-block:: python

    """
    Load .dat files from a directory
    Script: examples/Read_Files/load_dat_demo.py
    """
    from intan.io import load_dat_file

    # Load all .dat files from directory
    dat_folder = "/path/to/dat/files"
    result = load_dat_file(dat_folder)

    # Access data (same structure as .rhd)
    emg_data = result['amplifier_data']
    fs = result['frequency_parameters']['amplifier_sample_rate']

**Parsing .dat with custom parameters:**

.. code-block:: python

    """
    Parse .dat file with specific configuration
    Script: examples/Read_Files/parse_dat_data.py
    """
    from intan.io import parse_dat_file

    result = parse_dat_file(
        folder_path="/path/to/dat",
        num_channels=128,
        sample_rate=20000,
        dtype='int16'
    )

----

Loading .csv Files
-------------------

Load preprocessed EMG data from CSV format.

.. code-block:: python

    """
    Load EMG data from CSV files
    Script: examples/Read_Files/load_csv_demo.py
    """
    from intan.io import load_csv_data
    import pandas as pd

    # Simple CSV loading
    data = pd.read_csv('emg_data.csv')

    # Using intan's CSV loader (supports metadata)
    result = load_csv_data(
        csv_path='emg_data.csv',
        sample_rate=4000,
        channel_names=None  # Auto-detect from CSV header
    )

    emg_data = result['data']  # Shape: (channels, samples)
    channel_names = result['channel_names']

----

Segmenting Data by Events
---------------------------

Extract specific time windows around labeled events (e.g., gesture onsets).

.. code-block:: python

    """
    Segment EMG data based on event markers
    Script: examples/Read_Files/segment_emg_from_events.py
    """
    from intan.io import load_rhd_file, load_labeled_file
    from intan.processing import extract_segments

    # Load data
    result = load_rhd_file('session.rhd')
    emg_data = result['amplifier_data']
    fs = result['frequency_parameters']['amplifier_sample_rate']

    # Load event labels (from notes.txt or labels.csv)
    labels_df = load_labeled_file('notes.txt')

    # Extract segments around events
    segments = extract_segments(
        emg_data,
        labels_df,
        fs,
        window_before=0.5,  # 500ms before event
        window_after=1.5,   # 1500ms after event
    )

    # segments is a dict: {label: list_of_arrays}
    for gesture_name, trials in segments.items():
        print(f"{gesture_name}: {len(trials)} trials")
        for trial in trials:
            print(f"  Shape: {trial.shape}")

**Event file format (notes.txt):**

::

    0.0, rest, start
    5.2, flex, gesture_onset
    6.5, rest, gesture_end
    10.1, extend, gesture_onset
    11.3, rest, gesture_end

----

Working with Channel Information
----------------------------------

.. code-block:: python

    """
    Access and filter channel metadata
    """
    from intan.io import load_rhd_file, get_channel_names

    result = load_rhd_file('data.rhd')

    # Get all channel names
    channel_names = get_channel_names(result)
    print(f"Available channels: {channel_names}")

    # Filter by channel type
    amp_channels = [ch for ch in result['amplifier_channels']
                    if ch['port_name'] == 'A']

    # Get specific channel by name
    channel_index = channel_names.index('A-015')
    channel_data = result['amplifier_data'][channel_index, :]

----

Data Structure Reference
-------------------------

All loading functions return a dictionary with the following structure:

.. code-block:: python

    result = {
        'amplifier_data': np.ndarray,           # Shape: (channels, samples)
        'amplifier_channels': list,             # Channel metadata
        'frequency_parameters': dict,           # Sampling rates
        't_amplifier': np.ndarray,              # Time vector (seconds)
        'board_adc_data': np.ndarray,           # Analog input data (optional)
        'board_dig_in_data': np.ndarray,        # Digital input data (optional)
        'notes': dict,                          # Recording notes (optional)
    }

**Channel metadata structure:**

.. code-block:: python

    channel = {
        'native_channel_name': 'A-015',
        'custom_channel_name': 'Biceps_R',
        'native_order': 15,
        'board_stream': 0,
        'chip_channel': 15,
        'port_name': 'A',
        'port_prefix': 'A',
        'port_number': 1,
        'electrode_impedance_magnitude': 50000.0,
        'electrode_impedance_phase': -5.2,
    }

----

Performance Tips
-----------------

1. **Large files**: Use memory-mapped loading for files >1GB:

   .. code-block:: python

       result = load_rhd_file('large_file.rhd', mmap_mode='r')

2. **Selective channel loading**: Load only specific channels to reduce memory:

   .. code-block:: python

       result = load_rhd_file('data.rhd', channels=[0, 1, 2, 3])

3. **Batch processing**: Use generators for processing multiple files:

   .. code-block:: python

       from intan.io import rhd_file_generator

       for result in rhd_file_generator('/path/to/files'):
           # Process each file
           process_data(result['amplifier_data'])

----

See Also
---------

- :doc:`file_loading` - Visualization examples
- :doc:`../info/load_files` - Detailed API documentation
- API Reference: :mod:`intan.io`
