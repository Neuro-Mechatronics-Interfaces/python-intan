Load Files
========================

In this tutorial, we will learn how to load Intan data files using the `io` module.

.. code-block:: python

    from intan import io

    # Load a rhd file
    result = io.load_rhd_file('path/to/file.rhd')

We can also leave the file path empty which will open a file dialog to select the file.

We can also get some data from the file.

.. code-block:: python

    emg_data = result['amplifier_data']                           # Shape: (num_channels, num_samples)
    fs = result['frequency_parameters']['amplifier_sample_rate']  # Sampling frequency
    t_s = result['t_amplifier']                                   # t_amplifier contains the time vector

    analog_data = result.get('board_adc_data']                    # Shape: (num_channels, num_samples)
    fs_analog = result['frequency_parameters']['board_adc_sample_rate']  # Sampling frequency
    t_analog = result['t_board_adc']                             # t_board_adc contains the time vector

    ch_info = result['amplifier_channels']                             # Channel info

We can check out what channels are included in the recording:

.. code-block:: python

    io.print_all_channel_names(result)


For now, the `.dat` files are loaded by passing a directory or leaving the path open. It assumes that an `info.rhd` file exists in the directory:

.. code-block:: python

    result = io.load_dat_file('path/to/directory')
    # or
    result = io.load_dat_file()

It should contain the same structure as the `.rhd` files. The only difference is that the data is stored in a binary format. The data is loaded into memory as a numpy array. The data is not loaded into memory until it is accessed. This means that you can load large files without running out of memory.
