Plotting EMG Data
===================

This tutorial covers the plotting functions provided by the ``intan.plotting`` module for visualizing EMG and analog data acquired from Intan devices.

You can use these plotting functions on data loaded from ``.rhd`` or ``.dat`` files using the ``intan.io`` module.

First, load your data from an ``.rhd`` file:

.. code-block:: python

    from intan.io import load_rhd_file
    result = load_rhd_file()  # Opens a file dialog to select a .rhd file

    emg_data = result['amplifier_data']                # (num_channels, num_samples)
    t_s = result['t_amplifier']

    # Optional: load analog data if present
    analog_data = result.get('board_adc_data')         # (num_channels, num_samples)
    t_analog = result.get('t_board_adc')                   # (num_samples,)

----

Plot a Single Channel by Name
-----------------------------

If you know the channel's native name (e.g., "A-005"), use ``plot_channel_by_name``:

.. code-block:: python

    from intan.plotting import plot_channel_by_name

    # Get the name of the 5th channel
    ch_name = ch_info[4]['native_channel_name']
    plot_channel_by_name(ch_name, result)

----

Plot a Single Channel by Index
----------------------------------

You can also plot by index directly:

.. code-block:: python

    from intan.plotting import plot_channel_by_index

    plot_channel_by_index(8, result)   # Plots channel index 8

----

Plot Any Signal with plot_figure
----------------------------------

The ``plot_figure`` function is a generic plotter for any 1D signal, including EMG or analog input:

.. code-block:: python

    from intan.plotting import plot_figure

    # Plot the first EMG channel
    plot_figure(emg_data[0], t_s, "EMG Channel 0", x_label="Time (s)", y_label="Amplitude (µV)")

    # If analog data is present, plot the first analog channel
    if analog_data is not None:
        plot_figure(analog_data[0], t_s, "Analog Channel 0", x_label="Time (s)", y_label="Voltage (V)")

----


Waterfall Plot (Multi-channel Visualization)
----------------------------------------------

To visualize activity across many EMG channels simultaneously, use ``waterfall``:

.. code-block:: python

    from intan.plotting import waterfall

    # Plot channels 64 to 127 in a waterfall plot
    waterfall(emg_data, range(64, 128), t_s, plot_title='Intan EMG data')

----


Summary
-------------------

These functions make it easy to visualize both raw and processed EMG/analog signals from Intan recordings.
You can quickly inspect data quality, channel names, and signal characteristics for further analysis.

For more advanced usage (such as real-time streaming plots), see the corresponding tutorials.

