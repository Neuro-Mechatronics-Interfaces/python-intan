File Loading and Visualization Example
=========================================

This example demonstrates how to load EMG data from an Intan `.rhd` file and visualize it using the plotting utilities in the `intan` package.

**Requirements:**
- An Intan `.rhd` data file (can be selected via file dialog)
- `matplotlib` for plotting

You’ll see how to:
- Load a file interactively
- Print available channel names
- Plot a multi-channel "waterfall" EMG visualization
- Plot individual channels by name and index
- Plot analog input channels

----

**Example code:**

.. code-block:: python

    """
    Example: Loading and visualizing EMG data from an Intan .rhd file.
    """

    from intan.io import load_rhd_file, print_all_channel_names
    from intan.plotting import waterfall, plot_channel_by_index, plot_channel_by_name, plot_figure

    if __name__ == "__main__":
        # ==== Load the data (file dialog opens) ====
        result = load_rhd_file()

        # ==== Get some data from the file ====
        emg_data = result.get('amplifier_data')           # Shape: (num_channels, num_samples)
        fs = result['frequency_parameters']['amplifier_sample_rate']
        t_s = result.get('t_amplifier')                   # Time vector
        analog_data = result.get('board_adc_data')        # Analog inputs (if present)

        # ==== Print all available channel names ====
        print_all_channel_names(result)

        # ==== Multi-channel visualization: Waterfall plot ====
        waterfall(emg_data, range(64, 128), t_s, plot_title='Intan EMG data (Channels 64-127)')

        # ==== Single channel visualization ====
        ch_info = result.get('amplifier_channels')
        ch_name = ch_info[4].get('native_channel_name') if ch_info else None
        if ch_name:
            plot_channel_by_name(ch_name, result)       # By name
        plot_channel_by_index(8, result)                # By index

        # ==== Plot the first analog channel, if available ====
        if analog_data is not None:
            plot_figure(analog_data[0, :], t_s, 'Analog data (Channel 0)')

----

**Expected outcome:**

- A window will open to select your `.rhd` data file.
- Channel names are printed in the terminal.
- Plots of EMG and analog signals will be displayed for visual inspection.

**Tip:** You can modify the code to load specific files or to plot different channels. See the API Reference for all available plotting utilities.

