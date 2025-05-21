"""
Waterfall Plotting Module
"""
import matplotlib.pyplot as plt
import numpy as np


def waterfall(ax=None, data=None, channel_indices=None, time_vector=None, offset_increment=200, edges=[], plot_title="",
              line_width=0.2, colormap='rainbow', downsampling_factor=1, verbose=False):
    """
    Creates a waterfall plot for the specified channels with a user-defined title,
    custom color styling, and scale bars for time and voltage.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to plot on. If None, a new figure and axes are created.
        data (numpy.ndarray): 2D array of shape (num_channels, num_samples).
        channel_indices (list): List of channel indices to plot.
        time_vector (numpy.ndarray): 1D array of time values corresponding to the samples.
        offset_increment (float): Increment for offsetting each channel's plot vertically.
        edges (list): List of edges to plot as vertical lines.
        plot_title (str): Title for the plot.
        line_width (float): Width of the lines in the plot.
        colormap (str): Colormap for the plot.
        downsampling_factor (int): Factor by which to downsample the data for plotting.
        verbose (bool): If True, prints additional information.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 12))

    offset = 0  # Start with no offset
    cmap = plt.get_cmap(colormap)  # You can also experiment with other color maps like 'jet', 'viridis', etc.
    num_channels = len(channel_indices)

    # Downsample the data by the specified factor
    data = data[:, ::downsampling_factor]
    time_vector = time_vector[::downsampling_factor]

    for i, channel_idx in enumerate(channel_indices):
        channel_data = data[channel_idx, :]
        # Use colormap to assign a color based on channel index
        color = cmap(i / num_channels)
        ax.plot(time_vector, channel_data + offset, color=color, linewidth=line_width)
        offset += offset_increment

    # If edges are provided, plot them as vertical lines
    for edge in edges:
        ax.axvline(x=edge, color='red', linestyle='--', linewidth=1)

    # Labeling and visualization
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Amplitude + Offset')
    ax.set_title(plot_title, fontsize=14, fontweight='bold')

    # Custom scale bar
    add_scalebars(ax)

    # Turn off x and y axis numbers
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove the black box around the plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Insert text labels for channel 0 and channel 128
    insert_channel_labels(ax, time_vector, num_channels)

    # Insert vertical labels for "Extensor" and "Flexor"
    insert_vertical_labels(ax)

    plt.show()


def add_scalebars(ax, scale_time=5, scale_voltage=10):
    """
    Adds time and voltage scale bars to the plot in the lower left corner.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to add the scale bars to.
        scale_time (float): Length of the timescale bar in seconds.
        scale_voltage (float): Length of the voltage scale bar in mV.

    """
    # Add horizontal time scale bar (5 sec)
    ax.plot([0, scale_time], [-1000, -1000], color='gray', lw=3)
    ax.text(scale_time / 2, -1500, '5 sec', va='center', ha='center', fontsize=12, color='gray')

    # Add vertical voltage scale bar (10 mV)
    ax.plot([0, 0], [-1000, -1000 + scale_voltage * 10], color='gray', lw=3)
    ax.text(-0.5, -500, '10 mV', va='center', ha='center', rotation='vertical', fontsize=12, color='gray')
    # ax.text(-0.5, voltage_scale_length / 2 - 1000, '10 mV', va='center', ha='center', rotation='vertical', fontsize=12, color='gray')


# def insert_channel_labels(ax, time_vector, num_channels):
#     """
#     Inserts text labels to indicate specific channels on the plot.
#
#     Args:
#         ax: The plot axes to add the text to.
#         time_vector: The time vector for placing the text appropriately.
#         num_channels: Total number of channels being plotted.
#     """
#     # Position the text for Channel 0 (near the bottom)
#     x_pos = time_vector[-1]  # Position at the end of the time range
#     ax.text(x_pos + 1, 200, 'Channel 0', fontsize=8, va='center', ha='left', color='black', fontweight='bold')
#
#     # Position the text for Channel 128 (near the top)
#     ax.text(x_pos + 1, 25500, 'Channel 128', fontsize=8, va='center', ha='left', color='black', fontweight='bold')

def insert_channel_labels(ax, time_vector, num_channels, num_labels=2, label_color='black', font_size=8):
    """
    Inserts evenly spaced text labels on a multichannel plot to indicate selected channels.

    Parameters:
        ax: Matplotlib Axes object.
        time_vector: Time vector (used for x position of labels).
        num_channels: Total number of channels plotted vertically.
        num_labels: Number of labels to insert (default is 2).
        label_color: Text color.
        font_size: Size of the text.
    """
    if num_labels > num_channels:
        raise ValueError("Number of labels cannot exceed total number of channels.")

    x_pos = time_vector[-1] + 1  # Place labels slightly past the right edge

    # Evenly space label positions from bottom to top
    channel_indices = np.linspace(0, num_channels - 1, num_labels, dtype=int)
    y_offsets = np.linspace(200, 25500, num_channels)  # Example scaling (adjust if needed)

    for ch in channel_indices:
        y = y_offsets[ch]
        ax.text(x_pos, y, f"Channel {ch}", fontsize=font_size, va='center',
                ha='left', color=label_color, fontweight='bold')


def insert_vertical_labels(ax):
    """
    Inserts vertical labels "Extensor" and "Flexor" for the two groups of channels.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to plot on. If None, a new figure and axes are created.
    """
    # Insert "Extensor" label vertically for the first 64 channels (left side)
    ax.text(-1, 5000, 'Extensor', fontsize=12, va='center', ha='center', color='black', rotation='vertical',
            fontweight='bold')

    # Insert "Flexor" label vertically for channels 65-128 (left side)
    ax.text(-1, 22000, 'Flexor', fontsize=12, va='center', ha='center', color='black', rotation='vertical',
            fontweight='bold')
