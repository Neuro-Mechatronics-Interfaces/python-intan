import sys
import argparse
from PyQt5.QtWidgets import QApplication
from intan.plotting import StackedPlot
from intan.interface import LSLClient

def parse_channel_args(channels_arg, default_channels=[0, 1, 2, 3]):
    """
    Parses the --channels argument from the command line.
    Allows for:
      --channels all
      --channels 0 1 2 3
      --channels 0:64
    """
    print(f"Received argument: {channels_arg}")
    if len(channels_arg) == 1 and channels_arg[0].lower() == "all":
        return "all"
    elif len(channels_arg) == 1 and ":" in channels_arg[0]:
        # Support slice format, e.g. --channels 0:64
        start, end = map(int, channels_arg[0].split(":"))
        return list(range(start, end))
    else:
        try:
            return list(map(int, channels_arg))
        except ValueError:
            print("[Warning] Invalid --channels argument. Using default:", default_channels)
            return default_channels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch real-time EMG stacked plot from LSL stream.")
    parser.add_argument("--channels", nargs="+", default=["0", "1", "2", "3"],
                        help="Channels to plot: e.g., --channels 0 1 2 or --channels 0:64 or --channels all")
    parser.add_argument("--stream-type", type=str, default="EMG", help="LSL stream type to look for (default: EMG)")
    args = parser.parse_args()

    # Parse channel selection
    channels_to_plot = parse_channel_args(args.channels)
    print(f"Channels to plot: {channels_to_plot}")

    # Launch the Qt Application
    app = QApplication(sys.argv)

    client = LSLClient(stream_type=args.stream_type, auto_start=True)
    #client.start_streaming()

    # Create and launch the stacked plotter
    plotter = StackedPlot(client=client)
    #plotter.start()
    plotter.show()
    sys.exit(app.exec_())
