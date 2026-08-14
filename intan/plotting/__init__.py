from ._waterfall import (
    waterfall,
    add_scalebars,
    insert_channel_labels,
    insert_vertical_labels,
)
from ._realtime_plotter import RealtimePlotter, run_realtime_plot


def __getattr__(name):
    if name == "StackedPlot":
        try:
            from ._stacked_plot import StackedPlot
        except ImportError as exc:
            raise ImportError(
                "StackedPlot requires the GUI extra; install 'python-intan[gui]'"
            ) from exc
        return StackedPlot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "waterfall",
    "add_scalebars",
    "insert_channel_labels",
    "insert_vertical_labels",
    "RealtimePlotter",
    "run_realtime_plot",
    "StackedPlot",
]
