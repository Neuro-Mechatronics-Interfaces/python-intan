"""Launch the packaged desktop applications."""
import tkinter as tk


def launch_emg_viewer():
    """Launch the PyQt viewer when available, otherwise use the Tk fallback."""
    try:
        from PyQt5 import QtWidgets
        import importlib
        mod = importlib.import_module('intan.applications._emg_viewer')
        EMGViewer = getattr(mod, 'EMGViewer', None)
        if EMGViewer is not None:
            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            viewer = EMGViewer()
            viewer.show()
            app.exec_()
            return
    except Exception:
        pass

    from intan.applications._emg_viewer import EMGViewerTk

    root = tk.Tk()
    root.title("EMG Viewer")
    EMGViewerTk(root)
    root.mainloop()


def launch_emg_trial_selector():
    """
    Launch the EMG trial selector GUI.
    """
    from intan.applications import EMGTrialSelector
    root = tk.Tk()
    app = EMGTrialSelector(root)
    root.mainloop()
