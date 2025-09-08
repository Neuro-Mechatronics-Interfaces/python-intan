"""
intan.applications._launcher

This module creates a centered splash screen displaying the NML logo, waits briefly,
and then transitions into the main EMGViewer GUI or EMGTrialSelector. It ensures proper Tkinter initialization
and clean teardown of the splash before launching the main interface.

Main Functions:
- `center_window`: Utility to center any Tk window on screen
- `launch_main`: Destroys the splash and launches EMGViewer
"""
import tkinter as tk
from PIL import Image, ImageTk
from intan.samples import findFile


def launch_emg_viewer():
    def center_window(window, width, height):
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        window.geometry(f"{width}x{height}+{x}+{y}")

    def launch_main():
        splash.destroy()  # Close the splash screen

        from intan.applications import EMGViewer
        main_root = tk.Tk()
        main_root.title("EMG Viewer")
        app = EMGViewer(main_root)
        main_root.mainloop()

    # === Splash screen ===
    splash = tk.Tk()
    splash.overrideredirect(True)  # Remove window decorations

    image_path = findFile("nml-logo.jpg")  # Make sure this path is correct
    img = Image.open(image_path)
    img = img.resize((300, 300), Image.Resampling.LANCZOS)
    tk_img = ImageTk.PhotoImage(img)

    center_window(splash, img.width, img.height)  # Center the window

    # Display image
    label = tk.Label(splash, image=tk_img)
    label.pack()

    # Show splash for 1000ms, then launch main app
    splash.after(1000, launch_main)
    splash.mainloop()


def launch_emg_trial_selector():
    """
    Launch the EMG trial selector GUI.
    """
    from intan.applications import EMGTrialSelector
    root = tk.Tk()
    app = EMGTrialSelector(root)
    root.mainloop()
