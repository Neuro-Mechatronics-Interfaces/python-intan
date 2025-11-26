Examples
=====================

Explore hands-on examples for common use-cases with the `intan` package.
Each example includes full scripts, usage notes, and expected outcomes.

Getting Started
---------------

If you're new to the package, start with these fundamental examples:

- :doc:`loading_data` - Load and parse EMG data from various file formats
- :doc:`file_loading` - Visualize data from files
- :doc:`live_plotting` - Real-time streaming visualization

Advanced Topics
---------------

.. toctree::
   :maxdepth: 2
   :caption: Data Handling

   loading_data
   file_loading
   stream_vs_file

.. toctree::
   :maxdepth: 2
   :caption: Visualization

   live_plotting
   lsl_streaming
   gui_applications

.. toctree::
   :maxdepth: 2
   :caption: Machine Learning

   gesture_classification
   realtime_predict

.. toctree::
   :maxdepth: 2
   :caption: Hardware Integration

   hardware_control

Quick Reference
---------------

**By Task:**

- **Load data from files**: :doc:`loading_data`
- **Visualize EMG signals**: :doc:`file_loading`, :doc:`live_plotting`
- **Stream from RHX device**: :doc:`live_plotting`, :doc:`stream_vs_file`
- **Use Lab Streaming Layer**: :doc:`lsl_streaming`
- **Train gesture classifiers**: :doc:`gesture_classification`
- **Real-time prediction**: :doc:`realtime_predict`
- **Control robots/hardware**: :doc:`hardware_control`
- **GUI applications**: :doc:`gui_applications`

**By Data Format:**

- `.rhd files`: :doc:`loading_data`, :doc:`file_loading`
- `.dat files`: :doc:`loading_data`
- `.csv files`: :doc:`loading_data`
- `.npz files`: :doc:`gesture_classification`
- Live TCP streaming: :doc:`live_plotting`, :doc:`stream_vs_file`

**By Hardware:**

- Intan RHX Controller: :doc:`live_plotting`, :doc:`stream_vs_file`
- Raspberry Pi Pico: :doc:`hardware_control`
- 3D-printed robotic arm: :doc:`hardware_control`
- IMU sensors: :doc:`hardware_control`
- RHD2164 wireless: :doc:`hardware_control`

Example Scripts Location
--------------------------

All example scripts can be found in the ``examples/`` directory of the repository:

.. code-block:: text

    examples/
    ├── Read_Files/              # File loading examples
    ├── RHXDevice/               # Device streaming examples
    ├── LSL/                     # Lab Streaming Layer
    ├── gesture_classifier/      # ML pipeline (training & prediction)
    ├── applications/            # GUI applications
    ├── 3D_printed_arm_control/  # Robotic control
    ├── interface/               # Hardware interfacing
    ├── intan_tcp/               # Low-level TCP examples
    └── RHD2164_wireless/        # Wireless headstage

Contributing Examples
---------------------

Have an interesting use case? We welcome example contributions! See the
`contribution guide <https://github.com/Neuro-Mechatronics-Interfaces/python-intan/blob/main/CONTRIBUTING.md>`_
for details on submitting new examples.
