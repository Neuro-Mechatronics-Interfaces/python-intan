GUI applications
================

Install the GUI extra before launching the packaged applications:

.. code-block:: bash

   python -m pip install 'python-intan[gui]'
   intan-emg-viewer --help
   intan-trial-selector --help

EMG viewer
----------

.. code-block:: bash

   intan-emg-viewer

The viewer opens local EMG files and provides multichannel visualization. Use the maintained gesture-classifier scripts for PyTorch model training and prediction.

Trial selector
--------------

.. code-block:: bash

   intan-trial-selector

The selector supports interactive inspection and event/trial marking. It requires a desktop display.

Repository workflow GUIs
------------------------

.. code-block:: bash

   python examples/applications/dataset_builder_gui.py
   python examples/applications/gesture_pipeline_gui.py

The pipeline GUI invokes the maintained scripts in ``examples/gesture_classifier``. It does not bundle data, models, RHX hardware, or LSL streams. See ``examples/applications/README.md`` for its limitations.
