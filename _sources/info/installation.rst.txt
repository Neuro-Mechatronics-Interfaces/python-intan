Installation
============

``python-intan`` requires Python 3.10 or later. A virtual environment is recommended.

.. code-block:: bash

   python -m venv .venv
   # Linux/macOS: source .venv/bin/activate
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install python-intan

The base installation supports RHD/DAT/NPZ I/O, RHX/LSL streaming, and signal processing.
CSV/event tables, plotting, and scikit-learn are optional:

.. code-block:: bash

   python -m pip install 'python-intan[data]'      # pandas tables
   python -m pip install 'python-intan[plot]'      # Matplotlib
   python -m pip install 'python-intan[analysis]'  # data, plots, scikit-learn PCA
   python -m pip install 'python-intan[gui]'
   python -m pip install 'python-intan[ml]'
   python -m pip install 'python-intan[video]'

The ``video`` extra provides OpenCV and MediaPipe. The separate ``handtrack`` project used by one finger-kinematics example is not installed automatically.

Use ``python-intan[analysis]`` to retain the analysis dependencies previously
installed by default. The ``gui`` and ``ml`` extras include these dependencies.
Documentation sources and media remain in the repository; build the documentation
from a checkout rather than a package source archive.

Development checkout
--------------------

.. code-block:: bash

   git clone https://github.com/Neuro-Mechatronics-Interfaces/python-intan.git
   cd python-intan
   python -m pip install -e '.[test]'
   python -m pytest

Examples should be launched from the repository root. Hardware and LSL examples additionally require a running device, RHX TCP server, or LSL stream.
