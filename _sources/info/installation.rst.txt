Installation
============

``python-intan`` requires Python 3.10 or later. A virtual environment is recommended.

.. code-block:: bash

   python -m venv .venv
   # Linux/macOS: source .venv/bin/activate
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install python-intan

Optional features are grouped so a base installation does not download GUI, video, or machine-learning stacks:

.. code-block:: bash

   python -m pip install 'python-intan[gui]'
   python -m pip install 'python-intan[ml]'
   python -m pip install 'python-intan[video]'

The ``video`` extra provides OpenCV and MediaPipe. The separate ``handtrack`` project used by one finger-kinematics example is not installed automatically.

Development checkout
--------------------

.. code-block:: bash

   git clone https://github.com/Neuro-Mechatronics-Interfaces/python-intan.git
   cd python-intan
   python -m pip install -e '.[test]'
   python -m pytest

Examples should be launched from the repository root. Hardware and LSL examples additionally require a running device, RHX TCP server, or LSL stream.
