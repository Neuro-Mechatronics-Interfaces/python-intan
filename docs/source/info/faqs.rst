Frequently asked questions
==========================

Which Python versions are supported?
------------------------------------

Python 3.10 and later are declared in the package metadata. The release workflow validates the versions listed by the PyPI classifiers.

Which Intan files can I load?
-----------------------------

The package reads RHD recordings and Intan per-signal ``.dat`` recording directories. CSV and NPZ helpers support derived datasets. RHS parsing is not currently advertised.

What array orientation is expected?
-----------------------------------

Signal-processing functions generally expect ``(channels, samples)``. Check individual docstrings when integrating data from libraries that use ``(samples, channels)``.

Why does a hardware example fail to connect?
--------------------------------------------

Confirm that Intan RHX software is running, its TCP server is enabled, the configured command and waveform ports are reachable, and a firewall is not blocking them. Pico and arm examples also require their matching firmware and peripherals.

Are GUI and ML libraries installed by default?
-----------------------------------------------

No. Install ``python-intan[gui]`` for the supported GUI applications and ``python-intan[ml]`` for PyTorch workflows.

Does the video extra install ``handtrack``?
--------------------------------------------

No. The video extra installs OpenCV and MediaPipe. ``handtrack`` is a separate project used by one optional example and must be evaluated and installed independently.

Where are the current examples?
-------------------------------

Start with ``examples/Read_Files``, ``examples/RHXDevice``, and ``examples/gesture_classifier``. Run scripts from the repository root and use ``--help`` on scripts that expose a command-line interface.

How do I report a problem?
--------------------------

Open an issue at https://github.com/Neuro-Mechatronics-Interfaces/python-intan/issues and include the package version, Python version, operating system, traceback, and a minimal reproduction that does not contain sensitive data.
