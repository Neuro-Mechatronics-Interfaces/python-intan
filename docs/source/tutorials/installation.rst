Installation
===================

Here we will setup your environment to best run all tutorials.

We can setup a virtual environment using conda or venv. This is the recommended way to run the tutorials.

If you are using conda, run the following command in your terminal:

.. code-block:: bash

    conda create -n intan python=3.10
    conda activate intan

If you are using **venv**, run the following command in your terminal:

.. code-block:: bash

    python -m venv intan
    source intan/bin/activate

We can install the package from PyPI:

.. code-block:: bash

    pip install python-intan

Or, install the latest version from GitHub:

.. code-block:: bash

    git clone https://github.com/Neuro-Mechatronics-Interfaces/python-intan
    cd python-intan
    pip install -e .

Now that we have the package installed, we can start using it.
The package is designed to be easy to use and understand. Check out the tutorials below to get started.