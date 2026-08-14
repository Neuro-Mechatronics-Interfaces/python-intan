Real-time prediction
====================

Real-time prediction uses a trained PyTorch model plus the metadata written by the training workflow. Install the machine-learning extra and inspect stream-mode help:

.. code-block:: bash

   python -m pip install -e '.[ml]'
   python examples/gesture_classifier/3_predict.py stream --help

With Intan RHX software running and its TCP server enabled:

.. code-block:: bash

   python examples/gesture_classifier/3_predict.py stream \
       --root_dir /path/to/project \
       --label demo \
       --infer_hz 20 \
       --smooth_k 5

Add ``--use_lsl`` to publish predictions over Lab Streaming Layer. The repository does not provide a trained model, so complete the dataset and training steps first. Channel order and feature dimensions must match the saved training metadata.
