Gesture classification
======================

The maintained gesture workflow has three scripts. Run them from the repository root and install the machine-learning extra first:

.. code-block:: bash

   python -m pip install -e '.[ml]'
   python examples/gesture_classifier/1_build_dataset.py --help
   python examples/gesture_classifier/2_train_model.py --help
   python examples/gesture_classifier/3_predict.py --help

Dataset construction
--------------------

The builder accepts RHD, NPZ, and CSV inputs and requires event labels for supervised training. Example:

.. code-block:: bash

   python examples/gesture_classifier/1_build_dataset.py \
       --root_dir /path/to/project \
       --file_type rhd \
       --file_path /path/to/recording.rhd \
       --events_file /path/to/recording.event \
       --label demo

Training
--------

.. code-block:: bash

   python examples/gesture_classifier/2_train_model.py \
       --root_dir /path/to/project \
       --train_npz /path/to/project/demo_training_dataset.npz \
       --label demo

Training uses PyTorch. Parameters and channel order are saved with the model so prediction can validate input dimensions.

Prediction
----------

The unified predictor has ``file``, ``batch``, ``record``, and ``stream`` subcommands:

.. code-block:: bash

   python examples/gesture_classifier/3_predict.py file --help
   python examples/gesture_classifier/3_predict.py batch --help
   python examples/gesture_classifier/3_predict.py record --help
   python examples/gesture_classifier/3_predict.py stream --help

``record`` and ``stream`` need an RHX TCP server. ``--use_lsl`` additionally needs an available LSL runtime. The repository does not include training recordings or trained models.

See ``examples/gesture_classifier/README.md`` for the complete parameter-oriented workflow.
