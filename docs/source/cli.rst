Command-Line Interface
======================

The command line interface is the central point to train agents and run evaluation rollouts.
Just make sure that the training and configuration is properly set in the config-file before running the commands.

Training
^^^^^^^^

.. code-block::

    python3 cli.py train [OPTIONS]

--algorithm     Specify the algorithm to train. Options are PPO, PPO-MASKED, TD3, DQN, DQN-MASKED, MPS-TD3, and PAM.
--environment   Specify the environment to use. Options are obstacle_avoidance, oil_extraction, and fuel_saving. Add one of -random, -masking, -euclidean, -discretization, -p_discretization for random replacement, continuous masking, euclidean projection, discretization, and parameterized discretization
--hpo           Flag to indicate whether that hyperparameter optimization should be performed.

For example, run the following command with the standard configuration to train PPO with random
replacement without dynamic obstacles:

.. code-block::

    python3 cli.py train --algorithm PPO --environment obstacle_avoidance-random

Keep in mind that the training can take some time (up to 72 hours).
PPO is generally faster than the off-policy algorithms.

Alternatively, the already converged models, which were used in the experiments of this thesis, can be used in the evaluation.

Evaluation
^^^^^^^^^^

.. code-block::

    python3 cli.py evaluate

For example, run the above command with the standard configuration to reproduce the results with the pretrained
MPS-TD3 agent and complex restrictions (14 obstacles):

Afterward, you can load the step- and episode-results files in the evaluation notebook to visualize the outcomes.
