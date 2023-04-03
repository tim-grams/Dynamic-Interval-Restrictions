Project Structure
=================

The code in this GitHub repository is structured as follows:

:configs:
    Configuration files including hyperparameter settings, search spaces, and environment parameters.

:notebooks:
    Notebooks for debugging, preprocessing, hyperparameter search, and evaluation.

:src/wrapper:
    Environment wrapper classes for continuous masking, random replacement, projection, discretization, and parameterized discretization.

:src/utils:
    Utility functions, for example to load an agent.

:src/evaluation:
    Evaluation and visualization functionality.

:src/envs:
    Obstacle avoidance, oil extraction, and fuel saving environments.

:src/algorithms:
    Rllib class modifications for MPS-TD3, PAM, and modifications to the exploration for dynamic interval action spaces.
    Note that independent ddpg was the previous name for MPS-TD3. Unfortunately, it was not possible to rename the class without running all experiments again.

:src/run.py:
    Main functions to train and test agents.
