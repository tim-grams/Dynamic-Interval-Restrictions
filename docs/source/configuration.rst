Configuration
=============

The configuration of this project differentiates training and evaluation.
Both are described by the `training_config <https://github.com/timg339/Dynamic-Interval-Restrictions/blob/master/configs/base_config.py>`__ and `evaluation_config <https://github.com/timg339/Dynamic-Interval-Restrictions/blob/master/configs/base_config.py>`__ in `configs/base_config <https://github.com/timg339/Dynamic-Interval-Restrictions/blob/master/configs/base_config.py>`__.

.. raw:: html

    <div align="center", style="margin-bottom: 15px;">
        <img src="https://i.ibb.co/VYJLWqY/Unbenanntes-Diagramm-drawio-1.png" />
    </div>

The training configuration contains the hyperparameters and the learning environment.
Every converged agent is contained in the experiments.pkl which must be specified in the evaluation configuration.


Training Configuration
^^^^^^^^

:NAME:
    Name of the experiment.
:HYPERPARAMETERS:
    Dictionary with the algorithm configuration.
:ALGORITHM_NAME:
    Name of the algorithm.
:ENV_CONFIG:
    Dictionary of the environment parameters.
:TRAINING_ITERATIONS:
    Number of training iterations.
:LOCAL_DIR:
    Dictionary to save the models and result files.
:NUM_GPUS:
    Number of GPU's to use for training.
:NUM_WORKERS:
    Number of Rllib workers.
:NUM_SAMPLES:
    Number of configuration trials. Only relevant for hyperparameter optimization.
:SEEDS:
    List of seeds to train.
:VERBOSE:
    Log information level.
:NUM_DISCRETIZATION_BINS:
    Number of discretized actions or bins. Only relevant for PAM and the discretization wrapper.
:HPO_RESULTS_PATH:
    Path to store a dataframe with the results for each trial during the hyperparameter optimization.
:TRAINING_RESULTS_PATH:
    Path to store the final models.
:CONFIGURATION_PATH:
    Path to store a dictionary of the training configuration for reproducability.
:EXPERIMENTS_PATH:
    Path to store a summary of all trained models describing the location of the corresponding config and model.


Evaluation Configuration
^^^^^^^^

:ALGORITHM_NAME:
    Name of the algorithm to evaluate. The agent will be looked up in a experiments file and the corresponding config and model loaded.
:ENV_CONFIG:
    Dictionary of the environment parameters.
:SEEDS:
    List of seeds to evaluate.
:OBSTACLES:
    List with the numbers of obstacles to evaluate.
:EXPERIMENTS_PATH:
    Path to an existing experiments file which contains the algorithm to evaluate.
:EPISODE_RESULTS_PATH:
    Path to store the results summarizing entire episodes.
:STEP_DATA_PATH:
    Path to store the results for describing every time step of each episode.
:RECORD:
    Path to store videos of the rollouts. None if videos should not be saved.
