Evaluation
==========

The results of the thesis can be obtained by optionally training an agent with the described configuration
and then running the evaluation runs.

.. raw:: html

    <div align="center", style="margin-bottom: 15px;">
        <img src="https://i.ibb.co/VHLWP8g/Unbenanntes-Diagramm-drawio-4.png" />
    </div>

1. **Training**. First configure the configuration files and then use the command line interface to start learning.
Optionally, this step can be skipped and the already converged agents loaded. Note that the training can take up to 72 hours.
The process creates four types of data:

    - *results/models/<agent>* - This is the checkpoint of the Rllib model to load and reuse the agent.
    - *results/configurations/<agent>.pkl* - The configuration as a dictionary with which the agent was trained
    - *results/experiments.pkl* - Dataframe containing the path to the Rllib checkpoint and configuration for each algorithm.
    - *results/training_results.pkl* - The dataframe contains metrics and other information about each training iteration.

Afterward, evaluation runs can be started with agents described in the experiments file.

2. **Evaluation**. Make sure that the configuration points to the recently created experiments.pkl and run the rollouts.
The evaluation creates three types of data:

    - *results/complex_evaluation/step_results.pkl* - The dataframe contains metrics for each time step.
    - *results/complex_evaluation/episode_results.pkl* - The dataframe describes metrics over entire episodes.

The step and episode results can afterward be loaded in the evaluation jupyter notebook to analyze the outcomes.

We provide all results of the thesis in the corresponding files in the results dictionary.
The dataframes can also be directly loaded and analyzed in the notebooks.
