.. raw:: html

    <h2 align="center">Dynamic interval restrictions on action spaces<br>in deep reinforcement learning for obstacle avoidance</h2>

    <div align="center">
        <img src="https://img.shields.io/github/repo-size/timg339/master_thesis" />
        <a href="https://www.youtube.com/@dynamicintervalrestrictions/playlists"><img src="https://img.shields.io/youtube/channel/views/UCcdVlXWTKPiX43u1vrNx5gQ?style=social"></a>
    </div>

    <div align="center">
        <img src="https://i.ibb.co/1qW6R2X/environment-overview.png" />
    </div>

    <p align="center">A full documentation can be found <a href="https://action-space-thesis.readthedocs.io/">here</a>.</p>
    <p align="center">Videos of the evaluation rollouts are uploaded to <a href="https://www.youtube.com/@dynamicintervalrestrictions/playlists">YouTube</a></p>

Abstract
########
    Deep reinforcement learning algorithms typically act on the same set of actions.
    However, this is not sufficient for a wide range of real-world applications where
    different subsets are available at each step. In this thesis, we consider the problem of
    interval restrictions as they occur in obstacle avoidance. When actions that lead to
    collisions are avoided, the action space is split into different parts. Recent research
    is limited to mostly convex subsets and is not able to find an optimal policy over
    varying disjoint ranges. Therefore, we propose two approaches, namely PAM and
    MPS-TD3, that extend parameterized reinforcement learning and ConstraintNet
    to handle an arbitrary number of intervals. We demonstrate their performance in
    an obstacle avoidance task and compare the approaches to penalties, projection,
    replacement, as well as discrete and continuous masking from the literature. The
    results suggest that discrete masking of action values is the only effective method
    when constraints did not emerge in training episodes. Contrarily, the decision
    between MPS-TD3, projection, and masking seems to depend on the task at hand
    when restrictions are learned. We compare the results with varying complexity
    and give directions for future work.

Quickstart
##########

Adhere to the following steps:

1. Start by cloning this GitHub repository

.. code-block::

    $  git clone https://github.com/timg339/master_thesis.git
    $  cd master_thesis

2. Install the dependencies

.. code-block::

    $  pip install -r requirements.txt

3. Optionally train an agent or skip this step and use the provided converged algorithms

.. code-block::

    $ python3 cli.py train --environment obstacle_avoidance-random --algorithm PPO

4. Evaluate the approach without exploration

.. code-block::

    $ python3 cli.py evaluate

5. Explore the results in the corresponding notebook

.. code-block::

    $ jupyter notebook

License
#######

This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
