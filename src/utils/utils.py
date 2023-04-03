import os
from decimal import Decimal

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray.rllib.algorithms.ppo
from ray.tune import register_env
from ray.rllib.algorithms.registry import ALGORITHMS
from ray.tune.registry import get_trainable_cls
from shapely.geometry import LineString
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from src.wrapper.hierarchical import HierarchicalWrapper

MULTI_GEOM_TYPES = ['MultiPolygon', 'MultiLineString', 'GeometryCollection', 'MultiPoint']
NO_EXTERIOR_TYPES = ['Point', 'LineString']

SHAPE_COLLECTION = [
    # Rectangle
    np.array([[0, 0], [0, 1], [1, 1], [1, 0]]),
    # Trapeze
    np.array([[0.0, 0.0], [0.33, 1.0], [0.66, 1.0], [1.0, 0.0]]),
    # Triangle
    np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]]),
    # Octagon
    np.array([[0.0, 0.66], [0.33, 1.0], [0.66, 1.0], [1.0, 0.66], [1.0, 0.33], [0.66, 0.0], [0.33, 0.0], [0.0, 0.33]])
]


def register_environments(bins: int):
    """ Register the possible environments with Rllib.
        Warehouse is in to support the old naming of the obstacle avoidance task

    Args:
        bins (int): Number of discretized actions or bins
    """
    from src.envs.fuel_saving import FuelSaving
    from src.envs.oil_extraction import OilField
    from src.envs.obstacle_avoidance import ObstacleAvoidance

    from src.wrapper.api_translation import APITranslationWrapper
    from src.wrapper.continuous_masking import ContinuousMaskingWrapper
    from src.wrapper.discretization import DiscretizationWrapper
    from src.wrapper.euclidean_projection import EuclideanProjectionWrapper
    from src.wrapper.random_replacement import RandomReplacementWrapper
    from src.wrapper.parametrized_discretization import ParametrizedDiscretizationWrapper

    register_env('fuel_saving', lambda conf: APITranslationWrapper(FuelSaving(conf)))
    register_env('oil_extraction', lambda conf: APITranslationWrapper(OilField(conf)))
    register_env('obstacle_avoidance', lambda conf: APITranslationWrapper(ObstacleAvoidance(conf)))
    register_env('warehouse', lambda conf: APITranslationWrapper(ObstacleAvoidance(conf)))

    register_env('fuel_saving-masking', lambda conf: ContinuousMaskingWrapper(FuelSaving(conf)))
    register_env('oil_extraction-masking', lambda conf: ContinuousMaskingWrapper(OilField(conf)))
    register_env('obstacle_avoidance-masking', lambda conf: ContinuousMaskingWrapper(ObstacleAvoidance(conf)))
    register_env('warehouse-masking', lambda conf: ContinuousMaskingWrapper(ObstacleAvoidance(conf)))

    register_env('fuel-saving-discretization', lambda conf: DiscretizationWrapper(FuelSaving(conf), bins))
    register_env('oil_extraction-discretization', lambda conf: DiscretizationWrapper(OilField(conf), bins))
    register_env('obstacle_avoidance-discretization', lambda conf: DiscretizationWrapper(ObstacleAvoidance(conf), bins))
    register_env('warehouse-discretization', lambda conf: DiscretizationWrapper(ObstacleAvoidance(conf), bins))

    register_env('fuel-saving-p_discretization', lambda conf: ParametrizedDiscretizationWrapper(FuelSaving(conf), bins))
    register_env('oil_extraction-p_discretization', lambda conf: ParametrizedDiscretizationWrapper(OilField(conf), bins))
    register_env('obstacle_avoidance-p_discretization', lambda conf: ParametrizedDiscretizationWrapper(
        ObstacleAvoidance(conf), bins))
    register_env('warehouse-p_discretization', lambda conf: ParametrizedDiscretizationWrapper(
        ObstacleAvoidance(conf), bins))

    register_env('fuel-saving-euclidean', lambda conf: EuclideanProjectionWrapper(FuelSaving(conf)))
    register_env('oil_extraction-euclidean', lambda conf: EuclideanProjectionWrapper(OilField(conf)))
    register_env('obstacle_avoidance-euclidean', lambda conf: EuclideanProjectionWrapper(ObstacleAvoidance(conf)))
    register_env('warehouse-euclidean', lambda conf: EuclideanProjectionWrapper(ObstacleAvoidance(conf)))

    register_env('fuel-saving-random', lambda conf: RandomReplacementWrapper(FuelSaving(conf)))
    register_env('oil_extraction-random', lambda conf: RandomReplacementWrapper(OilField(conf)))
    register_env('obstacle_avoidance-random', lambda conf: RandomReplacementWrapper(ObstacleAvoidance(conf)))
    register_env('warehouse-random', lambda conf: RandomReplacementWrapper(ObstacleAvoidance(conf)))

    register_env('fuel-saving-hierarchical', lambda conf: HierarchicalWrapper(FuelSaving(conf)))
    register_env('oil_extraction-hierarchical', lambda conf: HierarchicalWrapper(OilField(conf)))
    register_env('obstacle_avoidance-hierarchical', lambda conf: HierarchicalWrapper(ObstacleAvoidance(conf)))
    register_env('warehouse-hierarchical', lambda conf: HierarchicalWrapper(ObstacleAvoidance(conf)))


def get_algorithm(algorithm: str):
    """ Returns the algorithm class for its name

    Args:
        algorithm (str): Name of the algorithm

    Returns:
        algorithm (Algorithm): Rllib algorithm object
    """
    if algorithm in ALGORITHMS:
        return get_trainable_cls(algorithm)

    from src.algorithms.independent_ddpg import IndependentDDPG
    from src.algorithms.ddpg_action_replacement import ReplacementDDPG
    from src.algorithms.masked_dqn import MaskedDQN
    from src.algorithms.parametric_masked_dqn import ParametricMaskedDQN

    if algorithm in ['MPS-TD3', 'INDEPENDENT-DDPG']:
        return IndependentDDPG
    if algorithm == 'DQN-MASKED':
        return MaskedDQN
    if algorithm == 'PPO-MASKED':
        return ray.rllib.algorithms.ppo.PPO
    if algorithm in ['PAM', 'PARAMETRIC-MASKED']:
        return ParametricMaskedDQN
    if algorithm == 'DDPG-REPLACEMENT':
        return ReplacementDDPG


def line_intersection(c1, c2, n1, n2, agent_x1, agent_y1, agent_x2, agent_y2):
    """ Determines how the first line intersects with the second.
        For example, if the second line is crossed from the bottom up.
        Used to see if a restriction goes beyond the possible range of actions ([-180,180]).

    Args:
        c1 (Decimal): x-coordinate of the first line's starting position
        c2 (Decimal): y-coordinate of the first line's starting position
        n1 (Decimal): x-coordinate of the first line's closing position
        n2 (Decimal): y-coordinate of the first line's closing position
        agent_x1 (Decimal): x-coordinate of the agent's line starting position
        agent_y1 (Decimal): y-coordinate of the agent's line starting position
        agent_x2 (Decimal): x-coordinate of the agent's line closing position
        agent_y2 (Decimal): y-coordinate of the agent's line closing position

    Returns:
        intersection_type (str): The first part indicates the start and the second the end of the first line with respect to the agent's line. For example, negative_positive
    """
    intersection = len(
        np.array(LineString([
            (float(agent_x1), float(agent_y1)),
            (float(agent_x2), float(agent_y2))
        ]).intersection(LineString([(c1, c2), (n1, n2)])).coords)) > 0
    if intersection and c2 > agent_y1 > n2:
        return 'negative_positive'
    if intersection and c2 < agent_y1 < n2:
        return 'positive_negative'
    if intersection and c2 == agent_y1 and n2 > agent_y1:
        return 'positive_line'
    if intersection and c2 == agent_y1 and n2 < agent_y1:
        return 'negative_line'
    if intersection and c2 == agent_y1 and n2 == agent_y1 and c1 >= agent_x1:
        return 'line_right_out'
    if intersection and c2 == agent_y1 and n2 == agent_y1 and c1 < agent_x1:
        return 'line_line'
    if intersection and c2 < agent_y1 and n2 == agent_y1:
        return 'line_negative'
    if intersection and c2 > agent_y1 and n2 == agent_y1:
        return 'line_positive'
    return 'none'


def midpoint(coordinates):
    """ Calculates the midpoint of a polygon

    Args:
        coordinates (list): Coordinates that define the shape of the polygon
    """
    return [(max(coordinates[:, 0]) - min(coordinates[:, 0])) / 2 + min(coordinates[:, 0]),
            (max(coordinates[:, 1]) - min(coordinates[:, 1])) / 2 + min(coordinates[:, 1])]


def dict_key_maximum(dictionary: dict, key: str):
    """ Finds the maximum key for a key in different subdictionaries

    Args:
        dictionary (dict): Dictionary with sub-dictionaries
        key (str): Key of the sub-dictionary to maximize

    Returns:
        maximum (float)
    """
    current_maximum = -np.inf

    for sub_dictionary in dictionary.values():
        if sub_dictionary[key] > current_maximum:
            current_maximum = sub_dictionary[key]

    return current_maximum


def project_intervals_into_action_space(intervals, low, high):
    """ Projects action spaces that go beyond [-180, 180] back into the range

    Args:
        intervals (list): Allowed action space
        low (float): Minimum of the allowed action space (In our case -180)
        high (float): Maximum of the allowed action space (In our case 180)

    Returns:
        maximum (float)
    """
    action_space_range = high - low
    for subspace in intervals:
        if subspace[0] != Decimal(np.inf):
            if subspace[0] > high:
                subspace[0] -= action_space_range
            elif subspace[0] < low:
                subspace[0] += action_space_range
            if subspace[1] > high:
                subspace[1] -= action_space_range
            elif subspace[1] < low:
                subspace[1] += action_space_range

    return [subspace for subspace in intervals if subspace[0] != Decimal(np.inf)]


def inverse_space(space, low, high):
    """ Finds the allowed given restrictions

    Args:
        space (list): Restrictions
        low (float): Minimum of the allowed action space
        high (float): Maximum of the allowed action space

    Returns:
        allowed (list)
    """
    inverse = [[low, high]]

    for original_subspace in space:
        to_test = []
        if original_subspace[0] > original_subspace[1]:
            if not original_subspace[0] == high:
                to_test.append([original_subspace[0], high])
            if not original_subspace[1] == low:
                to_test.append([low, original_subspace[1]])
        else:
            to_test = [original_subspace]
        for subspace in to_test:
            for index, inverse_subspace in enumerate(inverse):
                if subspace[0] < inverse_subspace[0] <= subspace[1] <= inverse_subspace[1]:
                    inverse_subspace[0] = subspace[1]
                if subspace[1] > inverse_subspace[1] >= subspace[0] >= inverse_subspace[0]:
                    inverse_subspace[1] = subspace[0]
                if subspace[0] >= inverse_subspace[0] and subspace[1] <= inverse_subspace[1]:
                    if inverse_subspace[0] != subspace[0]:
                        inverse.append([inverse_subspace[0], subspace[0]])
                    if inverse_subspace[1] != subspace[1]:
                        inverse.append([subspace[1], inverse_subspace[1]])
                    inverse_subspace[0] = Decimal(np.inf)

    inverse = [not_allowed_space for not_allowed_space in inverse if
               not_allowed_space[0] != Decimal(np.inf) and not_allowed_space[0] != not_allowed_space[1]]
    return inverse


def sample_from_gaussian_process(min_value: float, max_value: float, num_samples: int = 600,
                                 kernel=RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)), seed: int = 42):
    x = np.linspace(min_value, max_value, num_samples)
    X = x.reshape(-1, 1)

    gpr = GaussianProcessRegressor(kernel=1.0 * kernel, random_state=seed)
    y = gpr.sample_y(X, 1, random_state=seed)

    return x, y


def point_from_gaussian_process(unknown: float, min_value: float, max_value: float, num_samples: int = 600,
                                kernel=RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)), seed: int = 42):
    x = np.linspace(min_value, max_value, num_samples)
    X = x.reshape(-1, 1)

    gpr = GaussianProcessRegressor(kernel=1.0 * kernel, random_state=seed)
    y = gpr.sample_y(X, 1, random_state=seed)

    return np.interp(unknown, x, y.T[0])


def plot_gaussian_process_sample(min_value: float, max_value: float, num_samples: int = 600,
                                 kernel=RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)), seed: int = 42):
    x = np.linspace(min_value, max_value, num_samples)
    X = x.reshape(-1, 1)

    gpr = GaussianProcessRegressor(kernel=1.0 * kernel, random_state=seed)
    y_samples = gpr.sample_y(X, 1, random_state=seed)

    for idx, single_prior in enumerate(y_samples.T):
        plt.plot(
            x,
            single_prior,
            linestyle="--",
            alpha=0.7,
            label=f"Sampled function",
        )


def default_location_game_function(unknown):
    return 4 * np.sin(2 * unknown - 1) + 3 * np.sin(3 * unknown - 1) + 2 * np.sin(
        4 * unknown - 1) + 4 * np.cos(5 * unknown - 1) + 15


def load_agent(config, path):
    """ Loads an agent

    Args:
        config (dict): Configuration
        path (float): Location of the Rllib checkpoint

    Returns:
        agent (Trainable)
    """
    assert 'algorithm' in config
    assert 'config' in config

    algorithm = get_algorithm(config['algorithm'])
    agent = algorithm(config=config['config'])
    agent.restore(path)

    return agent


def make_dir_if_not_exists(path: str = None):
    """ Creates directory if it does not exist

    Args:
        path (dict): Path to the directory
    """
    if path is not None and not os.path.exists(path):
        os.makedirs(path)


def save_or_extend_dataframe(dataframe, path: str):
    """ Appends data to a dataframe and saves the updated version

    Args:
        dataframe (dataframe): Dataframe to append
        path (str): Path to store
    """
    if os.path.exists(path):
        results = [pd.read_pickle(path), dataframe]
        pd.concat(results).reset_index(drop=True).to_pickle(path)
    else:
        make_dir_if_not_exists(os.path.dirname(path))
        dataframe.to_pickle(path)


def load_pickle(path: str):
    """ Loads an existing pickle object from the disk

    Args:
        path (dict): Path to the .pkl file

    Returns:
        object (any)
    """
    with open(path, 'rb') as pickle_file:
        content = pickle.load(pickle_file)

    return content
