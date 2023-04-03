from abc import ABC
from typing import Optional
import random

import matplotlib.pyplot as plt
import numpy as np
from gym.error import DependencyNotInstalled
from gym.spaces import Box, Dict
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils import try_import_torch
from shapely.geometry import Point, Polygon
from decimal import *

from configs.base_config import MAX_PADDING_LEN
from src.utils.utils import line_intersection, MULTI_GEOM_TYPES, NO_EXTERIOR_TYPES, \
    project_intervals_into_action_space, inverse_space, dict_key_maximum, SHAPE_COLLECTION, midpoint

getcontext().prec = 5

torch, nn = try_import_torch()


class Agent:
    """ The agent representation

    Args:
        x (float): x-coordinate starting position
        y (float): y-coordinate starting position
        radius (float): Radius of the agent
        perspective (float): Starting perspective
        step_size (float): moving distance with each step
    """
    def __init__(self, x, y, radius, perspective, step_size):
        self.x = Decimal(repr(x))
        self.y = Decimal(repr(y))
        self.last_action = Decimal(0.0)
        self.radius = Decimal(repr(radius))
        self.perspective = Decimal(repr(perspective))
        self.step_size = Decimal(repr(step_size))
        self.collided = False
        self.distance_target = False
        self.distance_improvement = Decimal(0.0)

    def step(self, direction, dt):
        """ Take a step in a specific direction

        Args:
            direction (Decimal): Angle in which the next step should be taken
            dt (float)
        """
        self.x += Decimal(repr(np.cos(np.radians(float(direction))))) * self.step_size * dt
        self.y += Decimal(repr(np.sin(np.radians(float(direction))))) * self.step_size * dt
        self.perspective = direction

    def set_distance_target(self, new_distance):
        """ Sets the improvement and new distance to the target

        Args:
             new_distance: (float): The new distance to the target
        """
        self.distance_improvement = self.distance_target - new_distance
        self.distance_target = new_distance

    def geometric_representation(self):
        """ Returns the shapely geometry representation of the agent

        Returns:
            shapely geometry object
        """
        return Point(float(self.x), float(self.y)).buffer(float(self.radius))


class Obstacle:
    """ The obstacle representation

    Args:
        coordinates (list): Polygon coordinates for the shape of the obstacle
        step_size (float): moving distance with each step
    """
    def __init__(self, coordinates: list, step_size):
        self.coordinates = np.array([[
            Decimal(repr(coordinate[0])), Decimal(repr(coordinate[1]))
        ] for coordinate in coordinates])
        self.step_size = Decimal(repr(step_size))
        self.waypoints = [self.geometric_representation().centroid.coords[0]]
        self.distance = Decimal(0.0)
        self.x = Decimal(repr(self.waypoints[0][0]))
        self.y = Decimal(repr(self.waypoints[0][1]))
        self.current_target = 1
        self.moving_direction = 'forward'

    def add_waypoint(self, waypoint):
        """ Adds a new waypoint to which the obstacle moves

        Args:
            waypoint (list): waypoint coordinates
        """
        self.waypoints.append(waypoint)

    def step(self, dt):
        """ Takes a step in the direction of the next waypoint in the list

        Args: dt (float)
        """
        if len(self.waypoints) > 1:
            distance = Decimal(repr(Point(float(self.x), float(self.y)).distance(
                Point(self.waypoints[self.current_target][0], self.waypoints[self.current_target][1])
            )))
            if distance - self.step_size >= Decimal(0.0):
                unit_vector = [
                    (Decimal(repr(self.waypoints[self.current_target][0])) - self.x) / distance,
                    (Decimal(repr(self.waypoints[self.current_target][1])) - self.y) / distance
                ]
                step = np.array(unit_vector) * self.step_size * dt

                self.x += step[0]
                self.y += step[1]
                self.coordinates = np.add(self.coordinates, step)
            else:
                self.x = Decimal(repr(self.waypoints[self.current_target][0]))
                self.y = Decimal(repr(self.waypoints[self.current_target][1]))
                if self.current_target >= (len(self.waypoints) - 1) or self.current_target <= 0:
                    self.moving_direction = 'forward' if self.moving_direction == 'backward' else 'backward'
                else:
                    self.current_target += 1 if self.moving_direction == 'forward' else -1

    def geometric_representation(self):
        """ Returns the shapely geometry representation of the obstalce

        Returns:
            shapely geometry object
        """
        return Polygon(self.coordinates)

    def collision_area(self, radius):
        """ Returns the area which would lead to a collision when the agent enters it

        Args:
            radius: The radius of the agent

        Returns:
            shapely geometry object
        """
        return Polygon(self.coordinates).buffer(radius)


class ObstacleAvoidance(TaskSettableEnv, ABC):
    """ The obstacle avoidance environment

    Args:
        env_config (dict): Setup of the environment
        render_mode (str): Select how to visualize the environment. Options are human, rgb_array, and jupyter
    """
    metadata = {
        'render_modes': ['human', 'rgb_array', 'jupyter'],
        'render_fps': 4,
        'video.frames_per_second': 4
    }

    def __init__(self, env_config, render_mode: Optional[str] = None):
        assert 'STEPS_PER_EPISODE' in env_config
        assert 'ACTION_RANGE' in env_config
        assert 'DT' in env_config
        assert 'SAFETY_DISTANCE' in env_config
        assert 'REWARD' in env_config
        assert 'REWARD_COEFFICIENT' in env_config['REWARD']
        assert 'TIMESTEP_PENALTY_COEFFICIENT' in env_config['REWARD']
        assert 'GOAL' in env_config['REWARD']
        assert 'COLLISION' in env_config['REWARD']
        assert 'LEVELS' in env_config
        assert 1 in env_config['LEVELS']
        assert 'HEIGHT' in env_config['LEVELS'][1]
        assert 'WIDTH' in env_config['LEVELS'][1]
        assert 'AGENT' in env_config['LEVELS'][1]
        assert 'GOAL' in env_config['LEVELS'][1]

        self.STEPS_PER_EPISODE = env_config['STEPS_PER_EPISODE']
        self.MAX_LEVEL = max(env_config['LEVELS'].keys())
        self.ACTION_RANGE = Decimal(repr(env_config["ACTION_RANGE"]))
        self.SAFETY_DISTANCE = Decimal(repr(env_config["SAFETY_DISTANCE"]))
        self.MAXIMUM_HEIGHT = dict_key_maximum(env_config['LEVELS'], 'HEIGHT')
        self.MAXIMUM_WIDTH = dict_key_maximum(env_config['LEVELS'], 'WIDTH')
        self.REWARD_COEFFICIENT = Decimal(repr(env_config["REWARD"]["REWARD_COEFFICIENT"]))
        self.REWARD_GOAL = Decimal(repr(env_config["REWARD"]["GOAL"]))
        self.REWARD_COLLISION = Decimal(repr(env_config["REWARD"]["COLLISION"]))
        self.TIMESTEP_PENALTY_COEFFICIENT = Decimal(repr(env_config['REWARD']['TIMESTEP_PENALTY_COEFFICIENT']))
        self.DT = Decimal(repr(env_config["DT"]))

        self.levels = env_config['LEVELS']
        self.current_height = 0.0
        self.current_width = 0.0
        self.goal = None
        self.goal_radius = None
        self.agent = None
        self.map = None
        self.obstacles = []
        self.map_collision_area = None
        self.current_step = 0
        self.previous_position = [Decimal(0.0), Decimal(0.0)]
        self.safety_angle = Decimal(0.0)
        self.allowed_actions = []
        self.last_reward = 0.0
        self.current_level = 1
        self.trajectory = []
        self.current_seed = -1 if 'GENERATE_OBSTACLES' not in env_config['LEVELS'][1] else env_config[
            'LEVELS'][1]['GENERATE_OBSTACLES']['START_SEED']

        self.window_scale = 50
        self.window = None
        self.clock = None
        self.render_mode = render_mode
        self.reload = False

        self.load_map(self.levels[self.current_level])

        # Observation and Action Space
        self.observation_space = Dict({
            'observation': Dict({
                'location': Box(low=-2.0, high=np.max([self.MAXIMUM_WIDTH, self.MAXIMUM_HEIGHT]) + 2.0, shape=(2,),
                                dtype=np.float32),
                'perspective': Box(low=0.0, high=360.0, shape=(1,), dtype=np.float32),
                'target_angle': Box(low=0.0, high=360.0, shape=(1,), dtype=np.float32),
                'target_distance': Box(low=0.0, high=np.sqrt(self.MAXIMUM_WIDTH ** 2 + self.MAXIMUM_HEIGHT ** 2),
                                       shape=(1,), dtype=np.float32),
                'current_step': Box(low=0.0, high=self.STEPS_PER_EPISODE, shape=(1,), dtype=np.float32)
            }),
            'allowed_actions': Repeated(Box(low=-180.0, high=180.0, shape=(2,)), max_len=MAX_PADDING_LEN)
        })
        self.action_space = Box(low=float(-self.ACTION_RANGE / 2), high=float(self.ACTION_RANGE / 2), shape=(1,),
                                dtype=np.float32)

        if 'RANDOM_SEED' in env_config:
            self.seed(env_config['RANDOM_SEED'])

    def load_map(self, structure: dict):
        """ Loads the current level of the map.

        Args:
            structure (dict): Setup of the environment
        """
        self.current_height = structure['HEIGHT']
        self.current_width = structure['WIDTH']
        self.goal_radius = structure['GOAL']['radius']
        self.goal = Point(structure['GOAL']['x'], structure['GOAL']['y']).buffer(self.goal_radius)
        self.agent = Agent(x=structure['AGENT']['x'], y=structure['AGENT']['y'], radius=structure['AGENT']['radius'],
                           perspective=structure['AGENT']['angle'], step_size=structure['AGENT']['step_size'])
        self.safety_angle = Decimal(2.0) * Decimal(
            repr(np.rad2deg(np.arcsin(float((self.SAFETY_DISTANCE / Decimal(2.0)) / self.agent.radius)))))
        self.map = Polygon([(0.0, 0.0), (self.current_width, 0.0), (self.current_width, self.current_height),
                            (0.0, self.current_height)])
        self.map_collision_area = self.map.exterior.buffer(self.agent.radius)

        obstacles = None
        if 'GENERATE_OBSTACLES' in structure:
            obstacle_properties = structure['GENERATE_OBSTACLES']
            if self.current_seed < 10e7:
                self.current_seed += 1
            else:
                self.current_seed = obstacle_properties['START_SEED']

            obstacles = {
                **generate_obstacles(
                    self.current_width, self.current_height, obstacle_properties['COUNT'],
                    obstacle_properties['POSITION_COVARIANCE'], obstacle_properties['MEAN_SIZE'],
                    obstacle_properties['VARIANCE_SIZE'], obstacle_properties['RANGE_SIZE'],
                    obstacle_properties['WAYPOINTS'], obstacle_properties['DISTANCE_WAYPOINTS'],
                    obstacle_properties['VARIANCE_DISTANCE'], obstacle_properties['STEP_SIZE'],
                    self.current_seed,
                    obstacles=structure['OBSTACLES'].copy(),
                    forbidden_circles=[(self.agent.x, self.agent.y, float(self.agent.radius+self.agent.step_size)),
                                       (structure['GOAL']['x'], structure['GOAL']['y'], self.goal_radius)])
            }

        if 'OBSTACLES' in structure:
            if obstacles is None:
                obstacles = structure['OBSTACLES'].copy()
            for key, obstacle_structure in obstacles.items():
                obstacle = Obstacle(coordinates=obstacle_structure['coordinates'],
                                    step_size=obstacle_structure['step_size'])
                for waypoint in obstacle_structure['waypoints']:
                    obstacle.add_waypoint(waypoint)
                self.obstacles.append(obstacle)

    def angle_to_target(self):
        """ Calculates the angle between the agent and the goal

        Returns:
            angle_to_target (float)
        """
        if Decimal(repr(self.goal.centroid.coords[0][0])) == self.agent.x:
            angle_agent = Decimal(0.0)
        else:
            angle_agent = self.agent.perspective - Decimal(
                repr((
                    np.rad2deg(np.arctan(float(np.abs(
                        Decimal(repr(self.goal.centroid.coords[0][1])) - self.agent.y
                    ) / np.abs(Decimal(repr(self.goal.centroid.coords[0][0])) - self.agent.x)))))))

        return angle_agent if angle_agent >= Decimal(0.0) else Decimal(360.0) + angle_agent

    def distance_to_target(self):
        """ Calculates the distance between the agent and the goal

        Returns:
            distance_to_target (float)
        """
        return Decimal(repr(self.goal.centroid.distance(self.agent.geometric_representation())))

    def detect_collision(self):
        """ Checks if the agent violated any of the restrictions

        Returns:
            violation (bool)
        """
        # Check if allowed actions got violated
        if not np.any([action_interval[0] <= self.agent.last_action <= action_interval[1] for action_interval in
                       self.allowed_actions]):
            return True

        # Check if agent is on the map and not collided with the boundaries
        if not self.map.contains(self.agent.geometric_representation()) or self.agent.radius - Decimal(
                repr(self.map.exterior.distance(Point(self.agent.x, self.agent.y)))) > Decimal(0.0):
            return True

        # Check if agent collided with one of the obstacles
        if np.any([
            obstacle.geometric_representation().distance(self.agent.geometric_representation()
                                                         ) < Decimal(0.0) for obstacle in self.obstacles]):
            return True
        return False

    def get_reward(self):
        """ Calculates the reward based on collisions, improvement and the distance to the goal

        Returns:
            reward (float)
        """
        if self.agent.collided:
            reward = self.REWARD_COLLISION
        elif self.agent.distance_target <= self.goal_radius:
            reward = self.REWARD_GOAL
        else:
            reward = self.REWARD_COEFFICIENT * self.agent.distance_improvement - (
                    Decimal(repr(self.current_step)) * self.TIMESTEP_PENALTY_COEFFICIENT)
        return float(reward)

    def seed(self, seed: int = None):
        """ Set the seed of the environment

        Args:
            seed (int)
        """
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.action_space.seed(seed)

    def step(self, action):
        """ Perform an environment iteration including moving the agent and obstacles.

        Args:
            action (list): Angle of the agent's next step

        Returns:
            observation (dict)
        """
        action = Decimal(repr(action[0]))
        step_direction = self.agent.perspective + action

        if step_direction < Decimal(0.0):
            step_direction += Decimal(360.0)
        elif step_direction >= Decimal(360.0):
            step_direction -= Decimal(360.0)

        self.agent.step(step_direction, self.DT)
        self.agent.last_action = action
        self.agent.collided = self.detect_collision()
        self.agent.set_distance_target(self.distance_to_target())
        self.last_reward = self.get_reward()

        for moving_obstacle in self.obstacles:
            moving_obstacle.step(self.DT)
        self.allowed_actions = self.get_allowed_actions()

        self.trajectory.append([float(self.agent.x),
                                float(self.agent.y)])
        self.current_step += 1
        observation = {'location': np.array([self.agent.x, self.agent.y], dtype=np.float32),
                       'perspective': np.array([self.agent.perspective], dtype=np.float32),
                       'target_angle': np.array([self.angle_to_target()], dtype=np.float32),
                       'target_distance': np.array([self.agent.distance_target], dtype=np.float32),
                       'current_step': np.array([self.current_step], dtype=np.float32)}
        done = self.agent.collided or (self.agent.distance_target <= self.goal_radius)
        truncated = self.current_step >= self.STEPS_PER_EPISODE
        info = {
            'goal_distance': float(self.agent.distance_target),
            'solved': self.agent.distance_target <= Decimal(f'{self.goal_radius}'),
            'level': self.current_level
        }

        return {'observation': observation,
                'allowed_actions': np.array(self.allowed_actions, dtype=np.float32)
                }, self.last_reward, done or truncated, info

    def reset(self):
        """ Resets and loads the structure of the map again

        Returns:
            observation (dict)
        """
        self.trajectory.append([float(self.agent.x),
                                float(self.agent.y)])
        self.obstacles = []
        self.load_map(self.levels[self.current_level])
        self.agent.distance_target = self.distance_to_target()
        self.allowed_actions = self.get_allowed_actions()
        self.current_step = 0

        return {'observation': {'location': np.array([self.agent.x, self.agent.y], dtype=np.float32),
                                'perspective': np.array([self.agent.perspective], dtype=np.float32),
                                'target_angle': np.array([self.angle_to_target()], dtype=np.float32),
                                'target_distance': np.array([self.agent.distance_target], dtype=np.float32),
                                'current_step': np.array([self.current_step], dtype=np.float32)},
                'allowed_actions': np.array(self.allowed_actions, dtype=np.float32)}

    def set_task(self, task: TaskType) -> None:
        """ Sets the next environment level when the episode is reset

        Args:
            task (int): next environment level to load
        """
        if task <= self.MAX_LEVEL:
            self.current_level = task
            self.reload = True

    def get_task(self) -> TaskType:
        """ Returns the level of the environment

        Returns:
            current_level (int)
        """
        return self.current_level

    def get_restrictions_for_polygon(self, polygon_coordinates):
        """ Calculates the restriction angles for the agent and a single polygon

        Args:
            polygon_coordinates (list): List of polygon corner points that define the shape of the obstacle

        Returns:
            restrictions (list): List of intervals which would lead to a collision. For example [[-10, 30]]
        """
        max_angle = Decimal(-np.inf)
        min_angle = Decimal(np.inf)
        agent_on_action_space_boundary = self.agent.y == Decimal(repr(polygon_coordinates[0][0])
                                                                 ) if len(polygon_coordinates) > 0 else False
        boundary_crossed_negative = False
        boundary_crossed_positive = False

        for index, coordinates in enumerate(polygon_coordinates):
            coordinates = list(coordinates)
            coordinates[0] = Decimal(repr(coordinates[0]))
            coordinates[1] = Decimal(repr(coordinates[1]))

            # Check if next coordinates go beyond max and min action space boundaries.
            # For example: Coordinate 1 -> -170 and coordinate 2 -> -190 with boundary -180
            if index != 0:
                coordinate_direction_line = (coordinates[0], coordinates[1],
                                             Decimal(repr(polygon_coordinates[index - 1][0])),
                                             Decimal(repr(polygon_coordinates[index - 1][1])))
                action_space_boundary_line = (self.agent.x, self.agent.y,
                                              self.agent.x - self.agent.radius - self.agent.step_size, self.agent.y)
                line_crossed = line_intersection(*coordinate_direction_line, *action_space_boundary_line)

                if not boundary_crossed_positive and line_crossed in ['negative_positive', 'negative_line']:
                    boundary_crossed_negative = True
                elif not boundary_crossed_negative and line_crossed in ['positive_negative', 'line_negative']:
                    boundary_crossed_positive = True
                elif boundary_crossed_negative and line_crossed in ['positive_negative', 'line_negative',
                                                                    'line_right_out'
                                                                    ] and not agent_on_action_space_boundary:
                    boundary_crossed_negative = False
                elif boundary_crossed_positive and line_crossed in ['negative_positive', 'negative_line']:
                    boundary_crossed_positive = False
                if agent_on_action_space_boundary and line_crossed in ['line_positive']:
                    agent_on_action_space_boundary = False
                if agent_on_action_space_boundary and line_crossed in ['line_negative']:
                    agent_on_action_space_boundary = False

            # Angle to polygon corner
            if Decimal(coordinates[0]) == self.agent.x:
                angle_to_coordinates = Decimal(90.0)
            else:
                angle_to_coordinates = Decimal(repr(np.rad2deg(np.arctan(float(
                    np.abs(coordinates[1] - self.agent.y) / np.abs(
                        coordinates[0] - self.agent.x))))))

            # Subtract 180 if polygon corner lies left to agent
            if self.agent.x > coordinates[0]:
                angle_to_coordinates = Decimal(180.0) - angle_to_coordinates

            # Negative if polygon corner is below agent
            if self.agent.y > coordinates[1] or index == 0 and self.agent.y == coordinates[1] and index + 1 != len(
                    polygon_coordinates) and Decimal(
                repr(polygon_coordinates[index + 1][1])) < self.agent.y:
                angle_to_coordinates = -angle_to_coordinates

            # Correct if polygon corner goes beyond possible action space
            if boundary_crossed_negative and angle_to_coordinates != -180:
                angle_to_coordinates = angle_to_coordinates - Decimal(360.0)
            elif boundary_crossed_positive and angle_to_coordinates != 180:
                angle_to_coordinates = angle_to_coordinates + Decimal(360.0)

            if angle_to_coordinates > max_angle:
                max_angle = angle_to_coordinates
            if angle_to_coordinates < min_angle:
                min_angle = angle_to_coordinates

        return [min_angle - self.agent.perspective,
                max_angle - self.agent.perspective]

    def get_allowed_actions(self):
        """ Iterates through the obstacles and calculates the intervals which are allowed and do not lead to a collision

        Returns:
            allowed_actions (list): Allowed action space
        """
        step_circle = Point(float(self.agent.x), float(self.agent.y)).buffer(float(self.agent.step_size * self.DT))
        restrictions = []

        for obstacle in self.obstacles + [self.map_collision_area]:
            if isinstance(obstacle, Obstacle):
                obstacle = obstacle.collision_area(float(self.agent.radius))

            is_in_collision_area = obstacle.contains(
                Point(float(self.agent.x), float(self.agent.y))) or obstacle.boundary.contains(
                Point(float(self.agent.x), float(self.agent.y)))

            obstacle_step_circle_intersection = step_circle.intersection(obstacle) if not is_in_collision_area else (
                step_circle.boundary.difference(obstacle))

            # If intersection consists of multiple parts, iterate through them
            if obstacle_step_circle_intersection.geom_type in MULTI_GEOM_TYPES:
                restrictions_for_part = []

                for polygon in obstacle_step_circle_intersection.geoms:
                    restriction = self.get_restrictions_for_polygon(
                        polygon.exterior.coords if not is_in_collision_area and not (
                                polygon.geom_type in NO_EXTERIOR_TYPES) else polygon.coords)

                    restrictions_for_part.append(restriction)

                # Bring each restriction into the action space
                restrictions_for_part = project_intervals_into_action_space(restrictions_for_part,
                                                                            low=Decimal(-180), high=Decimal(180))
                for restriction in restrictions_for_part:
                    if restriction[0] < Decimal(-180.0):
                        restrictions_for_part.append([Decimal(-180.0), restriction[1]])
                        restriction[0] = Decimal(360) + restriction[0]
                        restriction[1] = Decimal(180)

                # Merge overlapping restrictions for different parts
                if len(restrictions_for_part) > 1:
                    for index, restriction in enumerate(restrictions_for_part):
                        if index != (len(restrictions_for_part) - 1):
                            if restriction[1] == restrictions_for_part[index + 1][0]:
                                restrictions_for_part[index + 1][0] = restriction[0]
                                restriction[0] = Decimal(np.inf)
                    restrictions_for_part = [res for res in restrictions_for_part if res[0] != Decimal(np.inf)]

                    # When agent is inside the collision area, inverse the space to get restrictions
                    if is_in_collision_area:
                        restrictions_for_part = inverse_space(restrictions_for_part,
                                                              low=Decimal(-180.0), high=Decimal(180.0))
                else:
                    restrictions_for_part = [np.flip(restrictions_for_part[0])
                                             ] if is_in_collision_area else restrictions_for_part

                restrictions += restrictions_for_part
            else:
                object_restrictions = self.get_restrictions_for_polygon(
                    obstacle_step_circle_intersection.exterior.coords if not is_in_collision_area and not (
                            obstacle_step_circle_intersection.geom_type in NO_EXTERIOR_TYPES
                    ) else obstacle_step_circle_intersection.coords)
                restrictions.append(np.flip(object_restrictions) if is_in_collision_area else object_restrictions)
                restrictions = project_intervals_into_action_space(restrictions,
                                                                   low=Decimal(-180.0), high=Decimal(180.0))

        restrictions = [restriction for restriction in restrictions if restriction[0] != restriction[1]]

        # Build allowed action space from restrictions
        allowed_action_space = [[-self.ACTION_RANGE / 2, self.ACTION_RANGE / 2]]
        for restriction in restrictions:
            for index, allowed_subset in enumerate(allowed_action_space):
                if restriction[0] <= restriction[1]:
                    if restriction[0] < allowed_subset[0] <= restriction[1] <= allowed_subset[1]:
                        allowed_subset[0] = restriction[1]
                    if restriction[1] > allowed_subset[1] >= restriction[0] >= allowed_subset[0]:
                        allowed_subset[1] = restriction[0]
                    if restriction[0] >= allowed_subset[0] and restriction[1] <= allowed_subset[1]:
                        if allowed_subset[0] != restriction[0]:
                            allowed_action_space.append([allowed_subset[0], restriction[0]])
                        if allowed_subset[1] != restriction[1]:
                            allowed_action_space.append([restriction[1], allowed_subset[1]])
                        allowed_subset[0] = np.inf
                    if restriction[0] < allowed_subset[0] and restriction[1] > allowed_subset[1]:
                        allowed_subset[0] = np.inf
                else:
                    if restriction[0] <= allowed_subset[0] and restriction[1] <= allowed_subset[0] or (
                            restriction[0] >= allowed_subset[1]) and restriction[1] >= allowed_subset[1]:
                        allowed_subset[0] = np.inf
                    if allowed_subset[1] > restriction[0] > allowed_subset[0]:
                        allowed_subset[1] = restriction[0]
                    if allowed_subset[0] < restriction[1] < allowed_subset[1]:
                        allowed_subset[0] = restriction[1]

        allowed_action_space = np.array(
            [subset for subset in allowed_action_space if subset[0] != np.inf and subset[0] != subset[1]])

        if len(allowed_action_space) > 0:
            allowed_action_space[allowed_action_space[:, 0] != -self.ACTION_RANGE / 2, 0] += self.safety_angle
            allowed_action_space[allowed_action_space[:, 1] != self.ACTION_RANGE / 2, 1] -= self.safety_angle

        return [list(subset) for subset in allowed_action_space if subset[0] < subset[1]]

    def render(self, render_mode: Optional[str] = 'rgb_array', draw_trajectory: bool = False, draw_information=True):
        """ Renders the environment

        Args:
            render_mode (str)
            draw_trajectory (bool): Whether past steps should be indicated on the map
            draw_information (bool): Whether to show information about the reward, target distance, and allowed actions
        """
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled('Pygame is not installed, run `pip install pygame`')

        def draw_polygon_border(polygon_coordinates):
            for index, coordinate in enumerate(polygon_coordinates):
                if index == len(polygon_coordinates) - 1:
                    pygame.draw.line(canvas, (0, 0, 0), coordinate, polygon_coordinates[0], 2)
                else:
                    pygame.draw.line(canvas, (0, 0, 0), coordinate, polygon_coordinates[index + 1], 2)

        self.window_scale = 50 if self.current_width < 15 else 30

        window_width = self.current_width * self.window_scale
        window_height = self.current_height * self.window_scale
        if self.window is None or self.reload:
            pygame.init()
            pygame.font.init()
            if self.render_mode == 'human':
                pygame.display.init()
                self.window = pygame.display.set_mode((window_width, window_height))
                pygame.display.set_caption('Obstacle Avoidance')
            else:
                self.window = pygame.Surface((window_width, window_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((window_width, window_height))
        canvas.fill((232, 232, 232))

        for obstacle in self.obstacles:
            gfxdraw.filled_polygon(canvas, obstacle.coordinates * self.window_scale, (136, 136, 136))
            draw_polygon_border(obstacle.coordinates * self.window_scale)

        gfxdraw.pie(canvas, int(float(self.agent.x) * self.window_scale),
                    int(float(self.agent.y) * self.window_scale),
                    int(float(self.agent.step_size + self.agent.radius) * self.window_scale),
                    int(float(self.agent.perspective) - float(self.ACTION_RANGE) / 2),
                    int(float(self.agent.perspective) + float(self.ACTION_RANGE) / 2), (0, 0, 0))

        gfxdraw.filled_circle(canvas, int(float(self.agent.x) * self.window_scale),
                              int(float(self.agent.y) * self.window_scale),
                              int(float(self.agent.radius) * self.window_scale), (65, 105, 225))

        gfxdraw.circle(canvas, int(self.goal.centroid.coords[0][0] * self.window_scale),
                       int(self.goal.centroid.coords[0][1] * self.window_scale),
                       int((self.goal.bounds[3] - self.goal.centroid.coords[0][1]) * self.window_scale), (34, 139, 34))

        if draw_trajectory and len(self.trajectory) > 1:
            pygame.draw.aalines(canvas, (232, 232, 232), False, np.multiply(self.trajectory, self.window_scale), 0)

        if draw_information:
            font = pygame.font.SysFont('Arial', 14)
            text_canvas = font.render(
                f'Reward: {np.round(self.last_reward, 2)}',
                True, (0, 0, 0))
            perspective_canvas = font.render(
                f'Perspective: {np.round(float(self.agent.perspective), 2)}',
                True, (0, 0, 0))
            allowed_actions_canvas = font.render(
                f'Allowed Actions: {[(np.round(float(subset[0]), 2), np.round(float(subset[1]), 2)) for subset in self.allowed_actions]}',
                True, (0, 0, 0))

        canvas = pygame.transform.flip(canvas, False, True)
        self.window.blit(canvas, (0, 0))
        if draw_information:
            self.window.blit(text_canvas, (self.window_scale / 4, self.window_scale / 4))
            self.window.blit(perspective_canvas, (self.window_scale / 4, 4 * self.window_scale / 4))
            self.window.blit(allowed_actions_canvas, (self.window_scale / 4, 7 * self.window_scale / 4))
        if self.render_mode == 'human':
            pygame.event.pump()
            self.clock.tick(self.metadata['render_fps'])
            pygame.display.flip()
        elif self.render_mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )
        elif self.render_mode == 'jupyter':
            plt.imshow(np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            ))
            plt.axis('off')
            plt.show()

    def close(self):
        """ Closes the visualization
        """
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


def generate_obstacles(width: float, height: float, num_obstacles: int,
                       position_covariance: list = None,
                       mean_size_obstacle: float = 1.0, sigma_size_obstacle: float = 0.2,
                       range_size_obstacle: float = 0.5,
                       num_waypoints: int = 0, distance_waypoints: float = 2.0,
                       sigma_distance: float = 1.0, step_size: float = 0.3,
                       seed: int = None, obstacles: dict = None, forbidden_circles: list = None,
                       max_iterations: int = 10000, uniform: bool = False):
    """ Algorithm to generate environment setups

    Args:
        width (float): Width of the map
        height ( float): Height of the map
        num_obstacles (int): Number of obstacles
        position_covariance (list): Covariance matrix
        mean_size_obstacle (float): Mean size of an obstacle
        sigma_size_obstacle (float): Standard deviation of the obstacle size
        range_size_obstacle (float): Defines the minimum and maximum allowed obstacle sizes
        num_waypoints (int): Number of waypoints for each obstacle
        distance_waypoints (float): Mean distance of the straight path
        sigma_distance (float): Standard deviation of the waypoints' distance
        step_size (float): Step size of the obstacles
        seed (int): Seed to make generations reproducible
        obstacles (dict): Already existing obstacles in the environment
        forbidden_circles (list): List of circles in which no obstacle or waypoint should be placed
        max_iterations (int): Maximum generation trials before the next setup is taken
        uniform (bool): Whether to sample uniformly instead of a normal distribution

    Returns:
        setup (dict): obstacle setup which can be used in an environment configuration
    """
    if position_covariance is None:
        position_covariance = [[4.0, 0.0], [0.0, 4.0]]

    def is_valid(el_coordinates):
        out_of_map = minimum_distance > el_coordinates[0] or el_coordinates[0] > width - minimum_distance or (
                minimum_distance > el_coordinates[1]) or el_coordinates[1] > height - minimum_distance

        collision = np.any(
            [Point(midpoint(geometry['coordinates'])).distance(Point(el_coordinates)) < minimum_distance + np.sqrt(2 * (
                    (max(geometry['coordinates'][:, 1]) - min(geometry['coordinates'][:, 1])) / 2) ** 2) or np.any(
                [Point(waypoint).distance(Point(el_coordinates)) < minimum_distance for waypoint in
                 geometry['waypoints']]
            ) for geometry in obstacles.values()])

        if forbidden_circles is not None:
            in_forbidden_circle = np.any(
                [Point(circle[0], circle[1]).distance(Point(el_coordinates)
                                                      ) < circle[2] + minimum_distance for circle in forbidden_circles])
        else:
            in_forbidden_circle = False

        return not collision and not out_of_map and not in_forbidden_circle

    if seed is not None:
        np.random.seed(seed)

    if obstacles is None:
        obstacles = {}

    iteration = 0
    while len(obstacles) < num_obstacles:
        iteration += 1

        if uniform:
            size_obstacle = np.random.uniform(mean_size_obstacle - range_size_obstacle,
                                              mean_size_obstacle + range_size_obstacle)
        else:
            size_obstacle = np.clip(np.random.normal(mean_size_obstacle, sigma_size_obstacle),
                                    mean_size_obstacle - range_size_obstacle,
                                    mean_size_obstacle + range_size_obstacle)

        minimum_distance = np.sqrt(2 * (size_obstacle / 2) ** 2) + 0.95

        position = np.random.multivariate_normal([width / 2, height / 2],
                                                 position_covariance)

        position[0] = np.clip(position[0], 0.0, width - size_obstacle)
        position[1] = np.clip(position[1], 0.0, height - size_obstacle)
        coordinates = SHAPE_COLLECTION[np.random.randint(0, len(SHAPE_COLLECTION) - 1)] * size_obstacle + position - (
                size_obstacle / 2)
        centroid = np.array(Polygon(coordinates).centroid.coords[0])

        if is_valid(position) or iteration > max_iterations:
            iteration = 0
            waypoints = []
            while len(waypoints) < num_waypoints:
                iteration += 1
                radius = np.clip(np.random.normal(distance_waypoints, sigma_distance), 1.0, 4.0)
                update = np.array([np.cos(np.radians(np.random.uniform(low=0.0, high=259.9))) * radius,
                                   np.sin(np.radians(np.random.uniform(low=0.0, high=259.0))) * radius])

                if is_valid(centroid + update) or iteration > max_iterations:
                    iteration = 0
                    centroid += update
                    waypoints.append(centroid.copy())

            obstacles[len(obstacles)] = {'coordinates': coordinates, 'waypoints': waypoints, 'step_size': step_size}

    return obstacles
