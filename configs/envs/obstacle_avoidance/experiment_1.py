experiment_1_setup = {
    'STEPS_PER_EPISODE': 40,
    'ACTION_RANGE': 220,
    'DT': 1.0,
    'SAFETY_DISTANCE': 0.05,
    'REWARD': {
        'TIMESTEP_PENALTY_COEFFICIENT': 0.05,
        'REWARD_COEFFICIENT': 5.0,
        'GOAL': 50.0,
        'COLLISION': -20.0
    },
    'LEVELS': {
        1: {
            'HEIGHT': 15.0,
            'WIDTH': 15.0,
            'AGENT': {'x': 1.0, 'y': 1.0, 'angle': 90.0, 'step_size': 1.0, 'radius': 0.4},
            'GOAL': {'x': 12.0, 'y': 12.0, 'radius': 0.5},
            'OBSTACLES': {}
        }
    }
}
