import argparse

from configs.search_spaces.ppo_space import ppo_search_space
from configs.search_spaces.td3_space import td3_search_space
from configs.search_spaces.dqn_space import dqn_search_space
from configs.search_spaces.mps_td3_space import mps_td3_search_space
from configs.search_spaces.pam_space import pam_search_space
from configs.base_config import training_config
from src.run import train_agent, test_agent


def _train(args):
    if args.algorithm.startswith('PPO'):
        train_agent(args.algorithm, args.environment,
                    training_config['HYPERPARAMETERS'],
                    ppo_search_space if args.hpo else None)
    elif args.algorithm == 'TD3':
        train_agent(args.algorithm, args.environment,
                    training_config['HYPERPARAMETERS'],
                    td3_search_space if args.hpo else None)
    elif args.algorithm.startswith('DQN'):
        train_agent(args.algorithm, args.environment,
                    training_config['HYPERPARAMETERS'],
                    dqn_search_space if args.hpo else None)
    elif args.algorithm == 'MPS-TD3':
        train_agent(args.algorithm, args.environment,
                    training_config['HYPERPARAMETERS'],
                    mps_td3_search_space if args.hpo else None)
    elif args.algorithm == 'PAM':
        train_agent(args.algorithm, args.environment,
                    training_config['HYPERPARAMETERS'],
                    pam_search_space if args.hpo else None)


def _evaluate(args):
    test_agent()


def _get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='action', help='Action to perform')

    train = subparsers.add_parser('train', help='Train an agent or perform hyperparameter optimization.')
    train.add_argument('--algorithm', type=str, default='PPO', help='The algorithm to train.')
    train.add_argument('--environment', type=str, default='obstacle_avoidance', help='The environment to train in.')
    train.add_argument('--hpo', action='store_true')
    train.set_defaults(action=_train)

    evaluate = subparsers.add_parser('evaluate', help='Evaluate a locally saved agent.')
    evaluate.set_defaults(action=_evaluate)

    return parser


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    if not hasattr(args, 'action'):
        parser.print_help()
        parser.exit()

    args.action(args)
