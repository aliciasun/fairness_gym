import argparse
import os




from environment import Environment


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', type=str, default='causal',
                        help='environment type')
    args = parser.parse_args()

    env = Environment(args.environment)