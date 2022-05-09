#!/usr/bin/env python3
import argparse
import sys

sys.path.append("./lib")
from learner import Learner
from cfg.parameters import create_params

def main():
    parser = argparse.ArgumentParser(description='Test Planning Network')
    parser.add_argument('--params_file', help='Path to settings yaml', required=True)

    args = parser.parse_args()
    params = create_params(args.params_file, mode='test')

    learner = Learner(params=params)
    learner.test()

if __name__ == "__main__":
    main()