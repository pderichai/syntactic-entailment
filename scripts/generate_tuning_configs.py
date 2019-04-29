#!/usr/bin/env python3

import os
import argparse
import copy
import json
import _jsonnet

import numpy as np

TUNE_OUT = 'tune-out'
CONFIG_DIR = 'config'


def main(args):
    config_filename = os.path.basename(args.config_path)
    model_version = os.path.basename(os.path.dirname(args.config_path))
    tune_config_dir = os.path.join(CONFIG_DIR, model_version, 'tuning')
    json_str = _jsonnet.evaluate_file(args.config_path)
    config_json = json.loads(json_str)

    with open(args.params_file, 'r') as f:
        params_json = json.load(f)
    for i in range(args.num_configs):
        new_config_json = copy.deepcopy(config_json)
        config_json_iter = new_config_json
        for key_str, param_range in params_json.items():
            keys = key_str.split(':')
            for key in keys[:-1]:
                config_json_iter = config_json_iter[key]
            if len(param_range) > 2 or isinstance(param_range[0], str):
                config_json_iter[keys[-1]] = np.random.choice(param_range)
            elif isinstance(param_range[0], int):
                config_json_iter[keys[-1]] = np.random.randint(param_range[0],
                                                               param_range[1] + 1)
            elif isinstance(param_range[0], float):
                config_json_iter[keys[-1]] = np.random.uniform(param_range[0],
                                                               param_range[1])
        new_config_file = open(os.path.join(tune_config_dir,
                                            config_filename + '-' + str(i + 1)),
                               'w')
        json.dump(new_config_json, new_config_file, indent=4)


if __name__ == '__main__':
    np.random.seed(1337)
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('params_file')
    parser.add_argument('num_configs', type=int)
    args = parser.parse_args()
    main(args)
