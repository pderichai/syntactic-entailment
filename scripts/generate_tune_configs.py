#!/usr/bin/env python3

import os
import argparse
import copy
import json
import _jsonnet

import numpy as np

TUNE_OUT = 'tune-out'
CONFIG_DIR = 'config'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('params_file')
    parser.add_argument('num_configs', type=int)
    args = parser.parse_args()

    config_filename = os.path.basename(args.config_file)
    model_version = os.path.basename(os.path.dirname(args.config_file))
    tune_config_dir = os.path.join(CONFIG_DIR, model_version, os.path.basename(os.path.dirname(args.params_file)))
    json_str = _jsonnet.evaluate_file(args.config_file)
    config_json = json.loads(json_str)

    with open(args.params_file, 'r') as f:
        params_json = json.load(f)

    np.random.seed(params_json['params_random_seed'])

    for i in range(args.num_configs):
        new_config_json = copy.deepcopy(config_json)
        for key_str, param_range in params_json.items():
            if key_str == 'params_random_seed':
                continue
            config_json_iter = new_config_json
            keys = key_str.split(':')
            for key in keys[:-1]:
                config_json_iter = config_json_iter[key]

            if any(isinstance(el, list) for el in param_range):
                param_list = []
                for sub_range in param_range:
                    param_list.append(get_param_val(sub_range, keys))
                config_json_iter[keys[-1]] = param_list
            else:
                config_json_iter[keys[-1]] = get_param_val(param_range, keys)

        new_config_file = open(
            os.path.join(tune_config_dir,
                         config_filename.split('.')[0] +  '-' + str(i + 1)),
            'w')
        json.dump(new_config_json, new_config_file, indent=4)


def get_param_val(param_range, keys):
    if len(param_range) == 1:
        return param_range[0]
    if len(param_range) > 2 or isinstance(param_range[0], str):
        return np.random.choice(param_range)
    if isinstance(param_range[0], int):
        return np.random.randint(param_range[0], param_range[1] + 1)
    if isinstance(param_range[0], float):
        # in the case that we're tuning the learning rate
        if keys[-1] == 'lr':
            return 10**(np.random.uniform(param_range[0], param_range[1]))
        return np.random.uniform(param_range[0], param_range[1])


if __name__ == '__main__':
    main()
