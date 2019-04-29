#!/usr/bin/env python3

import argparse
import json
import _jsonnet

def main(args):
    json_str = _jsonnet.evaluate_file(args.config_path)
    config_json = json.loads(json_str)
    config_json['trainer']['optimizer']['type'] = 'adagrad'
    json.dump(config_json, open('test', 'w'), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    args = parser.parse_args()
    main(args)
