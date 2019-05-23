#!/usr/bin/env python3

import glob
import json
import csv
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Generate results csv from directory.')
    parser.add_argument('dir', help='Directory where all results are stored.')
    parser.add_argument('out_file', help='What file to save results to.')
    args = parser.parse_args()

    with open(args.out_file, 'w') as data:
        fieldnames = ['model', 'best_epoch', 'training_epochs',
                      'training_accuracy', 'validation_accuracy',
                      'best_validation_accuracy']
        writer = csv.DictWriter(data, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for filename in glob.iglob(args.dir + '/**/metrics.json'):
            model_name = os.path.basename(os.path.dirname(filename))
            with open(filename) as f:
                metrics = json.load(f)
                metrics['model'] = model_name
            writer.writerow(metrics)


if __name__ == '__main__':
    main()
