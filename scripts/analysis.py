#!/usr/bin/env python3

import csv
import argparse


def main():
    parser = argparse.ArgumentParser(description='Generate analysis tsv.')
    parser.add_argument('csv1')
    parser.add_argument('csv2')
    parser.add_argument('out1')
    parser.add_argument('out2')
    args = parser.parse_args()

    with open(args.csv1, 'r') as csv1, open(args.csv2, 'r') as csv2, open(args.out1, 'w') as out1, open(args.out2, 'w') as out2:
        dict1 = csv.DictReader(csv1, delimiter='\t')
        dict2 = csv.DictReader(csv2, delimiter='\t')
        out1.write('premise\thypothesis\tvanilla_label\tvariant_label\tgold_label\n')
        out2.write('premise\thypothesis\tvanilla_label\tvariant_label\tgold_label\n')
        for line1, line2 in zip(dict1, dict2):
            assert line1['gold_label'] == line2['gold_label'] and line1['premise'] == line2['premise'] and line1['hypothesis'] == line2['hypothesis']
            gold_label = line1['gold_label']
            premise = line1['premise'].rstrip()
            hypothesis = line1['hypothesis'].rstrip()
            if line1['label'] != gold_label and line2['label'] == gold_label:
                out1.write(premise.rstrip() + '\t' + hypothesis.rstrip() + '\t' + line1['label'] + '\t' + line2['label'] + '\t' + gold_label + '\n')
            if line1['label'] == gold_label and line2['label'] != gold_label:
                out2.write(premise.rstrip() + '\t' + hypothesis.rstrip() + '\t' + line1['label'] + '\t' + line2['label'] + '\t' + gold_label + '\n')


if __name__ == '__main__':
    main()
