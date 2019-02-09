#!/usr/bin/env python3

import os
import glob
import argparse


def main(args):
    if not os.path.isdir(args.directory):
        print(args.directory, '''is not a directory. Please enter a valid path
            to a directory.''')
        sys.exit(1)

    train = open(args.out + '/train.ptb', 'w')
    dev = open(args.out + '/dev.ptb', 'w')
    test = open(args.out + '/test.ptb', 'w')

    for filename in glob.iglob(args.directory + '/data/penntree/**',
            recursive=True):
        if os.path.isfile(filename):
            split_num = int(filename.split(os.sep)[-2])
            with open(filename) as f:
                for line in f:
                    if split_num < 23:
                        print(line, end='', file=train)
                    elif split_num == 23:
                        print(line, end='', file=test)
                    elif split_num == 24:
                        print(line, end='', file=dev)

    train.close()
    dev.close()
    test.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Flatten PTB-revised data
            into train, dev, and test files.''')

    parser.add_argument('directory', help='''The directory of the
            PTB-revised data.''')
    parser.add_argument('--out', '-o', dest='out', help='''The directory of
            where the output should be saved.''')


    args = parser.parse_args()

    main(args)
