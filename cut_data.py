#!/usr/bin/env python3

import os
import argparse
import sys

def main(args):
    if not os.path.isfile(args.file):
        print("%s is not a file, please enter a valid path to a file" % args.file)
        sys.exit(1)

    in_file = open(args.file, "r")
    out_file = open(args.output, "w")

    lines = in_file.read().split(args.delim)
    lines = lines[:args.count]
    for line in lines:
        out_file.write(line)
        out_file.write(args.delim)

    out_file.close()
    in_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cut a portion of a training file on a delimiter")
    parser.add_argument("file", metavar="file", help="The file to cut")
    parser.add_argument("--output", "-o", dest="output", metavar="file", default="output.txt", type=str, help="The output file, default is 'output.txt' in the current directory")
    parser.add_argument("--delim", "-d", dest="delim", metavar="D", default="\n", type=str, help="The delimiter to split inputs on, default is '\\n' (newline)")
    parser.add_argument("--count", "-c", dest="count", metavar="N", default=1, type=int, help="The number of inputs to keep, default is 1")

    args = parser.parse_args()

    main(args)
