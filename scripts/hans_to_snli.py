#!/usr/bin/env python3

import csv
import jsonlines


with open('/home/dericp/workspace/hans/heuristics_evaluation_set.txt', 'r') as hans_file, open('heuristics_evaluation_set.jsonl', 'w') as hans_jsonl_file:
    hans = csv.DictReader(hans_file, delimiter='\t')
    hans_jsonl = jsonlines.Writer(hans_jsonl_file)
    for line in hans:
        hans_jsonl.write(line)
