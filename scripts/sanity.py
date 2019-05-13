#!/usr/bin/env python3

import json

dev_file = open('multinli_1.0/multinli_1.0_dev_matched.jsonl')
dev_ids = []
dev_labels = []
for line in dev_file:
    line_json = json.loads(line)
    dev_ids.append(line_json['pairID'])
    dev_labels.append(line_json['gold_label'])

pred_file = open('da-multinli-matched-dev-pred.csv')

pred_ids = []
pred_labels = []
for line in pred_file:
    pred_ids.append(line.split(',')[0])
    pred_labels.append(line.split(',')[1])

correct = 0
for idx in range(len(pred_labels)):
    #print(dev_labels[idx])
    #print(pred_labels[idx])
    #print()
    assert dev_ids[idx] == pred_ids[idx]
    if dev_labels[idx].strip() == pred_labels[idx].strip():
        correct += 1

print('dev acc', correct / len(dev_labels))
