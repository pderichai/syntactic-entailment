#!/usr/bin/env python3

import argparse
import csv
import json
import sys
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix

# For SNLI, MultiNLI
#tags = ['entailment', 'contradiction', 'neutral']
# For SciTail
tags = ['entailment', 'neutral']


def main():
    parser = argparse.ArgumentParser(description='Generate confusion matrix from analysis file.')
    parser.add_argument('analysis_file')
    parser.add_argument('output_file')
    args = parser.parse_args()

    expectations = []
    predictions = []
    with open(args.analysis_file, 'r') as f:
        f_csv = csv.DictReader(f, delimiter='\t')
        for line in f_csv:
            expectations.append(line['gold_label'])
            predictions.append(line['label'])


    cnf_matrix = confusion_matrix(expectations, predictions, labels=tags);
    plot_confusion_matrix(cnf_matrix, tags, True)
    plt.savefig(args.output_file)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Gold Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()


if __name__ == '__main__':
    main()
