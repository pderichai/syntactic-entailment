#!/usr/bin/env bash

DIR=$1

for CONFIG in $DIR/*; do
    if [[ "$CONFIG" != *json ]] ;
    then
        OUT_DIR=$(basename "$CONFIG")
        echo "Training config at: $CONFIG ..."
        allennlp train "$CONFIG" -s "out/$OUT_DIR" --include-package syntactic_entailment
    fi
done
