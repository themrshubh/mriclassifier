#!/bin/bash

for dataset_name in FLAIR T1w T1wCE T2w
do
  python -u cnn_model_evaluation.py -datasetName $dataset_name
done
