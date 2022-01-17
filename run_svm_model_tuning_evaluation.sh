#!/bin/bash

for dataset_name in FLAIR T1w T1wCE T2w
do
  python -u svm_model_tuning_evaluation.py -datasetName $dataset_name
done

