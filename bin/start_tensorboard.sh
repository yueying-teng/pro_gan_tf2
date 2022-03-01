#! /bin/bash

ARG1=${1:-/work/notebooks/test_train/logs}

echo "Starting tensorboard with logs in $ARG1"
tensorboard --logdir $ARG1 --host 0.0.0.0 --port 6006


