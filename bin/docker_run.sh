#! /bin/bash

ENV_VAR_FILE="config/env_vars.sh"
source $ENV_VAR_FILE
echo "Using env vars in: $ENV_VAR_FILE"

IMG_NAME="$PROJECT_NAME"

# for gpu server
docker run --gpus all \
    -v $PWD:/work \
    -e PYTHONUNBUFFERED=1 \
    -e PYTHONIOENCODING=UTF-8 \
    -p 8888:8888 \
    -p 6006:6006 \
    -itd $IMG_NAME

# for non gpu server
# docker run \
#     -v $PWD:/work \
#     -p 8888:8888 \
#     -itd $IMG_NAME
