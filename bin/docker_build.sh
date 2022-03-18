#! /bin/bash

ENV_VAR_FILE="config/env_vars.sh"
source $ENV_VAR_FILE
echo "Using env vars in: $ENV_VAR_FILE"

IMG_NAME="$PROJECT_NAME"
DOCKERFILE="docker/Dockerfile"

echo "Using Dockerfile: $DOCKERFILE"
echo "Building locally: $IMG_NAME"

docker build --no-cache -f $DOCKERFILE -t $IMG_NAME .
