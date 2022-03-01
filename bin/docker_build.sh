#! /bin/bash

ENV_VAR_FILE="config/env_vars.sh"
source $ENV_VAR_FILE
echo "Using env vars in: $ENV_VAR_FILE"

IMG_NAME="$PROJECT_NAME"
DOCKERFILE="docker/Dockerfile"

echo "Using Dockerfile: $DOCKERFILE"
echo "Building locally: $IMG_NAME"

if [[ -z "${http_proxy}" ]]; then
  docker build --no-cache -f $DOCKERFILE -t $IMG_NAME .
else
  echo "Proxy configuration found: Building with build-arg proxy=1"
  docker build --no-cache -f $DOCKERFILE -t $IMG_NAME --build-arg proxy=1 .
fi

