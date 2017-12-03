#!/bin/bash

sdk_dir=$(pwd -P)

# default values unless variables are set
opt_dir=${TINYDNN_OPT_DIR:='/opt/tiny-dnn'}
docker_img=${DOCKER_IMG:='tinydnn/tinydnn:latest'}

docker run --rm -it -v $sdk_dir:$opt_dir -w $opt_dir $docker_img "$@"
