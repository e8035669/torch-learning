#!/bin/bash

proxy=${http_proxy}
if [ -n "${proxy}" ]; then
    p_export="export http_proxy=${proxy} https_proxy=${proxy} &&"
    echo "with proxy"
fi

gpu=""
if [ -e /dev/nvidia0 ]; then
    gpu="--gpus all"
    echo "with gpu"
fi

sudo docker run -it ${gpu} -v ${PWD}/model:/app/model -v ${PWD}/log:/log -p 5000:5000 --name "model-deploy" continuumio/miniconda3 bash -c "${p_export} pip install mlflow && mlflow models serve -m /app/model -p 5000 -h 0.0.0.0"
