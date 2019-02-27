#! /usr/bin/env bash

set -e
set -x

if [ $# -ne 1 ]; then
  echo "usage: ./benchmark_inference.sh [cuda_device]"
  exit 1
fi

cuda_device=$1

make clean && make -j

for batch_size in 1 2 4 8 16 32 64 128; do
  for model in "resnet18" "resnet34" "resnet50" "resnet101" "resnet152"; do
    for x in `seq 1 $batch_size`; do 
      echo resnets/${model}v2/test_data_set_0/input_0.pb
    done | xargs ./simpleOnnx resnets/${model}v2/${model}v2.onnx ${cuda_device} > ${model}_${batch_size}_output.txt
  done
done
