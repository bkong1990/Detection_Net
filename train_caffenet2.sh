#!/usr/bin/env sh

/usr/local/caffe/build/tools/caffe train \
    --solver=solver.prototxt --gpu=2 >> train.log 2>&1
#    --solver=solver2.prototxt --weights=bvlc_reference_caffenet.caffemodel --gpu=3 >> train2.log 2>&1
