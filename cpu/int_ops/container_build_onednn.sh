#!/bin/bash

mkdir -p oneDNN/build
mkdir -p onednn_package

cd oneDNN/build
cmake -DCMAKE_INSTALL_PREFIX="$PWD"/../../onednn_package -DDNNL_CPU_RUNTIME=OMP -DDNNL_LIBRARY_TYPE=STATIC ..
make -j16