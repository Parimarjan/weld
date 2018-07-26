#!/bin/bash
#llvm-link /tmp/kernel.ll /usr/local/cuda-8.0/nvvm/libdevice/libdevice.compute_50.10.bc -o /tmp/t2.linked.bc
echo "hello compile ptx started"
llvm-link /tmp/kernel.ll $CUDA_PATH/nvvm/libdevice/libdevice.compute_50.10.bc -o /tmp/t2.linked.bc
#rm -f /tmp/kernel.ll
opt -internalize -internalize-public-api-list=kernel -nvvm-reflect-list=__CUDA_FTZ=0 -nvvm-reflect -O3 /tmp/t2.linked.bc -o /tmp/t2.opt.bc
#rm -f /tmp/t2.linked.bc
echo "going to make kernel.ptx"
llc -mcpu=sm_20 /tmp/t2.opt.bc -o /tmp/kernel.ptx
#rm -f /tmp/t2.opt.bc
