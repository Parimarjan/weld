#!/bin/bash
#llvm-link /tmp/kernel.ll $CUDA_PATH/nvvm/libdevice/libdevice.compute_50.10.bc -o /tmp/t2.linked.bc
#opt -internalize -internalize-public-api-list=kernel -nvvm-reflect-list=__CUDA_FTZ=0 -nvvm-reflect -O3 /tmp/t2.linked.bc -o /tmp/t2.opt.bc
#opt -nvvm-reflect-list=__CUDA_FTZ=0 -nvvm-reflect -O3 /tmp/t2.linked.bc -o /tmp/t2.opt.bc
opt -O3 t2.linked.bc -o t2.opt.bc
rm -f t2.linked.bc
llc -mcpu=sm_20 t2.opt.bc -o /tmp/kernel.ptx
rm -f /tmp/t2.opt.bc
