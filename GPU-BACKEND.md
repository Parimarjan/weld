# Weld GPU Backend

## Contents

* [Overview](#overview)
* [Compiling](#compiling)
* [Developer Documentation](#developer-documentation)
* [Examples](#examples)
* [TODO](#todo)

## Overview

#### Setup and Requirements

* Requires
    * CUDA (tested with version 8.0)
        * set environment variable CUDA_PATH to point to the base installation
        directory. On dawn this is, ``/usr/local/cuda-8.0''
	* set environment variable LD_LIBRARY_PATH to point to ``/usr/local/cuda-8.0/lib64''
    * pytest (for python based tests)
* We write (and clean up) temporary compilation files to /etc/
* WELD_HOME environment variable should be set up
* WELD_HOME/compile-ptx.sh should be executable
* Note: To run the NVVM code, ``NVVM_FLAG`` in weld/llvm.rs should be true

```bash
cd python/numpy
pytest tests-gpu.py
```

* All the current tests should pass

## Compiling

There seem to be two compilers we can use: the usual llvm compiler (with the
nvptx backend), or NVIDIA's proprietary compiler (libNVVM). In theory, both
should work, but with a few subtler differences:

* libNVVM is usually tied to a particular version of LLVM (not the latest
version) - thus the generated code might need to be slightly modified.
* libNVVM appears to have more optimization passes - and probably generates
better code according to a few online sources.

#### Manual Compilation

Currently, we emit the generated NVVM file, compile it by executing a shell
script and using LLVM's CLI, and then use the generated .ptx file in the cuda
code.

#### Programatic Compilation

Ideally, we should be able to use LLVM's API to compile the file / or libNVVM's
API. But we are still debugging some issues with the LLVM programmatic
compilation.

## Developer Documentation

#### New Code

* The new code is concentrated across the files weld/llvm.rs,
and weld_rt/cpp/weld_cuda_backend.cpp. In weld/llvm.rs, the new functions are
divided into two parts:
    * code generation for the gpu kernel.
    * llvm code that glues it together with the rest of the generated program

These are both in separate ``impl LlvmGenerator'' blocks.
(TODO: need to look further into subclassing LlvmGenerator -- but it wasn't
immediately clear if that would work without much modifications)

* Most of these functions have a similar function in the llvm code-gen, and we
try to call the llvm code gen functions whenever possible. In the comments at
the top of each function, we describe the differences between the two cases.

#### Code Gen

Note: there are two separate code generation contexts: the NVVM kernel code
(which is essentially the code of a ``for`` loop right now), and the glue llvm
code that ties it back in with the rest of the weld program, launches the
kernel, gets the result back etc.

* We generate NVVM code, mostly following the descriptions in LLVM's [NVPTX
backend](https://llvm.org/docs/NVPTXUsage.html).
* Currently, the kernel operates over a single element, at index:


##### Differences from usual Weld code-gen

* Continuation functions:
    * This wasn't completely trivial because the for loop is being executed as
    a gpu kernel. In some cases, I believe the continuation function involves
    calling a new function -- but we probably can't do it from the gpu kernel.
    This seems to be required for nested for loops, or weld code of the type:

    * Continuation functions are also used for supporting if blocks -- this
    should be possible to add without much of a change.

A few other minor differences are:

* Don't run the vectorization pass as no SIMD for gpu
*

## TODO

#### Reductions

* currently, we do sum by calling a thrust function on the gpu array generated
by previous computations, but this is clearly slower in comparison to
generating a single kernel call. (For instance, thrust's transform_and_reduce
option performs clearly better than our implementation)

#### Annotations

* Add annotations to each for loop so we can easily add the gpu v/s cpu flag in
the weld code - perhaps during one of the transformations / or let the user
specify it.

#### Programmatic Compilation


#### Choosing between GPU and CPU execution in weld

* First, we should manually experiment with the workloads we already have to
figure out what the correct behaviour should be.

* Finally, we will potentially need a cost model for the cpu / and gpu inside
weld. Choosing whether to execute a ``for'' loop on the GPU or CPU could happen
after all the transformation passes are done.

#### Minor

* free the device output in case of reductions
* Decide when we definitely don't want to offload to gpu:
    - e.g., strided accesses.
* support loops over contiguous arrays specified with iter(e0,start,end,1)
* support result(for([0,1],...merger ... (merge(b, result(for(iter(e0,....) appender)))))
    - this does not seem urgent for now. Probably if we start to use
    continuation functions when generating the nvvm kernels, then this might be
    easy.
* do we need to add if (i < n) in the kernel?
* Streaming kernel support: should only copy partial data at a time using
MemcpyAsync, and then operate on that.
    - Currently, just copy everything first.
    - probably just always do it from the c++ code?
    - need choose block sizes etc.
* All types seem to work, but needs to be tested more thoroughly.

