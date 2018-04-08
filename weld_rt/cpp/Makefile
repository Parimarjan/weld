OS = $(shell uname -s)
LLVM_VERSION = $(shell llvm-config --version | cut -d . -f 1,2)

# TODO: Maybe use -L instead of -I??
CFLAGS = -O2 -std=c++11 -stdlib=libstdc++ -Wall -fno-use-cxa-atexit -fPIC -flto 
CFLAGS2 = -O2 -std=c++11 -stdlib=libstdc++ -Wall -fno-use-cxa-atexit -fPIC
#CFLAGS = -O2 -std=c++11 -stdlib=libstdc++ -Wall -fno-use-cxa-atexit -fPIC -flto -I/usr/local/cuda-8.0/include -lcuda-8.0
#CFLAGS = -O2 -std=c++11 -fno-use-cxa-atexit -fPIC -flto -I/usr/local/cuda-8.0/include -L/usr/local/cuda-8.0/lib64 -L/usr/local/cuda-8.0/lib/stubs -lcuda -lcudart

ifeq (${OS}, Darwin)
  # OS X
  CLANG ?= clang++-${LLVM_VERSION}
else ifeq (${OS}, Linux)
  # Linux
  CLANG ?= clang++-${LLVM_VERSION} 
else
  $(error Unsupported platform: ${OS})
endif


all: libweldrt.bc

vb.o: vb.cpp
	${CLANG} ${CFLAGS} -c $< -o $@

merger.o: merger.cpp
	${CLANG} ${CFLAGS} -c $< -o $@

runtime.o: runtime.cpp
	${CLANG} ${CFLAGS} -c $< -o $@
	#${CLANG} ${CFLAGS} $< -o $@

cuda-test.o: cuda-test.cpp
	#${CLANG} -O2 $< -g -c -I/usr/local/cuda-8.0/include -lcuda -o $@
	${CLANG} ${CFLAGS} -I/usr/local/cuda-8.0/include -lcuda -c $< -o $@

libweldrt.bc: vb.o merger.o runtime.o cuda-test.o
	llvm-link-${LLVM_VERSION} $^ -o $@

clean:
	rm -f *.bc *.o

.phony: all

