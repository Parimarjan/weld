
LLVM_FLAGS:=$(shell llvm-config --cxxflags --libs --ldflags --system-libs)

all:libptxgen #ptxgen #llvmlink ptxgen

libptxgen:ptxgen.o llvmlink.o
	ar rc libptxgen.a ptxgen.o llvmlink.o

ptxgen:llvmlink.o ptxgen.o
	g++ ptxgen.o llvmlink.o -o ptxgen ${LLVM_FLAGS}

llvmlink.o:
	g++ -Wall -std=c++14 -fPIC -c llvmlink.c ${LLVM_FLAGS}

ptxgen.o:
	g++ -Wall -std=c++14 -fPIC -c ptxgen.c ${LLVM_FLAGS}

clean:
	rm -rf ptxgen.o libptxgen.a llvmlink.o ptxgen
