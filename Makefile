
LLVM_FLAGS:=$(shell llvm-config --cxxflags)

all:libptxgen

libptxgen:ptxgen.o
	ar rc libptxgen.a ptxgen.o
ptxgen.o:
	g++ -Wall -std=c++14 -fPIC -c ptxgen.c ${LLVM_FLAGS}

clean:
	rm -rf ptxgen.o libptxgen.a
