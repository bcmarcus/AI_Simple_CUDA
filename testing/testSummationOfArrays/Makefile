CC = gcc
CCPP = g++
NVCC = /usr/local/cuda-11.8/bin/nvcc

INCLUDE = ../../
DEFINES = -DUSE_ZLIB

OPTFLAGS = -O2 -std=c++2a -Wall

NVFLAGS = -I $(INCLUDE) -I $(INCLUDE)../ -Xcompiler -O2 -L/usr/local/cuda-11.8/lib64 -lcuda -lstdc++ -lcudart -dc --std=c++17
CPPFLAGS = $(OPTFLAGS) -I $(INCLUDE) -I $(INCLUDE)../ $(DEFINES)

MAGICKLINKFLAGS = `Magick++-config --ldflags --libs`
NVLINKFLAGS = -L/usr/local/cuda-11.8/lib64 -lcuda -lcudart
LINKFLAGS = -lz 

NVCCOBJS = src/testSummation.o \
	src/Matrix3D.o \

all: program

program: $(NVCCOBJS) src/gpuCode.o
	$(CCPP) $(CPPFLAGS) -o testSummation.out  $^ $(LINKFLAGS) $(NVLINKFLAGS)

src/gpuCode.o: $(NVCCOBJS)
	$(NVCC) --device-link -o $@ $^ $(LINKFLAGS) $(NVLINKFLAGS) $(CPPOBJS)

src/Matrix3D.o: $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/testSummation.o: testSummation.cu $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) -c $< -o $@

clean: 
	rm -f src/*.o program