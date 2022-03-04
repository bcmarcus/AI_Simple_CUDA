CC = gcc
CCPP = g++
NVCC = /usr/bin/nvcc

INCLUDE = ../../
DEFINES = -DUSE_ZLIB

OPTFLAGS = -O2 -std=c++2a -Wall

NVFLAGS = -I $(INCLUDE) -Xcompiler -O2 -L/usr/local/cuda-11.6/lib64 -lcuda -lstdc++ -lcudart
NVLINKFLAGS = -L/usr/local/cuda-11.6/lib64 -lcuda -lcudart 
#-lstdc++

CPPFLAGS = $(OPTFLAGS) -I $(INCLUDE) $(DEFINES) 
LINKFLAGS = -lz 

MAGICKLINKFLAGS = `Magick++-config --ldflags --libs`

objs = src/testSummation.o \
	src/Matrix3D.o \
	src/time.o \
	src/cudaErrors.o

all: program

program: $(objs)
	$(CCPP) $(CPPFLAGS) -o testSummation.out  $^ $(LINKFLAGS) $(NVLINKFLAGS) $(MAGICKLINKFLAGS)

src/time.o: $(INCLUDE)coreutils/util/time.cpp 
	$(CCPP) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

src/cudaErrors.o: $(INCLUDE)coreutils/util/cudaErrors.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/Matrix3D.o: $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/testSummation.o: testSummation.cu $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) -c $< -o $@

clean: 
	rm -f src/*.o program