# Makefile
CC = gcc
CCPP = g++
NVCC = /usr/local/cuda-11.6/bin/nvcc

INCLUDE = ../../../
DEFINES = -DUSE_ZLIB

OPTFLAGS = -O2 -std=c++2a -Wall

NVFLAGS = -I $(INCLUDE) -I $(INCLUDE)../ -Xcompiler -O2 -L/usr/local/cuda-11.6/lib64 -lcuda -lstdc++ -lcudart -dc -arch=sm_86 --std=c++17
NVLINKFLAGS = -L/usr/local/cuda-11.6/lib64 -lcuda -lcudart

CPPFLAGS = $(OPTFLAGS) -I $(INCLUDE) -I $(INCLUDE)../ $(DEFINES)
LINKFLAGS = -lz 

MAGICKLINKFLAGS = `Magick++-config --ldflags --libs`

NVCCOBJS = src/testGeneralImageNetwork.o \
	src/cudaErrors.o \
	src/BasicLayerList.o \
	src/Matrix3D.o \
	src/BasicLayer.o \
	src/BasicWeight.o \
	src/activationFunctions.o

CPPOBJS = src/time.o \
	src/print.o \
	src/generateInput.o \
	src/sortHelpers.o \
	src/sortingAlgorithms.o


all: linkCPU

linkCPU: $(CPPOBJS) $(NVCCOBJS) src/gpuCode.o
	$(CCPP) $(CPPFLAGS) -o testGeneralImageNetwork.out  $^ $(LINKFLAGS) $(NVLINKFLAGS) $(MAGICKLINKFLAGS)

src/gpuCode.o: $(NVCCOBJS)
	$(NVCC) --device-link -arch=sm_86 -o $@ $^ $(LINKFLAGS) $(NVLINKFLAGS) $(CPPOBJS)

src/time.o: $(INCLUDE)coreutils/util/time.cpp 
	$(CCPP) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

src/print.o: $(INCLUDE)coreutils/functions/debug/print.cpp 
	$(CCPP) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

src/sortHelpers.o: $(INCLUDE)coreutils/functions/sort/sortHelpers.cpp 
	$(CCPP) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

src/sortingAlgorithms.o: $(INCLUDE)coreutils/functions/sort/sortingAlgorithms.cpp 
	$(CCPP) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

src/generateInput.o: $(INCLUDE)../imageEdit/generate/generateInput.cpp $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(CCPP) $(CPPFLAGS) $(CFLAGS) -c $< -o $@ `Magick++-config --cxxflags --cppflags`

src/Matrix3D.o: $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cu
	$(NVCC) $(NVFLAGS) $< -o $@

src/cudaErrors.o: $(INCLUDE)coreutils/util/cudaErrors.cu $(INCLUDE)coreutils/util/cudaErrors.cuh 
	$(NVCC) $(NVFLAGS) $< -o $@

src/activationFunctions.o: $(INCLUDE)artificialIntelligence/functions/activationFunctions.cu $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

src/BasicWeight.o: $(INCLUDE)artificialIntelligence/classes/BasicWeight.cu $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

src/BasicLayer.o: $(INCLUDE)artificialIntelligence/classes/BasicLayer.cu $(INCLUDE)artificialIntelligence/classes/BasicWeight.cuh $(INCLUDE)artificialIntelligence/functions/activationFunctions.cuh $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

src/BasicLayerList.o: $(INCLUDE)artificialIntelligence/classes/BasicLayerList.cu $(INCLUDE)artificialIntelligence/classes/BasicLayer.cuh $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

src/testGeneralImageNetwork.o: $(INCLUDE)artificialIntelligence/neuralNetworks/testGeneralImageNetwork.cu $(CFLAGS) $(INCLUDE)artificialIntelligence/classes/BasicLayerList.hpp $(INCLUDE)artificialIntelligence/classes/BasicLayer.cuh $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

clean1: 
	rm -f src/*.o ./testGeneralImageNetwork.out

clean: 
	rm -f ./testGeneralImageNetwork.out