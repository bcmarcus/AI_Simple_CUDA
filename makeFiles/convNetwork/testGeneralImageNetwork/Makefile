CC = gcc
CCPP = g++
NVCC = /usr/local/cuda-11.6/bin/nvcc

INCLUDE = ../../../
DEFINES = -DUSE_ZLIB

OPTFLAGS = -O2 -std=c++2a -Wall

NVFLAGS = -I $(INCLUDE) -I $(INCLUDE)../ -Xcompiler -O2 -L/usr/local/cuda-11.6/lib64 -lcuda -lstdc++ -lcudart -dc --std=c++17
CPPFLAGS = $(OPTFLAGS) -I $(INCLUDE) -I $(INCLUDE)../ $(DEFINES)

MAGICKLINKFLAGS = `Magick++-config --ldflags --libs`
NVLINKFLAGS = -L/usr/local/cuda-11.6/lib64 -lcuda -lcudart
LINKFLAGS = -lz 

NVCCOBJS = src/testGeneralImageNetwork.o \
	src/BasicLayerList.o \
	src/Matrix3D.o \
	src/BasicLayer.o \
	src/BasicWeight.o \
	src/activationFunctions.o

CPPOBJS = src/generateInput.o \


all: linkCPU

linkCPU: $(CPPOBJS) $(NVCCOBJS) src/gpuCode.o
	$(CCPP) $(CPPFLAGS) -o testGeneralImageNetwork.out  $^ $(LINKFLAGS) $(NVLINKFLAGS) $(MAGICKLINKFLAGS)

src/gpuCode.o: $(NVCCOBJS)
	$(NVCC) --device-link -o $@ $^ $(LINKFLAGS) $(NVLINKFLAGS) $(CPPOBJS)

src/generateInput.o: $(INCLUDE)../Image_Manipulation/generate/generateInput.cpp $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(CCPP) $(CPPFLAGS) $(CFLAGS) -c $< -o $@ `Magick++-config --cxxflags --cppflags`

src/Matrix3D.o: $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cu
	$(NVCC) $(NVFLAGS) $< -o $@

src/activationFunctions.o: $(INCLUDE)artificialIntelligence/functions/activationFunctions.cu $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

src/BasicWeight.o: $(INCLUDE)artificialIntelligence/classes/weights/BasicWeight.cu $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

src/BasicLayer.o: $(INCLUDE)artificialIntelligence/classes/layers/BasicLayer.cu $(INCLUDE)artificialIntelligence/classes/weights/BasicWeight.cuh $(INCLUDE)artificialIntelligence/functions/activationFunctions.cuh $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

src/BasicLayerList.o: $(INCLUDE)artificialIntelligence/classes/layerLists/BasicLayerList.cu $(INCLUDE)artificialIntelligence/classes/layers/BasicLayer.cuh $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

src/testGeneralImageNetwork.o: $(INCLUDE)artificialIntelligence/neuralNetworks/testGeneralImageNetwork.cu $(CFLAGS) $(INCLUDE)artificialIntelligence/classes/layerLists/BasicLayerList.cuh $(INCLUDE)artificialIntelligence/classes/layers/BasicLayer.cuh $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

clean1: 
	rm -f src/*.o ./testGeneralImageNetwork.out

clean: 
	rm -f ./testGeneralImageNetwork.out