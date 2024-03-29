CC = gcc
CCPP = g++
NVCC = /usr/local/cuda-11.8/bin/nvcc

INCLUDE = ../../
DEFINES = -DUSE_ZLIB

OPTFLAGS = -Og -g -std=c++2a -Wall

NVFLAGS = -I $(INCLUDE) -I $(INCLUDE)../ -Xcompiler -Og -g -L/usr/local/cuda-11.8/lib64 -lcuda -lstdc++ -lcudart -dc --std=c++17
CPPFLAGS = $(OPTFLAGS) -I $(INCLUDE) -I $(INCLUDE)../ $(DEFINES)

MAGICKLINKFLAGS = `Magick++-config --ldflags --libs`
NVLINKFLAGS = -L/usr/local/cuda-11.8/lib64 -lcuda -lcudart
LINKFLAGS = -lz 

NVCCOBJS = src/testBasic.o \
	src/LayerList.o \
	src/BasicLayer.o \
	src/BasicWeight.o \
	src/ConvWeight.o \
	src/ConvLayer.o \
	src/PoolWeight.o \
	src/PoolLayer.o \
	src/activationFunctions.o \
	src/Matrix3D.o

CPPOBJS = 

all: linkCPU

linkCPU: $(CPPOBJS) $(NVCCOBJS) src/gpuCode.o
	$(CCPP) $(CPPFLAGS) -o testBasic.out  $^ $(LINKFLAGS) $(NVLINKFLAGS) $(MAGICKLINKFLAGS)

src/gpuCode.o: $(NVCCOBJS)
	$(NVCC) --device-link -o $@ $^ $(LINKFLAGS) $(NVLINKFLAGS) $(CPPOBJS)
	
src/Matrix3D.o: $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cu
	$(NVCC) $(NVFLAGS) $< -o $@

src/activationFunctions.o: $(INCLUDE)artificialIntelligence/functions/activationFunctions.cu $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

src/BasicWeight.o: $(INCLUDE)artificialIntelligence/classes/weights/BasicWeight.cu $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

src/BasicLayer.o: $(INCLUDE)artificialIntelligence/classes/layers/BasicLayer.cu $(INCLUDE)artificialIntelligence/classes/weights/BasicWeight.cuh $(INCLUDE)artificialIntelligence/functions/activationFunctions.cuh $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

src/PoolWeight.o: $(INCLUDE)artificialIntelligence/classes/weights/PoolWeight.cu $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

src/PoolLayer.o: $(INCLUDE)artificialIntelligence/classes/layers/PoolLayer.cu $(INCLUDE)artificialIntelligence/functions/activationFunctions.cuh $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

src/ConvWeight.o: $(INCLUDE)artificialIntelligence/classes/weights/ConvWeight.cu $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

src/ConvLayer.o: $(INCLUDE)artificialIntelligence/classes/layers/ConvLayer.cu $(INCLUDE)artificialIntelligence/classes/weights/ConvWeight.cuh $(INCLUDE)artificialIntelligence/functions/activationFunctions.cuh $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

src/LayerList.o: $(INCLUDE)artificialIntelligence/classes/layerLists/LayerList.cu $(INCLUDE)artificialIntelligence/classes/layers/BasicLayer.cuh $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

src/testBasic.o : $(INCLUDE)testing/testBasic/testBasic.cu  $(INCLUDE)artificialIntelligence/basicLearningTypes/generationalAIBasic.cu $(INCLUDE)artificialIntelligence/classes/layerLists/LayerList.cuh $(INCLUDE)artificialIntelligence/basicLearningTypes/generationalAIBasic.cuh $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

clean1: 
	rm -f src/*.o ./testBasic.out

clean: 
	rm -f src/B* ./testBasic.out