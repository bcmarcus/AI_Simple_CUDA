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

NVCCOBJS = src/testMatrix3D.o \
	src/BasicLayerList.o \
	src/BasicLayer.o \
	src/BasicWeight.o \
	src/activationFunctions.o \
	src/Matrix3D.o

all: program

program: $(NVCCOBJS) src/gpuCode.o
	$(CCPP) $(CPPFLAGS) -o testMatrix3D.out  $^ $(LINKFLAGS) $(NVLINKFLAGS) $(MAGICKLINKFLAGS)

src/gpuCode.o: $(NVCCOBJS)
	$(NVCC) --device-link -o $@ $^ $(LINKFLAGS) $(NVLINKFLAGS) $(CPPOBJS)

src/Matrix3D.o: $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/activationFunctions.o: $(INCLUDE)artificialIntelligence/functions/activationFunctions.cu $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/BasicWeight.o: $(INCLUDE)artificialIntelligence/classes/weights/BasicWeight.cu $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/BasicLayer.o: $(INCLUDE)artificialIntelligence/classes/layers/BasicLayer.cu $(INCLUDE)artificialIntelligence/classes/weights/BasicWeight.cuh $(INCLUDE)artificialIntelligence/functions/activationFunctions.cuh $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/BasicLayerList.o: $(INCLUDE)artificialIntelligence/classes/layerLists/BasicLayerList.cu $(INCLUDE)artificialIntelligence/classes/layers/BasicLayer.cuh $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/testMatrix3D.o : $(INCLUDE)testing/testMatrix3D/testMatrix3D.cu  $(INCLUDE)artificialIntelligence/basicLearningTypes/generationalAIBasic.cu $(INCLUDE)artificialIntelligence/classes/layerLists/BasicLayerList.cuh $(INCLUDE)artificialIntelligence/basicLearningTypes/generationalAIBasic.cuh $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) -c $< -o $@

clean: 
	rm -f src/*.o program