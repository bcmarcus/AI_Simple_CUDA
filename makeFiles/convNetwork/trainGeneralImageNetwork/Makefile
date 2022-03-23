CC = gcc
CCPP = g++
NVCC = /usr/local/cuda-11.6/bin/nvcc

INCLUDE = ../../../
DEFINES = -DUSE_ZLIB

OPTFLAGS = -O2 -std=c++2a -Wall

NVFLAGS = -I $(INCLUDE) -Xcompiler -O2 -L/usr/local/cuda-11.6/lib64 -lcuda -lstdc++ -lcudart -dc -arch=sm_86 --std=c++17
NVLINKFLAGS = -L/usr/local/cuda-11.6/lib64 -lcuda -lcudart
#-lstdc++

CPPFLAGS = $(OPTFLAGS) -I $(INCLUDE) $(DEFINES)
LINKFLAGS = -lz 

MAGICKLINKFLAGS = `Magick++-config --ldflags --libs`

# objs = src/generationalAIBasic.o \
# 	src/BasicLayerList.o \
# 	src/BasicLayer.o \
# 	src/BasicWeight.o \
# 	src/activationFunctions.o 

NVCCOBJS = src/generationalAIBasic.o \
	src/cudaErrors.o \
	src/BasicLayerList.o \
	src/BasicLayer.o \
	src/BasicWeight.o \
	src/activationFunctions.o \
	src/Matrix3D.o

CPPOBJS = src/simpleMath.o \
	src/time.o \
	src/trainGeneralImageNetwork.o \
	src/generateInput.o

all: linkCPU

linkCPU: $(CPPOBJS) $(NVCCOBJS) src/gpuCode.o
	$(CCPP) $(CPPFLAGS) -o trainGeneralImageNetwork.out  $^ $(LINKFLAGS) $(NVLINKFLAGS) $(MAGICKLINKFLAGS)

src/gpuCode.o: $(NVCCOBJS)
	$(NVCC) --device-link -arch=sm_86 -o $@ $^ $(LINKFLAGS) $(NVLINKFLAGS) $(CPPOBJS)

src/simpleMath.o: $(INCLUDE)coreutils/functions/math/simpleMath.cpp 
	$(CCPP) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

src/time.o: $(INCLUDE)coreutils/util/time.cpp 
	$(CCPP) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

src/generateInput.o: $(INCLUDE)artificialIntelligence/functions/images/generateInput.cpp $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(CCPP) $(CPPFLAGS) $(CFLAGS) -c $< -o $@ `Magick++-config --cxxflags --cppflags`

src/cudaErrors.o: $(INCLUDE)coreutils/util/cudaErrors.cu $(INCLUDE)coreutils/util/cudaErrors.cuh 
	$(NVCC) $(NVFLAGS) $< -o $@

src/Matrix3D.o: $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cu
	$(NVCC) $(NVFLAGS) $< -o $@

src/activationFunctions.o: $(INCLUDE)artificialIntelligence/functions/activationFunctions.cu $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

src/BasicWeight.o: $(INCLUDE)artificialIntelligence/classes/BasicWeight.cu $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

src/BasicLayer.o: $(INCLUDE)artificialIntelligence/classes/BasicLayer.cu $(INCLUDE)artificialIntelligence/classes/BasicWeight.cuh $(INCLUDE)artificialIntelligence/functions/activationFunctions.cuh $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

src/BasicLayerList.o: $(INCLUDE)artificialIntelligence/classes/BasicLayerList.cu $(INCLUDE)artificialIntelligence/classes/BasicLayer.cuh $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

src/generationalAIBasic.o: $(INCLUDE)artificialIntelligence/basicLearningTypes/generationalAIBasic.cu  $(INCLUDE)artificialIntelligence/classes/BasicLayerList.hpp $(INCLUDE)artificialIntelligence/classes/BasicLayer.cuh $(INCLUDE)artificialIntelligence/functions/activationFunctions.cuh $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh 
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/trainGeneralImageNetwork.o : $(INCLUDE)artificialIntelligence/neuralNetworks/trainGeneralImageNetwork.cu  $(INCLUDE)artificialIntelligence/basicLearningTypes/generationalAIBasic.cu $(INCLUDE)artificialIntelligence/classes/BasicLayerList.hpp $(INCLUDE)artificialIntelligence/basicLearningTypes/generationalAIBasic.hpp $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

clean: 
	rm -f src/*.o trainGeneralImageNetwork.out