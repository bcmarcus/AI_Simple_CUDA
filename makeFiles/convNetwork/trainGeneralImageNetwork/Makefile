CC = gcc
CCPP = g++
NVCC = /usr/bin/nvcc

INCLUDE = ../../../
DEFINES = -DUSE_ZLIB

OPTFLAGS = -O2 -std=c++2a -Wall

#-x c++ -Wno-deprecated-gpu-targets -ccbin g++
NVFLAGS = -I $(INCLUDE) -Xcompiler -O2 -L /usr/local/cuda-11.6/lib64 -lcuda -lstdc++
NVLINKFLAGS = -L /usr/local/cuda-11.6/lib64 -lcuda -lcudart
#-lstdc++

CPPFLAGS = $(OPTFLAGS) -I $(INCLUDE) $(DEFINES) -L /usr/local/cuda-11.6/lib64 -lcuda -lcudart
LINKFLAGS = -lz 

MAGICKLINKFLAGS = `Magick++-config --ldflags --libs`

objs = src/trainGeneralImageNetwork.o \
	src/generationalAIBasic.o \
	src/BasicLayerList.o \
	src/BasicLayer.o \
	src/BasicWeight.o \
	src/generateInput.o \
	src/activationFunctions.o \
	src/simpleMath.o

all: program

program: $(objs)
	$(CCPP) $(CPPFLAGS) -o trainGeneralImageNetwork.out  $^ $(LINKFLAGS) $(NVLINKFLAGS) $(MAGICKLINKFLAGS)

src/simpleMath.o: $(INCLUDE)coreutils/functions/math/simpleMath.cpp 
	$(CCPP) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

src/activationFunctions.o: $(INCLUDE)artificialIntelligence/functions/activationFunctions.cpp $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cpp
	$(CCPP) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

src/generateInput.o: $(INCLUDE)artificialIntelligence/functions/images/generateInput.cpp $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cpp
	$(CCPP) $(CPPFLAGS) $(CFLAGS) -c $< -o $@ `Magick++-config --cxxflags --cppflags`

src/BasicWeight.o: $(INCLUDE)artificialIntelligence/classes/BasicWeight.cu $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cpp
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/BasicLayer.o: $(INCLUDE)artificialIntelligence/classes/BasicLayer.cu $(INCLUDE)artificialIntelligence/classes/BasicWeight.hpp $(INCLUDE)artificialIntelligence/functions/activationFunctions.hpp $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cpp
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/BasicLayerList.o: $(INCLUDE)artificialIntelligence/classes/BasicLayerList.cu $(INCLUDE)artificialIntelligence/classes/BasicLayer.hpp $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cpp
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/generationalAIBasic.o: $(INCLUDE)artificialIntelligence/basicLearningTypes/generationalAIBasic.cu  $(INCLUDE)artificialIntelligence/classes/BasicLayerList.hpp $(INCLUDE)artificialIntelligence/classes/BasicLayer.hpp $(INCLUDE)artificialIntelligence/functions/activationFunctions.hpp $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cpp 
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/trainGeneralImageNetwork.o : $(INCLUDE)artificialIntelligence/neuralNetworks/trainGeneralImageNetwork.cpp  $(INCLUDE)artificialIntelligence/basicLearningTypes/generationalAIBasic.cu $(INCLUDE)artificialIntelligence/classes/BasicLayerList.hpp $(INCLUDE)artificialIntelligence/basicLearningTypes/generationalAIBasic.hpp $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cpp
	$(CCPP) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

#trainGeneralImageNetwork: objects.o
#	$(CCPP) -c --std=c++2a -I ../../../ `Magick++-config --cxxflags --cppflags` `Magick++-config --ldflags --libs` ../../../artificialIntelligence/functions/activationFunctions.cpp ../../../artificialIntelligence/functions/images/generateInput.cpp ../../../artificialIntelligence/neuralNetworks/trainGeneralImageNetwork.cpp

#objects.o:
#	$(NVCC) -c ../../../artificialIntelligence/classes/BasicWeight.cu ../../../artificialIntelligence/classes/BasicLayer.cu ../../../artificialIntelligence/classes/BasicLayerList.cu ../../../artificialIntelligence/basicLearningTypes/generationalAIBasic.cu -I ../../../ -L /usr/local/cuda/lib64 -lcuda -lstdc++

clean: 
	rm -f src/*.o program