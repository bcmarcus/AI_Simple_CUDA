CC = gcc
CCPP = g++
NVCC = /usr/bin/nvcc

INCLUDE = ../../
DEFINES = -DUSE_ZLIB

OPTFLAGS = -O2 -std=c++2a -Wall

NVFLAGS = -I $(INCLUDE) -Xcompiler -O2 -L/usr/local/cuda-11/lib64 -lcuda -lstdc++ -lcudart
NVLINKFLAGS = -L/usr/local/cuda-11/lib64 -lcuda -lcudart 
#-lstdc++

CPPFLAGS = $(OPTFLAGS) -I $(INCLUDE) $(DEFINES) 
LINKFLAGS = -lz 

MAGICKLINKFLAGS = `Magick++-config --ldflags --libs`

objs = src/testMatrix3D.o \
	src/BasicLayerList.o \
	src/BasicLayer.o \
	src/BasicWeight.o \
	src/activationFunctions.o \
	src/simpleMath.o 


all: program

program: $(objs)
	$(CCPP) $(CPPFLAGS) -o testMatrix3D.out  $^ $(LINKFLAGS) $(NVLINKFLAGS) $(MAGICKLINKFLAGS)

src/simpleMath.o: $(INCLUDE)coreutils/functions/math/simpleMath.cpp 
	$(CCPP) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

src/activationFunctions.o: $(INCLUDE)artificialIntelligence/functions/activationFunctions.cpp $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cpp
	$(CCPP) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

src/BasicWeight.o: $(INCLUDE)artificialIntelligence/classes/BasicWeight.cu $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cpp
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/BasicLayer.o: $(INCLUDE)artificialIntelligence/classes/BasicLayer.cu $(INCLUDE)artificialIntelligence/classes/BasicWeight.hpp $(INCLUDE)artificialIntelligence/functions/activationFunctions.hpp $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cpp
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/BasicLayerList.o: $(INCLUDE)artificialIntelligence/classes/BasicLayerList.cu $(INCLUDE)artificialIntelligence/classes/BasicLayer.hpp $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cpp
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/testMatrix3D.o : $(INCLUDE)testing/testMatrix3D/testMatrix3D.cu  $(INCLUDE)artificialIntelligence/basicLearningTypes/generationalAIBasic.cu $(INCLUDE)artificialIntelligence/classes/BasicLayerList.hpp $(INCLUDE)artificialIntelligence/basicLearningTypes/generationalAIBasic.hpp $(INCLUDE)coreutils/classes/matrixes/Matrix3D.cpp
	$(NVCC) $(NVFLAGS) -c $< -o $@

clean: 
	rm -f src/*.o program