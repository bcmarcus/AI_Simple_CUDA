NVCC = /usr/bin/nvcc

INCLUDE = ../../

NVFLAGS = -I $(INCLUDE) -Xcompiler -O2 -L/usr/local/cuda-11.6/lib64 -lcuda -lstdc++ -lcudart

all: program

program: testSummation.cu
	$(NVCC) $(NVFLAGS) -o testSummation.out  $^

clean: 
	rm -f src/*.o program