# CUDA_FLAGS=--gpu-architecture=sm_50 --relocatable-device-code=true
CUDA_FLAGS=--gpu-architecture=sm_50 -I../include
CFLAGS=-Wall
NVCC=/usr/local/cuda/bin/nvcc $(CUDA_FLAGS) --compiler-options $(CFLAGS)

HEADERS=../include/warpspeed.cuh
BINARIES=warpstream

%.o: %.cu $(HEADERS)
	$(NVCC) -o $@ -c $<

binaries : $(BINARIES)

test : $(BINARIES)
	./warpstream | dieharder -a -g 200

warpstream : warpstream.o
	$(NVCC) $^ -o $@

clean :
	rm -rf $(BINARIES) *.o *~

