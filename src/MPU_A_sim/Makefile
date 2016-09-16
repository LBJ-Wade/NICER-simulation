CUDA_FLAGS=--gpu-architecture=sm_50 --resource-usage -I../include
CFLAGS=-Wall
NVCC=/usr/local/cuda/bin/nvcc $(CUDA_FLAGS) --compiler-options $(CFLAGS)

HEADERS=../include/warpspeed.cuh
BINARIES=MPU_A_sim

%.o: %.cu $(HEADERS)
	$(NVCC) -o $@ -c $<

binaries : $(BINARIES)

test : $(BINARIES)
	time ./MPU_A_sim

MPU_A_sim : MPU_A_sim.o
	$(NVCC) $^ -o $@

clean :
	rm -rf $(BINARIES) *.o *~
