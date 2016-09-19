# GENCODE=-gencode arch=compute_30,code=sm_30 \
#	-gencode arch=compute_35,code=sm_35 \
#	-gencode arch=compute_37,code=sm_37 \
#	-gencode arch=compute_50,code=sm_50 \
#	-gencode arch=compute_52,code=sm_52 \
#	-gencode arch=compute_52,code=compute_52
GENCODE=-gencode arch=compute_50,code=sm_50
CUDA_FLAGS= $(GENCODE) --resource-usage -I../include
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
