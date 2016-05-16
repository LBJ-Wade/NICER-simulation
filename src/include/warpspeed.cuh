/*
Parallel linear lagged Fibonnaci pseudo-random number generator for CUDA.

warpspeed_urand() makes one warp's worth of unsigned integers in parallel,
one random integer per thread. It must be invoked in a block that contains
exactly one warp (32) threads, and all threads in a block must invoke it
synchonously (not conditionally in some threads). There's no restriction on
the number of blocks in the grid, nor must they synchronize.
*/

/* 
Put function definitions in this header 'cause separate compilation
is tricky in CUDA.
*/

#ifndef WARPSPEED_CUH
#define WARPSPEED_CUH

#define WARP 32
#define WARPSPEED_LENGTH (2*WARP)
#define TAP1 32
#define TAP2 43
#define TAP3 57

#include <stdlib.h>

/*
Create an initial history for a grid of generators. The parameter "warps"
is the number of warps in the grid: if that's variable, make it big enough
for the largest grid. The result is a pointer to a managed memory region,
usable on the host, passable to the grid.
*/

__host__ unsigned int *warpspeed_seed( unsigned int warps, unsigned int seed )
{
	unsigned int *initial_history;
	unsigned int i, j;
	
	cudaMallocManaged( &initial_history, 
		warps * WARPSPEED_LENGTH * sizeof( unsigned int ));
	srandom( seed );

/*
Use the stdlib random() function to create an initial faux history.

Since random() only makes 31 bit numbers, add a couple together with a shift
to make 32 bit numbers for initialization.

Each warp needs to have at least one odd number in its history,
so we force h[0] to be odd.
*/

	for( i = 0; i < warps; i += 1 ) {
		unsigned int *h = initial_history + i * WARPSPEED_LENGTH;
		
		h[0] = 1 + (random()<<1);
		for( j = 1; j < WARPSPEED_LENGTH; j += 1 ) {
			h[j] = random() + (random()<<1);
		}
	}
	
	return initial_history;
}


/*
Since shared memory has only the lifetime of its block (warp in this case)
we must copy the more persistent managed history to the block history when
we launch a block.
*/

__shared__ unsigned int warpspeed_history[64];

__device__ void warpspeed_initialize( unsigned int *initial_history )
{
	int block = blockIdx.x +
		blockIdx.y * gridDim.x +
		blockIdx.z * gridDim.x * gridDim.y;
	int offset = WARPSPEED_LENGTH * block + threadIdx.x;
	
	__syncthreads();
	warpspeed_history[ threadIdx.x ] = initial_history[ offset ];
	warpspeed_history[ threadIdx.x + WARP ] = initial_history[ offset + WARP ];
}


/*
When we end execution of a block, we must save the shared history
in the managed history if we intend to generate independent numbers
in future invocations.
*/

__device__ void warpspeed_save( unsigned int *initial_history )
{
	int block = blockIdx.x +
		blockIdx.y * gridDim.x +
		blockIdx.z * gridDim.x * gridDim.y;
	int offset = WARPSPEED_LENGTH * block + threadIdx.x;
	
	__syncthreads();
	initial_history[ offset ] = warpspeed_history[ threadIdx.x ];
	initial_history[ offset + WARP ] = warpspeed_history[ threadIdx.x + WARP ];
}

/*
The actual generator.
*/

__device__ unsigned int warpspeed_urand( void )
{
	int i = threadIdx.x;
	unsigned int current;
	
	__syncthreads();
	current = warpspeed_history[ i + TAP1 - WARP ]
		+ warpspeed_history[ i + TAP2 - WARP ]
		+ warpspeed_history[ i + TAP3 - WARP ]
		+ warpspeed_history[ i + WARPSPEED_LENGTH - WARP ];
	warpspeed_history[ i + WARP ] = warpspeed_history[ i ];
	warpspeed_history[ i ] = current;
	return current;
}

#endif /* ndef WARPSPEED_CUH */

