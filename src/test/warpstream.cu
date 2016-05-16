#include <unistd.h>
#include <stdio.h>
#include "warpspeed.cuh"

#define OUTPUT 65536	// output per block
#define BLOCKS 16	// blocks running in parallel

__managed__ unsigned int output[ BLOCKS * OUTPUT ];



__global__ void generate_output( unsigned int * state )
{
	int i;
	int offset = threadIdx.x + OUTPUT * blockIdx.x;
	
	warpspeed_initialize( state );
	
	for( i = 0; i < OUTPUT; i += WARP ) {
		output[ i + offset ] = warpspeed_urand();
	}
	
	warpspeed_save( state );
}
	
	


int main()
{
	unsigned int *random_state = warpspeed_seed( BLOCKS, 100951 );
	
	for(;;) {
		generate_output<<<BLOCKS,WARP>>>( random_state );
		cudaDeviceSynchronize();	// This waits for output
		write( 1, output, sizeof( output ));
	}
}
