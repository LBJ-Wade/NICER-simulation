/*
Mathematica's CUDALink wraps user source in extern "C" {}
This causes trouble with headers that have C++ constructs in them.
Use the fact that CUDALink defines USING_CUDA_FUNCTION to repair this.
*/

#if USING_CUDA_FUNCTION
}
#endif

// Put any includes of C++isms here.

#include <curand_kernel.h>

#if USING_CUDA_FUNCTION
extern "C" {
#endif

#define SETTLE 200

__global__ void events ( 
	double *ucout, double *ucback, 	// unipolar shaper parameters
	double *bcout, double *bcback, 	// bipolar shaper parameters
	double noise,			// electrons/step
	int when,			// step at which pulse happens
	int pulse,			// pulse height, electrons
	int steps, 			// steps of simulation
	double thresh,			// trigger threshold
	double *ft,			// forced trigger sample
	int *tz,			// time of zero crossing
	double *ph			// pulse height
	) {
	
	int thread = blockDim.x * blockIdx.x + threadIdx.x;
	curandState_t curand_state;
	double ucout0 = ucout[0];
	double ucout1 = ucout[1];
	double ucout2 = ucout[2];
	double ucout3 = ucout[3];
	double ucout4 = ucout[4];
	double ucout5 = ucout[5];
	double ucout6 = ucout[6];
	double ucback1 = ucback[0];
	double ucback2 = ucback[1];
	double ucback3 = ucback[2];
	double ucback4 = ucback[3];
	double ucback5 = ucback[4];
	double ucback6 = ucback[5];
	double ux1 = 0.0;
	double ux2 = 0.0;
	double ux3 = 0.0;
	double ux4 = 0.0;
	double ux5 = 0.0;
	double ux6 = 0.0;
	double uy, ufb;
	double bcout0 = bcout[0];
	double bcout1 = bcout[1];
	double bcout2 = bcout[2];
	double bcout3 = bcout[3];
	double bcout4 = bcout[4];
	double bcout5 = bcout[5];
	double bcout6 = bcout[6];
	double bcback1 = bcback[0];
	double bcback2 = bcback[1];
	double bcback3 = bcback[2];
	double bcback4 = bcback[3];
	double bcback5 = bcback[4];
	double bcback6 = bcback[5];
	double bx1 = 0.0;
	double bx2 = 0.0;
	double bx3 = 0.0;
	double bx4 = 0.0;
	double bx5 = 0.0;
	double bx6 = 0.0;
	double by, bfb;
	
	int trigger = 0;
	
	int i;

	curand_init ( 100951,
		thread,
		0,
		& curand_state );

	for ( i = -SETTLE; i < steps; i += 1) {
		
		double u = 0;		// input
		if ( i == when ) u = pulse;
		u += curand_poisson ( & curand_state, noise );

		uy = ucout0*u + ucout1*ux1 + ucout2*ux2 + ucout3*ux3 +
			ucout4*ux4 + ucout5*ux5 + ucout6*ux6;

		ufb = u + ucback1*ux1 + ucback2*ux2 + ucback3*ux3 + 
			ucback4*ux4 + ucback5*ux5 + ucback6*ux6;

		ux1 = ux2;
		ux2 = ux3;
		ux3 = ux4;
		ux4 = ux5;
		ux5 = ux6;
		ux6 = ufb;

		by = bcout0*u + bcout1*bx1 + bcout2*bx2 + bcout3*bx3 +
			bcout4*bx4 + bcout5*bx5 + bcout6*bx6;

		bfb = u + bcback1*bx1 + bcback2*bx2 + bcback3*bx3 + 
			bcback4*bx4 + bcback5*bx5 + bcback6*bx6;

		bx1 = bx2;
		bx2 = bx3;
		bx3 = bx4;
		bx4 = bx5;
		bx5 = bx6;
		bx6 = bfb;
		
		if( i == when - 1 ) ft[ thread ] = uy;
		
		if ( i >= 0 ) {
			if( trigger && by < 0.0 ) {
				tz[ thread ] = i;
				ph[ thread] = uy;
				trigger = 0;
			} else if( by > thresh ) trigger = 1;
		}
	}
}
 
