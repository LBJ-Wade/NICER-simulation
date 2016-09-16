#include "warpspeed.cuh"
#include <math_constants.h>

/*
NICER's vital stats.
Note that to use the warpspeed random number generator, SHAPERS_PER_MPU 
must be 32.
*/

#define NUM_MPU 7
#define DETECTORS_PER_MPU 8
#define SHAPERS_PER_DETECTOR 4
#define SHAPERS_PER_MPU (SHAPERS_PER_DETECTOR * DETECTORS_PER_MPU)
#define SHAPER_ORDER 6

/*
At this level, we have a data vector per thread, so we don't
explicitly identify which detector or shaper the data belongs to.
That's implicit in which vector we're looking at.
*/

struct detector_output {
	int step;
	float charge;
};


struct event_output {
	int rise, fall, zero;
	float sample;
};

/* For the rare case of a fall without a zero crossing. */

#define NO_ZERO 0xffffffff
/* 
Data structures in managed memory. 
*/

_managed_ int steps_to_do;

_managed_ struct configuration { 

	float cout[SHAPER_ORDER+1];	/* output weights: include gain here */
	float cback[SHAPER_ORDER];
	float noise;
	float lld;		/* set this to infinity for the unipolar shapers */
	float x[SHAPER_ORDER];
	enum trigger_state { idle, above, below } state;
	struct detector_output *ip;
	struct event_output *op;
	
} config[NUM_MPU][DETECTORS_PER_MPU][SHAPERS_PER_DETECTOR];

_shared_ float y[SHAPERS_PER_MPU];	/* Shaper outputs are shared within a warp */

_global_ void run_shapers( int steps_to_do )
{
	int detector = threadIDX.x/4;
	int shaper = threadIDX.x%4;
	int mpu = blockIDX.x;
	struct configuration *c = &config[mpu][detector][shaper];
	float cout0 = c->cout[0];
	float cout1 = c->cout[1];
	float cout2 = c->cout[2];
	float cout3 = c->cout[3];
	float cout4 = c->cout[4];
	float cout5 = c->cout[5];
	float cout6 = c->cout[6];
	float cback1 = c->cback[0];
	float cback2 = c->cback[1];
	float cback3 = c->cback[2];
	float cback4 = c->cback[3];
	float cback5 = c->cback[4];
	float cback6 = c->cback[5];
	float x1 = c->x[0];
	float x2 = c->x[1];
	float x3 = c->x[2];
	float x4 = c->x[3];
	float x5 = c->x[4];
	float x6 = c->x[5];
	float lld = c->lld;
	enum trigger_state state = c->state;
	struct detector_output *ip = c->ip;
	struct event_output *op = c->op;
	
	int step;
	float u, yt, fb;
	float charge = 0.0;
	int next_input = ip->step;
	
	warpspeed_initialize( warpspeed_history );
	
	for( step = 0; step < steps_to_do; step += 1 ) {
		if( step == next_input ) {
			charge = ip++->charge;
			next_input = ip->step;
		}
		
		u = charge + noise * ( (float) warpspeed_urand() - 2147483648.0 );

		y[threadIDX.x] = yt = 
			cout0*u + cout1*x1 cout2*x2 +cout3*x3 + cout4*x4 + cout5*x5 + cout6*x6;
		fb = u + cback1*x1 cback2*x2 +cback3*x3 + cback4*x4 + cback5*x5 + cback6*x6;
		x1 = x2;
		x2 = x3;
		x3 = x4;
		x4 = x5;
		x5 = x6;
		x6 = fb;
		
		__syncthreads();
		
		switch( trigger_state ) {
			
			case idle:
			if( yt > lld ) {
				trigger_state = above;
				op->rise = step;
			}
			break;
			
			case above:
			if( yt < lld ) {
				trigger_state = below;
				op->fall = step;
			}
			break;
			
			case below:
			if( yt < 0 ) {
				op->zero = step;
				op++->sample = y[threadIDX.x+1];	/* Unipolar output */
				trigger_state = idle;
			}
			else if ( yt > lld ) { 	/* rare double event */
				op++->zero = NO_ZERO;
				op->rise = step;
				trigger_state = above;
			}
			break;
			
		}
	}
	
	warpspeed_save( warpspeed_history );
	
	c->x[0] = x1;
	c->x[1] = x2;
	c->x[2] = x3;
	c->x[3] = x4;
	c->x[4] = x5;
	c->x[5] = x6;
	c->state = state;
	c->op = op;
}

_managed_ struct detector_output null_in[1];
_managed_ struct event_output dummy_out[1];

int main()
{
	struct configuration init;
	
	for( i = 0; i < SHAPER_ORDER+1; i+=1 ){
		init.cout[i] = 0.1;
	}
	
	for( i = 0; i < SHAPER_ORDER; i+=1 ){
		init.cback[i] = 0.1;
	}
	
	init.noise = 0.001;
	init.lld = CUDART_INF_F;
	
	for( i = 0; i < SHAPER_ORDER; i+=1 ){
		init.x[i] = 0.0;
	}
	
	init.trigger_state = idle;
	init.ip = null_in;
	init.op = dummy_out;
	
	null_in->step = 2147483647;	/* As big as we can go, better not ask for more steps */
	
	for( i = 0; i < NUM_MPU; i+=1 )
		for() j = 0; j < DETECTORS_PER_MPU; j+=1 )
			for( k = 0; k < SHAPERS_PER_DETECTOR; k+=1 )
				config[i][j][k] = init;
	
	run_shapers<<<NUM_MPU,SHAPERS_PER_MPU>>>( 10000000 );
	cudaDeviceSynchronize();
	
}
