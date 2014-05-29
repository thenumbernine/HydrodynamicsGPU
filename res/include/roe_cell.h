#pragma once

#if 0
#ifdef __OPENCL_VERSION__
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#endif

typedef float real;
#ifdef __OPENCL_VERSION__
typedef float2 real2;
typedef float4 real4;
#else
typedef cl_float2 real2;
typedef cl_float4 real4;
#endif

#define DIM	2
#define NUM_STATES	2+DIM
#define GAMMA 1.4f

struct Interface {
	//Roe-specific values
	real4 eigenvalues;
	real4 eigenvectors[NUM_STATES];			//stored row-major 
	real4 eigenvectorsInverse[NUM_STATES];	// so math matches array index notation
	real4 rTilde;
	real4 deltaQTilde;
	
	//base cell values
	real2 x;
	real4 flux;
	bool solid;
};
typedef struct Interface Interface;

struct Cell {
	//base cell values
	real4 q;
	real2 x;

	Interface interfaces[DIM];
};
typedef struct Cell Cell;

