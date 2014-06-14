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
typedef float8 real8;
typedef float16 real16;
#else
#include <OpenCL/cl.h>
typedef cl_float2 real2;
typedef cl_float4 real4;
typedef cl_float8 real8;
typedef cl_float16 real16;
#endif

#define DIM	2
#define NUM_GHOST_CELLS	2
#define NUM_STATES	2+DIM

enum {
	DISPLAY_DENSITY,
	DISPLAY_VELOCITY,
	DISPLAY_PRESSURE,
	DISPLAY_GRAVITY_POTENTIAL,
	NUM_DISPLAY_METHODS
};

enum {
	BOUNDARY_PERIODIC,
	BOUNDARY_MIRROR,
	BOUNDARY_FREEFLOW,
	NUM_BOUNDARY_METHODS
};

#ifdef __OPENCL_VERSION__

//I use the 'size.x' parameter instead of get_global_size(0) because kernels like the boundary update kernel have different global sizes than the buffer size 

//these are for dereferences 0..size that take place in a ghost-padded buffer (0...size+2*nghost)
#define INDEXGHOST(ix,iy)	(((ix) + NUM_GHOST_CELLS) + (size.x + 2 * NUM_GHOST_CELLS) * ((iy) + NUM_GHOST_CELLS))
#define INDEXGHOSTV(i)		INDEXGHOST((i).x, (i).y)

//these are for dereferences 0..size that take place in 0..size buffers
#define INDEX(ix,iy)		((ix) + size.x * (iy))
#define INDEXV(i)			INDEX((i).x, (i).y)

#endif

