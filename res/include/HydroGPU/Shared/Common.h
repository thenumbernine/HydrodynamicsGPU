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

enum {
	DISPLAY_DENSITY,
	DISPLAY_VELOCITY,
	DISPLAY_PRESSURE,
	DISPLAY_MAGNETISM,
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
#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
#endif

#define INDEX(a,b,c)	((a) + SIZE_X * ((b) + SIZE_Y * (c)))
#define INDEXV(i)		INDEX((i).x, (i).y, (i).z)

