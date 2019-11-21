#pragma once

#if 1	//double

#ifdef __OPENCL_VERSION__
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

typedef double real;
#ifdef __OPENCL_VERSION__
typedef double2 real2;
typedef double4 real4;
typedef double8 real8;
typedef double16 real16;
#else
#if PLATFORM_OSX
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
typedef cl_double2 real2;
typedef cl_double4 real4;
typedef cl_double8 real8;
typedef cl_double16 real16;
#endif

#else	//single

typedef float real;
#ifdef __OPENCL_VERSION__
typedef float2 real2;
typedef float4 real4;
typedef float8 real8;
typedef float16 real16;
#else

//same as in CLCommon/CLCommon.h
//and similar to src/HydroGPUApp.cpp
#if PLATFORM_OSX
#include <OpenCL/cl.h>
#elif PLATFORM_MSVC
#include "CL/cl.h"
#else
#include <CL/cl.h>
#endif

typedef cl_float2 real2;
typedef cl_float4 real4;
typedef cl_float8 real8;
typedef cl_float16 real16;
#endif

#endif	//single/double

#ifdef __OPENCL_VERSION__
#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
#endif

#define INDEX(a,b,c)	((a) + SIZE_X * ((b) + SIZE_Y * (c)))
#define INDEXV(i)		INDEX((i).x, (i).y, (i).z)
