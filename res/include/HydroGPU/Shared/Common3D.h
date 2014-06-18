#pragma once

#include "HydroGPU/Shared/Common.h"

#ifdef __OPENCL_VERSION__
#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
#endif

#define INDEX(a,b,c)	((a) + size.x * ((b) + size.y * (c)))
#define INDEXV(i)		INDEX((i).x, (i).y, (i).z)

