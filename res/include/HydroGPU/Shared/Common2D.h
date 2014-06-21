#pragma once

#include "HydroGPU/Shared/Common.h"

#define INDEX(a,b)		((a) + SIZE_X * (b))
#define INDEXV(i)		INDEX((i).x, (i).y)

