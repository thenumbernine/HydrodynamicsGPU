#pragma once

#include "HydroGPU/Shared/Common.h"

#define INDEX(a,b)		((a) + size.x * (b))
#define INDEXV(i)		INDEX((i).x, (i).y)

