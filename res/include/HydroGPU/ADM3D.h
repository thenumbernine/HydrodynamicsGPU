#pragma once

#include "HydroGPU/Shared/Common.h"

real det3x3sym(real xx, real xy, real xz, real yy, real yz, real zz);
real8 inv3x3sym(real xx, real xy, real xz, real yy, real yz, real zz, real det);
