#pragma once

#include "HydroGPU/Boundary/Boundary.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/Exception.h"

namespace HydroGPU {
namespace Equation {

struct SelfGravitationInterface {
	virtual int gravityGetBoundaryKernelForBoundaryMethod(int dim, int minmax) = 0;
};

template<typename Super_>
struct SelfGravitationBehavior : public Super_, public SelfGravitationInterface {
	using Super = Super_;
	using Super::Super;
	virtual int gravityGetBoundaryKernelForBoundaryMethod(int dim, int minmax);
};

template<typename Super_>
int SelfGravitationBehavior<Super_>::gravityGetBoundaryKernelForBoundaryMethod(int dim, int minmax) {
	switch (Super::app->boundaryMethods(dim, minmax)) {
	case BOUNDARY_METHOD_NONE:
		return BOUNDARY_KERNEL_NONE;
	case BOUNDARY_METHOD_PERIODIC:
		return BOUNDARY_KERNEL_PERIODIC;
	case BOUNDARY_METHOD_MIRROR:
		return BOUNDARY_KERNEL_FREEFLOW;
	case BOUNDARY_METHOD_FREEFLOW:
		return BOUNDARY_KERNEL_FREEFLOW;
	}
	throw Common::Exception() << "got an unknown boundary method " << Super::app->boundaryMethods(dim, minmax) << " for dim " << dim;
}

}
}
