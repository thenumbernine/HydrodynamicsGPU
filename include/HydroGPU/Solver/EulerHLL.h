#pragma once

#include "HydroGPU/Solver/HLL.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

struct EulerHLL : public HLL {
	typedef HLL Super;
	using Super::Super;
protected:
	virtual void createEquation();
	virtual std::string getFluxSource();
};

}
}

