#pragma once

#include "HydroGPU/Solver/EulerHLL.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

struct MHDHLLC : public EulerHLL {
	typedef EulerHLL Super;
	MHDHLLC(HydroGPUApp&);

protected:
	virtual std::string getFluxSource();
};

}
}

