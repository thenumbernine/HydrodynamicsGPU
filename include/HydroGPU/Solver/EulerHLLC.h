#pragma once

#include "HydroGPU/Solver/EulerHLL.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

struct EulerHLLC : public EulerHLL {
	typedef EulerHLL Super;
	using Super::Super;

protected:
	virtual std::string getFluxSource();
public:
	virtual std::string name() const { return "EulerHLLC"; }
};

}
}
