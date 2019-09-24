#pragma once

#include "HydroGPU/Solver/EulerHLL.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

struct EulerHLLC : public EulerHLL {
	using Super = EulerHLL;
	using Super::Super;

protected:
	virtual std::string getFluxSource();
public:
	virtual std::string name() const { return "EulerHLLC"; }
};

}
}
