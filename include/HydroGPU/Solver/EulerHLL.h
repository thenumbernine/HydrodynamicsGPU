#pragma once

#include "HydroGPU/Solver/SelfGravitationBehavior.h"
#include "HydroGPU/Solver/HLL.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

struct EulerHLL : public SelfGravitationBehavior<HLL> {
	using Super = SelfGravitationBehavior<HLL>;
	using Super::Super;
protected:
	virtual void initKernels();
	virtual void createEquation();
	virtual std::string getFluxSource();
	virtual void step(real dt);
public:
	virtual std::string name() const { return "EulerHLL"; }
};

}
}
