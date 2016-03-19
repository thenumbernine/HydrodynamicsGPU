#pragma once

#include "HydroGPU/Solver/SelfGravitationBehavior.h"
#include "HydroGPU/Solver/HLL.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

struct EulerHLL : public SelfGravitationBehavior<HLL> {
	typedef SelfGravitationBehavior<HLL> Super;
	using Super::Super;
public:
	virtual void init();
protected:
	virtual void createEquation();
	virtual std::string getFluxSource();
	virtual void step(real dt);
public:
	virtual std::string name() const { return "EulerHLL"; }
};

}
}
