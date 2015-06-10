#pragma once

#include "HydroGPU/Solver/SelfGravitationBehavior.h"
#include "HydroGPU/Solver/MHDRemoveDivergenceBehavior.h"
#include "HydroGPU/Solver/HLL.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

struct MHDHLLC : public MHDRemoveDivergenceBehavior<SelfGravitationBehavior<HLL>> {
	typedef MHDRemoveDivergenceBehavior<SelfGravitationBehavior<HLL>> Super;

public:
	using Super::Super;
	virtual void init();
	
protected:
	virtual void createEquation();
	virtual std::string getFluxSource();
	virtual void step(real dt);
};

}
}

