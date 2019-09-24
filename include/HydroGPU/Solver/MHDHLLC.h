#pragma once

#include "HydroGPU/Solver/SelfGravitationBehavior.h"
#include "HydroGPU/Solver/MHDRemoveDivergenceBehavior.h"
#include "HydroGPU/Solver/HLL.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

struct MHDHLLC : public MHDRemoveDivergenceBehavior<SelfGravitationBehavior<HLL>> {
	using Super = MHDRemoveDivergenceBehavior<SelfGravitationBehavior<HLL>>;

public:
	using Super::Super;
	virtual void init();
	
protected:
	virtual void createEquation();
	virtual std::string getFluxSource();
	virtual void step(real dt);
public:
	virtual std::string name() const { return "MHDHLLC"; }
};

}
}
