#pragma once

#include "HydroGPU/Solver/HLL.h"
#include "HydroGPU/Solver/MHDRemoveDivergence.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

struct MHDHLLC : public HLL {
	typedef HLL Super;

protected:
	std::shared_ptr<MHDRemoveDivergence> divfree;

public:
	using Super::Super;
	virtual void init();
	
protected:
	virtual void createEquation();
	virtual std::string getFluxSource();
	std::vector<std::string> getProgramSources();
	virtual void step();
};

}
}

