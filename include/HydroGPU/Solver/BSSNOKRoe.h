#pragma once

#include "HydroGPU/Solver/Roe.h"

namespace HydroGPU {
namespace Solver {

/*
Roe solver for BSSNOK equations
*/
struct BSSNOKRoe : public Roe {
	using Super = Roe;
	using Super::Super;
protected:
	virtual void createEquation();
	virtual std::vector<std::string> getProgramSources();
public:
	virtual std::string name() const { return "BSSNOKRoe"; }
};

}
}
