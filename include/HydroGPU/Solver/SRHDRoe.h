#pragma once

#include "HydroGPU/Solver/Roe.h"

namespace HydroGPU {
namespace Solver {

/*
Roe solver for Euler equations
*/
struct SRHDRoe : public Roe {
protected:
	typedef Roe Super;

	/*
	The primitive buffer holds density, velocity, pressure
	It is used as the initial guess for Newton for root-finding 
	when calculating the primitives from the state variables.
	*/
	cl::Buffer primitiveBuffer;

	/*
	performs initial conversion from Newtonian Euler equation state variables
	to both the primitives and the state variables of SRHD
	*/
	cl::Kernel initVariablesKernel;

public:
	using Super::Super;

protected:
	virtual void createEquation();
	virtual std::vector<std::string> getProgramSources();

	virtual void init();			//called last 

	/*
	super writes the newtonian state variables
	this function then deduces primitives, stores them separately,
	 deduces state variables and stores them.
	*/
	virtual void resetState();		//called third
};

}
}

