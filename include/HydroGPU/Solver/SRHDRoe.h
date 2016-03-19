#pragma once

#include "HydroGPU/Solver/SelfGravitationBehavior.h"
#include "HydroGPU/Solver/Roe.h"

namespace HydroGPU {
namespace Solver {

/*
Roe solver for Euler equations
TODO to re-enable SelfGravitationBehavior, get SRHD equation working with selfgrav by renaming STATE_REST_MASS_DENSITY to STATE_DENSITY
*/
struct SRHDRoe : public Roe/*SelfGravitationBehavior<Roe>*/ {
protected:
	typedef Roe/*SelfGravitationBehavior<Roe>*/ Super;

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
	virtual void setupConvertToTexKernelArgs();

	/*
	super writes the newtonian state variables
	this function then deduces primitives, stores them separately,
	 deduces state variables and stores them.
	*/
	virtual void resetState();		//called third
public:
	virtual std::string name() const { return "SRHDRoe"; }
};

}
}
