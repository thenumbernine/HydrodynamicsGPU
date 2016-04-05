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

	/*
	State Update:
	make sure D (relativistic density) and tau (relativistic total energy) don't go negative
	
	Primitive Update:
	Performs the Newton root-finding on the pressure variable to update primtivies each iteration
	...or in the case of multi-stage, multiple times per iteration ...
	*/
	cl::Kernel updatePrimitivesKernel;
	cl::Kernel constrainStateKernel;

public:
	using Super::Super;

protected:
	virtual void createEquation();
	virtual std::vector<std::string> getProgramSources();

	virtual void initBuffers();
	virtual void initKernels();
	virtual void setupConvertToTexKernelArgs();

	/*
	super writes the newtonian state variables
	this function then deduces primitives, stores them separately,
	 deduces state variables and stores them.
	*/
	virtual void resetState();		//called third

	virtual void step(real dt);
public:
	virtual std::string name() const { return "SRHDRoe"; }
};

}
}
