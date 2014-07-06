#pragma once

#include "HydroGPU/Roe.h"

struct HydroGPU;

/*
Roe solver for Euler equations
*/
struct SRHDRoe : public Roe {
	typedef Roe Super;
	
	SRHDRoe(HydroGPUApp& app);

protected:
	virtual std::vector<std::string> getProgramSources();

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

	virtual void init();			//called first
	virtual void initKernels();		//called second

	/*
	super writes the newtonian state variables
	this function then deduces primitives, stores them separately,
	 deduces state variables and stores them.
	*/
	virtual void resetState();		//called third
};

