#pragma once

#include "HydroGPU/Solver/FiniteVolumeSolver.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

/*
General Roe solver
subclasses need to implement calcEigenBasis
*/
struct Roe : public FiniteVolumeSolver {
protected:
	using Super = FiniteVolumeSolver;

	cl::Buffer eigenvaluesBuffer;
	cl::Buffer eigenvectorsBuffer;	//contains forward and inverse transform information
	cl::Buffer deltaQTildeBuffer;
	
	cl::Kernel calcEigenBasisKernel;
	cl::Kernel calcCellTimestepKernel;
	cl::Kernel calcDeltaQTildeKernel;

public:
	Roe(HydroGPUApp* app);
	virtual void init();
protected:
	virtual void initBuffers();
	virtual void initKernels();
	virtual std::vector<std::string> getProgramSources();
	virtual std::vector<std::string> getEigenProgramSources();
	virtual int getEigenSpaceDim();	//numStates() in most cases
	virtual int getEigenTransformStructSize();	//total size of forward and inverse
	virtual real calcTimestep();
	virtual void initFlux();
	virtual void step(real dt);
	virtual void calcDeriv(cl::Buffer derivBuffer, real dt);
	virtual void calcFlux(real dt);
};

}
}
