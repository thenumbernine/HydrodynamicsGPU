#pragma once

#include "HydroGPU/Solver/SelfGravitationBehavior.h"
#include "HydroGPU/Solver/MHDRemoveDivergenceBehavior.h"
#include "HydroGPU/Solver/FiniteVolumeSolver.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

struct MHDBurgers : public MHDRemoveDivergenceBehavior<SelfGravitationBehavior<FiniteVolumeSolver>> {
	typedef MHDRemoveDivergenceBehavior<SelfGravitationBehavior<FiniteVolumeSolver>> Super;

protected:
	cl::Buffer interfaceVelocityBuffer;
	cl::Buffer interfaceMagneticFieldBuffer;
	cl::Buffer pressureBuffer;

	cl::Kernel calcCellTimestepKernel;
	cl::Kernel calcInterfaceVelocityKernel;
	cl::Kernel calcVelocityFluxKernel;
	cl::Kernel calcInterfaceMagneticFieldKernel;
	cl::Kernel calcMagneticFieldFluxKernel;
	cl::Kernel computePressureKernel;
	cl::Kernel diffuseMomentumKernel;
	cl::Kernel diffuseWorkKernel;

	//matches MHDRoe -- belongs in the MHDEquation class maybe?
	cl::Kernel initVariablesKernel;

public:
	using Super::Super;
	virtual void initBuffers();
	virtual void initKernels();

protected:	
	virtual void createEquation();
	virtual std::vector<std::string> getProgramSources();
	
	virtual real calcTimestep();
	virtual void step(real dt);
	virtual void advectVelocity(real dt);
	virtual void advectMagneticField(real dt);
	virtual void diffusePressure(real dt);
	virtual void diffuseWork(real dt);
public:
	virtual std::string name() const { return "MHDBurgers"; }
};

}
}
