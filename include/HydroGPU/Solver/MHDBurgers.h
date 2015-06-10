#pragma once

#include "HydroGPU/Solver/SelfGravitationBehavior.h"
#include "HydroGPU/Solver/MHDRemoveDivergenceBehavior.h"
#include "HydroGPU/Solver/Solver.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

struct MHDBurgers : public MHDRemoveDivergenceBehavior<SelfGravitationBehavior<Solver>> {
	typedef MHDRemoveDivergenceBehavior<SelfGravitationBehavior<Solver>> Super;

protected:
	cl::Buffer interfaceVelocityBuffer;
	cl::Buffer interfaceMagneticFieldBuffer;
	cl::Buffer fluxBuffer;
	cl::Buffer pressureBuffer;

	cl::Kernel findMinTimestepKernel;
	cl::Kernel calcInterfaceVelocityKernel;
	cl::Kernel calcVelocityFluxKernel;
	cl::Kernel calcInterfaceMagneticFieldKernel;
	cl::Kernel calcMagneticFieldFluxKernel;
	cl::Kernel calcFluxDerivKernel;
	cl::Kernel computePressureKernel;
	cl::Kernel diffuseMomentumKernel;
	cl::Kernel diffuseWorkKernel;

	//matches MHDRoe -- belongs in the MHDEquation class maybe?
	cl::Kernel initVariablesKernel;

public:
	using Super::Super;
	virtual void init();

protected:	
	virtual void createEquation();
	virtual std::vector<std::string> getProgramSources();
	
	virtual void calcTimestep();
	virtual void step();
	virtual void advectVelocity();
	virtual void advectMagneticField();
	virtual void diffusePressure();
	virtual void diffuseWork();
};

}
}

