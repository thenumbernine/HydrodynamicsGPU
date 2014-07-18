#include "HydroGPU/Integrator/ForwardEuler.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Integrator {

ForwardEuler::ForwardEuler(HydroGPU::Solver::Solver& solver) 
: Super(solver)
{
	derivBuffer = solver.clAlloc(sizeof(real) * solver.numStates() * solver.getVolume());

	multAddKernel = cl::Kernel(solver.program, "multAdd");
	solver.app.setArgs(multAddKernel, solver.stateBuffer, derivBuffer, solver.dtBuffer, 1.f);
}

void ForwardEuler::integrate(std::function<void(cl::Buffer)> callback) {
	solver.commands.enqueueFillBuffer(derivBuffer, 0.f, 0, sizeof(real) * solver.getVolume() * solver.numStates());
	
	callback(derivBuffer);
	solver.commands.enqueueNDRangeKernel(multAddKernel, solver.offsetNd, solver.globalSize, solver.localSize);
}

}
}

