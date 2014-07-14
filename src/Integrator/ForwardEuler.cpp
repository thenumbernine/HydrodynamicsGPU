#include "HydroGPU/Integrator/ForwardEuler.h"
#include "HydroGPU/Solver.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Integrator {

ForwardEuler::ForwardEuler(Solver& solver) 
: Super(solver)
{
	int volume = solver.getVolume();
	
	derivBuffer = solver.clAlloc(sizeof(real) * solver.numStates() * volume);

	{
		std::vector<real> zero(volume * solver.numStates());
		solver.commands.enqueueWriteBuffer(derivBuffer, CL_TRUE, 0, sizeof(real) * solver.numStates() * volume, &zero[0]);
	}

	forwardEulerIntegrateKernel = cl::Kernel(solver.program, "forwardEulerIntegrate");
	solver.app.setArgs(forwardEulerIntegrateKernel, solver.stateBuffer, derivBuffer, solver.dtBuffer);
}

void ForwardEuler::integrate(std::function<void(cl::Buffer)> callback) {
	callback(derivBuffer);
	solver.commands.enqueueNDRangeKernel(forwardEulerIntegrateKernel, solver.offsetNd, solver.globalSize, solver.localSize);
}

}
}

