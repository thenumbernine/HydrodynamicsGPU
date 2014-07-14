#include "HydroGPU/Integrator/ForwardEuler.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Integrator {

ForwardEuler::ForwardEuler(HydroGPU::Solver::Solver& solver) 
: Super(solver)
{
	int volume = solver.getVolume();
	
	derivBuffer = solver.clAlloc(sizeof(real) * solver.numStates() * volume);

	{
		std::vector<real> zero(volume * solver.numStates());
		solver.commands.enqueueWriteBuffer(derivBuffer, CL_TRUE, 0, sizeof(real) * solver.numStates() * volume, &zero[0]);
	}

	multAddKernel = cl::Kernel(solver.program, "multAdd");
	solver.app.setArgs(multAddKernel, solver.stateBuffer, derivBuffer, solver.dtBuffer, 1.f);
}

void ForwardEuler::integrate(std::function<void(cl::Buffer)> callback) {
	callback(derivBuffer);
	solver.commands.enqueueNDRangeKernel(multAddKernel, solver.offsetNd, solver.globalSize, solver.localSize);
}

}
}

