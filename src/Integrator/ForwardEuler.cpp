#include "HydroGPU/Integrator/ForwardEuler.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Integrator {

ForwardEuler::ForwardEuler(HydroGPU::Solver::Solver* solver) 
: Super(solver)
{
	derivBuffer = solver->clAlloc(sizeof(real) * solver->numStates() * solver->getVolume(), "ForwardEuler::derivBuffer");

	//put this in parent class of ForwardEuler and RungeKutta4?
	multAddKernel = cl::Kernel(solver->program, "multAdd");
	multAddKernel.setArg(0, solver->stateBuffer);
	multAddKernel.setArg(1, solver->stateBuffer);
	multAddKernel.setArg(2, derivBuffer);
}

void ForwardEuler::integrate(std::function<void(cl::Buffer)> callback) {
	int length = solver->getVolume() * solver->numStates();
	
	//TODO store globalSize1d in Solver?
	cl::NDRange globalSize1d(length);
	
	solver->commands.enqueueFillBuffer(derivBuffer, 0.f, 0, sizeof(real) * length);

	callback(derivBuffer);

	//TODO do this in solver immediately after calcDT?
	real dt;
	solver->commands.enqueueReadBuffer(solver->dtBuffer, CL_TRUE, 0, sizeof(real), &dt);

	multAddKernel.setArg(3, dt);
	solver->commands.enqueueNDRangeKernel(multAddKernel, solver->offset1d, globalSize1d, solver->localSize1d);
}

}
}

