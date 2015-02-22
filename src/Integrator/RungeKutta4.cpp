#include "HydroGPU/Integrator/RungeKutta4.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Integrator {

RungeKutta4::RungeKutta4(HydroGPU::Solver::Solver* solver) 
: Super(solver)
{
	int volume = solver->getVolume();

	restoreBuffer = solver->clAlloc(sizeof(real) * solver->numStates() * volume);
	deriv1Buffer = solver->clAlloc(sizeof(real) * solver->numStates() * volume);
	deriv2Buffer = solver->clAlloc(sizeof(real) * solver->numStates() * volume);
	deriv3Buffer = solver->clAlloc(sizeof(real) * solver->numStates() * volume);
	deriv4Buffer = solver->clAlloc(sizeof(real) * solver->numStates() * volume);

	multAddKernel = cl::Kernel(solver->program, "multAdd");
	multAddKernel.setArg(0, solver->stateBuffer);
	multAddKernel.setArg(2, solver->dtBuffer);
}

void RungeKutta4::integrate(std::function<void(cl::Buffer)> callback) {
	size_t copySize = sizeof(real) * solver->numStates() * solver->getVolume();
	
	//store backup
	solver->commands.enqueueCopyBuffer(solver->stateBuffer, restoreBuffer, 0, 0, copySize);
	
	solver->commands.enqueueFillBuffer(deriv1Buffer, 0.f, 0, sizeof(real) * solver->getVolume() * solver->numStates());
	callback(deriv1Buffer);
	
	//integrate by dt/2 along deriv1
	multAddKernel.setArg(1, deriv1Buffer);
	multAddKernel.setArg(3, .5f);
	solver->commands.enqueueNDRangeKernel(multAddKernel, solver->offsetNd, solver->globalSize, solver->localSize);
	
	solver->commands.enqueueFillBuffer(deriv2Buffer, 0.f, 0, sizeof(real) * solver->getVolume() * solver->numStates());
	callback(deriv2Buffer);
	
	//restore backup
	solver->commands.enqueueCopyBuffer(restoreBuffer, solver->stateBuffer, 0, 0, copySize);
	
	//integrate by dt/2 along deriv2
	multAddKernel.setArg(1, deriv2Buffer);
	solver->commands.enqueueNDRangeKernel(multAddKernel, solver->offsetNd, solver->globalSize, solver->localSize);
	
	solver->commands.enqueueFillBuffer(deriv3Buffer, 0.f, 0, sizeof(real) * solver->getVolume() * solver->numStates());
	callback(deriv3Buffer);
	
	//restore backup
	solver->commands.enqueueCopyBuffer(restoreBuffer, solver->stateBuffer, 0, 0, copySize);
	
	//integrate by dt along deriv3
	multAddKernel.setArg(1, deriv3Buffer);
	multAddKernel.setArg(3, 1.f);
	solver->commands.enqueueNDRangeKernel(multAddKernel, solver->offsetNd, solver->globalSize, solver->localSize);
	
	solver->commands.enqueueFillBuffer(deriv4Buffer, 0.f, 0, sizeof(real) * solver->getVolume() * solver->numStates());
	callback(deriv4Buffer);
	
	//restore backup
	solver->commands.enqueueCopyBuffer(restoreBuffer, solver->stateBuffer, 0, 0, copySize);
	
	//integrate by dt/6 along (k1 + 2*k2 + 2*k3 + k4)
	multAddKernel.setArg(1, deriv1Buffer);
	multAddKernel.setArg(3, 1.f / 6.f);
	solver->commands.enqueueNDRangeKernel(multAddKernel, solver->offsetNd, solver->globalSize, solver->localSize);
	multAddKernel.setArg(1, deriv2Buffer);
	multAddKernel.setArg(3, 2.f / 6.f);
	solver->commands.enqueueNDRangeKernel(multAddKernel, solver->offsetNd, solver->globalSize, solver->localSize);
	multAddKernel.setArg(1, deriv3Buffer);
	multAddKernel.setArg(3, 2.f / 6.f);
	solver->commands.enqueueNDRangeKernel(multAddKernel, solver->offsetNd, solver->globalSize, solver->localSize);
	multAddKernel.setArg(1, deriv4Buffer);
	multAddKernel.setArg(3, 1.f / 6.f);
	solver->commands.enqueueNDRangeKernel(multAddKernel, solver->offsetNd, solver->globalSize, solver->localSize);
}

}
}

