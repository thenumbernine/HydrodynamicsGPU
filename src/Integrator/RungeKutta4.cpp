#include "HydroGPU/Integrator/RungeKutta4.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Integrator {

RungeKutta4::RungeKutta4(HydroGPU::Solver::Solver* solver) 
: Super(solver)
{
	int volume = solver->getVolume();

	restoreBuffer = solver->clAlloc(sizeof(real) * solver->numStates() * volume, "RungeKutta4::restoreBuffer");
	deriv1Buffer = solver->clAlloc(sizeof(real) * solver->numStates() * volume, "RungeKutta4::deriv1Buffer");
	deriv2Buffer = solver->clAlloc(sizeof(real) * solver->numStates() * volume, "RungeKutta4::deriv2Buffer");
	deriv3Buffer = solver->clAlloc(sizeof(real) * solver->numStates() * volume, "RungeKutta4::deriv3Buffer");
	deriv4Buffer = solver->clAlloc(sizeof(real) * solver->numStates() * volume, "RungeKutta4::deriv4Buffer");

	//put this in parent class of ForwardEuler and RungeKutta4?
	multAddKernel = cl::Kernel(solver->program, "multAdd");
	multAddKernel.setArg(0, solver->stateBuffer);
	multAddKernel.setArg(1, solver->stateBuffer);
}

void RungeKutta4::integrate(real dt, std::function<void(cl::Buffer)> callback) {
	size_t length = solver->numStates() * solver->getVolume();
	size_t bufferSize = sizeof(real) * length;
	cl::NDRange globalSize1d(length);

	//store backup
	solver->commands.enqueueCopyBuffer(solver->stateBuffer, restoreBuffer, 0, 0, bufferSize);
	
	solver->commands.enqueueFillBuffer(deriv1Buffer, 0.f, 0, bufferSize);
	callback(deriv1Buffer);
	
	//integrate by dt/2 along deriv1
	multAddKernel.setArg(2, deriv1Buffer);
	multAddKernel.setArg(3, .5f * dt);
	solver->commands.enqueueNDRangeKernel(multAddKernel, solver->offset1d, globalSize1d, solver->localSize1d);
	
	solver->commands.enqueueFillBuffer(deriv2Buffer, 0.f, 0, bufferSize);
	callback(deriv2Buffer);
	
	//restore backup
	solver->commands.enqueueCopyBuffer(restoreBuffer, solver->stateBuffer, 0, 0, bufferSize);
	
	//integrate by dt/2 along deriv2
	multAddKernel.setArg(2, deriv2Buffer);
	multAddKernel.setArg(3, .5f * dt);
	solver->commands.enqueueNDRangeKernel(multAddKernel, solver->offset1d, globalSize1d, solver->localSize1d);
	
	solver->commands.enqueueFillBuffer(deriv3Buffer, 0.f, 0, bufferSize);
	callback(deriv3Buffer);
	
	//restore backup
	solver->commands.enqueueCopyBuffer(restoreBuffer, solver->stateBuffer, 0, 0, bufferSize);
	
	//integrate by dt along deriv3
	multAddKernel.setArg(2, deriv3Buffer);
	multAddKernel.setArg(3, .5f * dt);
	solver->commands.enqueueNDRangeKernel(multAddKernel, solver->offset1d, globalSize1d, solver->localSize1d);
	
	solver->commands.enqueueFillBuffer(deriv4Buffer, 0.f, 0, bufferSize);
	callback(deriv4Buffer);
	
	//restore backup
	solver->commands.enqueueCopyBuffer(restoreBuffer, solver->stateBuffer, 0, 0, bufferSize);
	
	//integrate by dt/6 along (k1 + 2*k2 + 2*k3 + k4)
	multAddKernel.setArg(2, deriv1Buffer);
	multAddKernel.setArg(3, dt / 6.f);
	solver->commands.enqueueNDRangeKernel(multAddKernel, solver->offset1d, globalSize1d, solver->localSize1d);
	multAddKernel.setArg(2, deriv2Buffer);
	multAddKernel.setArg(3, dt / 3.f);
	solver->commands.enqueueNDRangeKernel(multAddKernel, solver->offset1d, globalSize1d, solver->localSize1d);
	multAddKernel.setArg(2, deriv3Buffer);
	multAddKernel.setArg(3, dt / 3.f);
	solver->commands.enqueueNDRangeKernel(multAddKernel, solver->offset1d, globalSize1d, solver->localSize1d);
	multAddKernel.setArg(2, deriv4Buffer);
	multAddKernel.setArg(3, dt / 6.f);
	solver->commands.enqueueNDRangeKernel(multAddKernel, solver->offset1d, globalSize1d, solver->localSize1d);
}

}
}

