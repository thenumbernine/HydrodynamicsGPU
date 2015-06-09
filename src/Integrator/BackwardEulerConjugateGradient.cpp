#include "HydroGPU/Integrator/BackwardEulerConjugateGradient.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Integrator {

/*
abstracting linear operations...
q(t) + dt*dq(t+dt)/dt = q(t+dt)
	let dq(t+dt)/dt = D*q(t+dt)
q(t) + dt*D*q(t+dt) = q(t+dt)
(I - dt*D)*q(t+dt) = q(t)
	let A = I - dt*D
	let x = q(t+dt)
	let b = q(t)

algorithm:
vector r, p, Ap, nextr
real rLenSq, alpha, nextrLenSq, beta

x = initial guess
r = b - A(x)
rLenSq = dot(r,r)
p = r
repeat
	Ap = A(p)
	alpha = rLenSq / dot(p, Ap) 
	x = x + alpha * p
	nextr = r - alpha * Ap
	nextrLenSq = dot(nextr ,nextr)
	if (nextrLenSq < epsilon) break;
	beta = nextrLenSq / rLenSq
	p = nextr + beta * p
	r = nextr
	rLenSq = nextrLenSq
x is the result
*/

BackwardEulerConjugateGradient::BackwardEulerConjugateGradient(HydroGPU::Solver::Solver* solver) 
: Super(solver)
{
	rBuffer = solver->clAlloc(sizeof(real) * solver->numStates() * solver->getVolume(), "BackwardEulerConjugateGradient::derivBuffer");
	pBuffer = solver->clAlloc(sizeof(real) * solver->numStates() * solver->getVolume(), "BackwardEulerConjugateGradient::derivBuffer");
	ApBuffer = solver->clAlloc(sizeof(real) * solver->numStates() * solver->getVolume(), "BackwardEulerConjugateGradient::derivBuffer");

	scratchScalarBuffer = solver->clAlloc(sizeof(real));
	
	multAddKernel = cl::Kernel(solver->program, "multAdd");

	subtractKernel = cl::Kernel(solver->program, "subtract");
	
	dotBufferKernel = cl::Kernel(solver->program, "dotBuffer");
	dotBufferKernel.setArg(0, scratchScalarBuffer);
	dotBufferKernel.setArg(3, cl::Local(solver->localSize[0] * sizeof(real)));
}

real BackwardEulerConjugateGradient::dot(cl::Buffer a, cl::Buffer b, int length) {
	dotBufferKernel.setArg(1, a);
	dotBufferKernel.setArg(2, b);
	dotBufferKernel.setArg(4, length);
	
	solver->commands.enqueueNDRangeKernel(dotBufferKernel, solver->offset1d, cl::NDRange(length), solver->localSize1d);
	
	real result;
	solver->commands.enqueueReadBuffer(scratchScalarBuffer, CL_TRUE, 0, sizeof(real), &result);
	return result;
}

/*
applies the CG backward Euler matrix, 
which is A = I - dt * D

reads x after modifying result, so they shouldn't be the same
*/
void BackwardEulerConjugateGradient::applyLinear(cl::Buffer result, cl::Buffer x, real dt) {
	int length = solver->getVolume() * solver->numStates();
	
	solver->applyDStateDtMatrix(result, x);
	//result = D * x

	solver->app->setArgs(multAddKernel, result, x, result, -dt);
	solver->commands.enqueueNDRangeKernel(multAddKernel, solver->offset1d, cl::NDRange(length), solver->localSize1d);
	//result = x - dt * D * x = (I - dt * D) * x
}

void BackwardEulerConjugateGradient::integrate(real dt, std::function<void(cl::Buffer)> callback) {
	size_t length = solver->getVolume() * solver->numStates();
	size_t bufferSize = sizeof(real) * length;
	const int maxIter = 20;
	const real epsilon = 1e-3;

	//implicit doesn't fill a partial derivative buffer
	//instead it operates on the coeff buffer
	//though I could instead be passing the sparse matrix data through the callback
	// (and then generating the derivatives in the explicit integrators)
	callback(scratchScalarBuffer);

	//r = b - A(x)
	{
		cl::Buffer& tmp = pBuffer;
		applyLinear(tmp, solver->stateBuffer, dt);
	
		solver->app->setArgs(subtractKernel, rBuffer, solver->stateBuffer, tmp);
		solver->commands.enqueueNDRangeKernel(subtractKernel, solver->offset1d, cl::NDRange(length), solver->localSize1d);
	}
	
	real rLenSq = dot(rBuffer, rBuffer, length);

	solver->commands.enqueueCopyBuffer(rBuffer, pBuffer, 0, 0, bufferSize);
	
	for (int i = 0; i < maxIter; ++i) {
		applyLinear(ApBuffer, pBuffer, dt);	//Ap = A(p)
		real alpha = rLenSq / dot(pBuffer, ApBuffer, length);
		
		//solver->stateBuffer = solver->stateBuffer + alpha * p
		solver->app->setArgs(multAddKernel, solver->stateBuffer, solver->stateBuffer, pBuffer, alpha);
		solver->commands.enqueueNDRangeKernel(multAddKernel, solver->offset1d, cl::NDRange(length), solver->localSize1d);
	
		//r = r - alpha * Ap
		solver->app->setArgs(multAddKernel, rBuffer, rBuffer, ApBuffer, -alpha);
		solver->commands.enqueueNDRangeKernel(multAddKernel, solver->offset1d, cl::NDRange(length), solver->localSize1d);
		
		real nextrLenSq = dot(rBuffer, rBuffer, length);
		if (nextrLenSq < epsilon) break;
		real beta = nextrLenSq / rLenSq;
		
		//p = r - beta * p
		solver->app->setArgs(multAddKernel, pBuffer, rBuffer, pBuffer, beta);
		solver->commands.enqueueNDRangeKernel(multAddKernel, solver->offset1d, cl::NDRange(length), solver->localSize1d);
		
		rLenSq = nextrLenSq;
	}
}

}
}


