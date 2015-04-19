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

	subtractKernel = cl::Kernel(solver->program, "subtract");
}

/*
applyLinear is provided by the Solver
(provided via lambda, or via interface for implicit solvers?)
*/

void BackwardEulerConjugateGradient::integrate(std::function<void(cl::Buffer)> callback) {
#if 0
	const int maxIter = 20;
	const real epsilon = 1e-3;
	
	//r = b - A(x)
	{
		cl::Buffer& tmp = pBuffer;
		applyLinear(tmp, solver->stateBuffer);
	
		int length = solver->getVolume() * solver->numStates();
		solver->app.setArgs(subtractKernel, rBuffer, solver->stateBuffer, tmp, length);
		solver->commands.enqueueNDRangeKernel(subtractKernel, solver->offset1d, cl::NDRange(length), solver->localSize1d);
	}
	real rLenSq = dot(rBuffer, rBuffer);
	copy(pBuffer, rBuffer);	//p = r
	for (int i = 0; i < maxIter; ++i) {
		applyLinear(ApBuffer, pBuffer);	//Ap = A(p)
		real alpha = rLenSq / dot(pBuffer, ApBuffer);
		multAddTo(solver->stateBuffer, solver->stateBuffer, pBuffer, alpha);	//solver->stateBuffer = solver->stateBuffer + alpha * p
		multAddTo(rBuffer, rBuffer, ApBuffer, -alpha);				//r = r - alpha * Ap
		real nextrLenSq = dot(rBuffer, rBuffer);
		if (nextrLenSq < epsilon) break;
		real beta = nextrLenSq / rLenSq;
		multAddTo(pBuffer, rBuffer, pBuffer, beta);	//p = r - beta * p
		rLenSq = nextrLenSq
	}
#endif
}

}
}


