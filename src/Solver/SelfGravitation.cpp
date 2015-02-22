#include "HydroGPU/Solver/SelfGravitation.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

namespace HydroGPU {
namespace Solver {

SelfGravitation::SelfGravitation(Solver* solver_) : solver(solver_) {}

void SelfGravitation::initBuffers() {
	int volume = solver->getVolume();
	potentialBuffer = solver->clAlloc(sizeof(real) * volume);
}

void SelfGravitation::initKernels() {
	cl::Program program = solver->program;
	
	gravityPotentialPoissonRelaxKernel = cl::Kernel(program, "gravityPotentialPoissonRelax");
	solver->app->setArgs(gravityPotentialPoissonRelaxKernel, potentialBuffer, solver->stateBuffer);
	
	calcGravityDerivKernel = cl::Kernel(program, "calcGravityDeriv");
	calcGravityDerivKernel.setArg(1, solver->stateBuffer);
	calcGravityDerivKernel.setArg(2, potentialBuffer);
		
	solver->convertToTexKernel.setArg(5, potentialBuffer);
}

std::vector<std::string> SelfGravitation::getProgramSources() {
	return {Common::File::read("SelfGravitation.cl")};
}

void SelfGravitation::resetState(std::vector<real>& potentialVec, std::vector<real>& stateVec) {
	cl::CommandQueue commands = solver->commands;
	cl::NDRange globalSize = solver->globalSize;
	cl::NDRange localSize = solver->localSize;
	cl::NDRange offsetNd = solver->offsetNd;
	
	int volume = solver->getVolume();
	
	//if using gravity then use the density field as an initial guess before poisson relaxiation
	if (solver->app->useGravity) {
		for (size_t i = 0; i < volume; ++i) {
			potentialVec[i] = stateVec[0 + solver->numStates() * i];
		}
	}
	
	commands.enqueueWriteBuffer(potentialBuffer, CL_TRUE, 0, sizeof(real) * volume, potentialVec.data());
	
	if (solver->app->useGravity) {
		//solve for gravitational potential via gauss seidel
		for (int i = 0; i < solver->app->gaussSeidelMaxIter; ++i) {
			potentialBoundary();
			commands.enqueueNDRangeKernel(gravityPotentialPoissonRelaxKernel, offsetNd, globalSize, localSize);
		}
	}

	//add potential energy into total energy
	for (int i = 0; i < volume; ++i) {
		int energyTotalIndex = 1 + solver->app->dim;
		stateVec[energyTotalIndex + solver->numStates() * i] += potentialVec[i];
	}

	commands.enqueueWriteBuffer(solver->stateBuffer, CL_TRUE, 0, sizeof(real) * solver->numStates() * volume, stateVec.data());
	commands.finish();
}

void SelfGravitation::applyPotential() {
	cl::CommandQueue commands = solver->commands;
	cl::NDRange globalSize = solver->globalSize;
	cl::NDRange localSize = solver->localSize;
	cl::NDRange offsetNd = solver->offsetNd;
	
	solver->integrator->integrate([&](cl::Buffer derivBuffer) {
		if (solver->app->useGravity) {
			for (int i = 0; i < solver->app->gaussSeidelMaxIter; ++i) {
				potentialBoundary();
				commands.enqueueNDRangeKernel(gravityPotentialPoissonRelaxKernel, offsetNd, globalSize, localSize);
			}	
		}

		calcGravityDerivKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(calcGravityDerivKernel, offsetNd, globalSize, localSize);
		
		solver->boundary();	
	});
}

void SelfGravitation::potentialBoundary() {
	cl::CommandQueue commands = solver->commands;
	
	cl::NDRange offset, global, local;
	for (int i = 0; i < solver->app->dim; ++i) {
		int boundaryKernelIndex = solver->equation->gravityGetBoundaryKernelForBoundaryMethod(i);
		cl::Kernel& kernel = solver->boundaryKernels[boundaryKernelIndex][i];
		solver->app->setArgs(kernel, potentialBuffer, 1, 0);
		solver->getBoundaryRanges(i, offset, global, local);
		commands.enqueueNDRangeKernel(kernel, offset, global, local);
	}
}

}
}
