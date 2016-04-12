#include "HydroGPU/Equation/SelfGravitationBehavior.h"
#include "HydroGPU/Solver/MHDRemoveDivergence.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/Equation/Equation.h"
#include "HydroGPU/Plot/Plot.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Solver {

MHDRemoveDivergence::MHDRemoveDivergence(Solver* solver_) : solver(solver_) {}

void MHDRemoveDivergence::init() {
	cl::Program program = solver->program;
	
	int volume = solver->getVolume();
	
	magneticFieldDivergenceBuffer = solver->cl.alloc(sizeof(real) * volume);
	magneticFieldPotentialBuffer = solver->cl.alloc(sizeof(real) * volume);
	magneticFieldPotential2Buffer = solver->cl.alloc(sizeof(real) * volume);

	calcMagneticFieldDivergenceKernel = cl::Kernel(program, "calcMagneticFieldDivergence");
	CLCommon::setArgs(calcMagneticFieldDivergenceKernel, magneticFieldDivergenceBuffer, solver->stateBuffer);

	magneticPotentialPoissonRelaxKernel = cl::Kernel(program, "magneticPotentialPoissonRelax");
	CLCommon::setArgs(magneticPotentialPoissonRelaxKernel, magneticFieldPotential2Buffer, magneticFieldPotentialBuffer, magneticFieldDivergenceBuffer);
	
	magneticFieldRemoveDivergenceKernel = cl::Kernel(program, "magneticFieldRemoveDivergence");
	CLCommon::setArgs(magneticFieldRemoveDivergenceKernel, solver->stateBuffer, magneticFieldPotentialBuffer);
}

std::vector<std::string> MHDRemoveDivergence::getProgramSources() {
	return {"#include \"MHDRemoveDivergence.cl\"\n"};
}

void MHDRemoveDivergence::update() {
	cl::CommandQueue commands = solver->commands;
	cl::NDRange globalSize = solver->globalSize;
	cl::NDRange localSize = solver->localSize;
	cl::NDRange offsetNd = solver->offsetNd;

	int volume = solver->getVolume();
	
	//calculate divergence
	commands.enqueueNDRangeKernel(calcMagneticFieldDivergenceKernel, offsetNd, globalSize, localSize);
	boundary(magneticFieldDivergenceBuffer);	//boundary to magnetic field potential buffer

	//poisson relax divergence into potential buffer
	solver->cl.zero(magneticFieldPotentialBuffer, volume * sizeof(real));
	for (int i = 0; i < solver->app->gaussSeidelMaxIter; ++i) {
		magneticPotentialPoissonRelaxKernel.setArg(0, magneticFieldPotential2Buffer);
		magneticPotentialPoissonRelaxKernel.setArg(1, magneticFieldPotentialBuffer);
		commands.enqueueNDRangeKernel(magneticPotentialPoissonRelaxKernel, offsetNd, globalSize, localSize);
		std::swap(magneticFieldPotential2Buffer, magneticFieldPotentialBuffer);
		boundary(magneticFieldPotentialBuffer);	//boundary to magnetic field potential buffer
	}
	magneticFieldRemoveDivergenceKernel.setArg(1, magneticFieldPotentialBuffer);
	commands.enqueueNDRangeKernel(magneticFieldRemoveDivergenceKernel, offsetNd, globalSize, localSize);
}

//looks a lot like Solver::potentialBoundary
// maybe I could merge them?
void MHDRemoveDivergence::boundary(cl::Buffer buffer) {
	cl::NDRange offset, global, local;
	std::shared_ptr<HydroGPU::Equation::SelfGravitationInterface> gravEqn = std::dynamic_pointer_cast<HydroGPU::Equation::SelfGravitationInterface>(solver->equation);
	for (int i = 0; i < solver->app->dim; ++i) {
		for (int minmax = 0; minmax < 2; ++minmax) {
			int boundaryKernelIndex = gravEqn->gravityGetBoundaryKernelForBoundaryMethod(i, minmax);
			if (boundaryKernelIndex < 0 || boundaryKernelIndex >= solver->boundaryKernels.size()) continue;
			cl::Kernel& kernel = solver->boundaryKernels[boundaryKernelIndex][i][minmax];
			CLCommon::setArgs(kernel, buffer, 1, 0);
			solver->getBoundaryRanges(i, offset, global, local);
			solver->commands.enqueueNDRangeKernel(kernel, offset, global, local);
		}
	}
}

cl::Buffer MHDRemoveDivergence::getMagneticFieldDivergenceBuffer() {
	return magneticFieldDivergenceBuffer;
}

}
}
