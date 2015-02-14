#include "HydroGPU/Solver/MHDRemoveDivergence.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

namespace HydroGPU {
namespace Solver {

MHDRemoveDivergence::MHDRemoveDivergence(Solver& solver_) : solver(solver_) {}

void MHDRemoveDivergence::init() {
	cl::Context context = solver.app.context;
	cl::Program program = solver.program;
	
	int volume = solver.getVolume();
	
	magneticFieldDivergenceBuffer = solver.clAlloc(sizeof(real) * volume);
	magneticFieldPotentialBuffer = solver.clAlloc(sizeof(real) * volume);
	magneticFieldPotential2Buffer = solver.clAlloc(sizeof(real) * volume);

	calcMagneticFieldDivergenceKernel = cl::Kernel(program, "calcMagneticFieldDivergence");
	solver.app.setArgs(calcMagneticFieldDivergenceKernel, magneticFieldDivergenceBuffer, solver.stateBuffer);

	magneticPotentialPoissonRelaxKernel = cl::Kernel(program, "magneticPotentialPoissonRelax");
	solver.app.setArgs(magneticPotentialPoissonRelaxKernel, magneticFieldPotential2Buffer, magneticFieldPotentialBuffer, magneticFieldDivergenceBuffer);
	
	magneticFieldRemoveDivergenceKernel = cl::Kernel(program, "magneticFieldRemoveDivergence");
	solver.app.setArgs(magneticFieldRemoveDivergenceKernel, solver.stateBuffer, magneticFieldPotentialBuffer);

	solver.convertToTexKernel.setArg(6, magneticFieldDivergenceBuffer);
}
	
void MHDRemoveDivergence::getProgramSources(std::vector<std::string>& sources) {
	sources.push_back(Common::File::read("MHDRemoveDivergence.cl"));
}

void MHDRemoveDivergence::update() {
	cl::CommandQueue commands = solver.commands;
	cl::NDRange globalSize = solver.globalSize;
	cl::NDRange localSize = solver.localSize;
	cl::NDRange offsetNd = solver.offsetNd;

	int volume = solver.getVolume();
	
	//calculate divergence
	commands.enqueueNDRangeKernel(calcMagneticFieldDivergenceKernel, offsetNd, globalSize, localSize);
	boundary(magneticFieldDivergenceBuffer);	//boundary to magnetic field potential buffer

	//poisson relax divergence into potential buffer
	commands.enqueueFillBuffer(magneticFieldPotentialBuffer, 0.f, 0, sizeof(real) * volume);
	for (int i = 0; i < solver.app.gaussSeidelMaxIter; ++i) {
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
	for (int i = 0; i < solver.app.dim; ++i) {
		int boundaryKernelIndex = solver.equation->gravityGetBoundaryKernelForBoundaryMethod(solver, i);
		cl::Kernel& kernel = solver.boundaryKernels[boundaryKernelIndex][i];
		solver.app.setArgs(kernel, buffer, 1, 0);
		solver.getBoundaryRanges(i, offset, global, local);
		solver.commands.enqueueNDRangeKernel(kernel, offset, global, local);
	}
}


}
}

