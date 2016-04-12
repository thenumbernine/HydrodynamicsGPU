#include "HydroGPU/Solver/SRHDRoe.h"
#include "HydroGPU/Equation/SRHD.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Plot/Plot.h"

namespace HydroGPU {
namespace Solver {

void SRHDRoe::initBuffers() {
	Super::initBuffers();
	
	primitiveBuffer = cl.alloc(sizeof(real) * numStates() * getVolume(), "SRHDRoe::initBuffers");
}

void SRHDRoe::initKernels() {
	Super::initKernels();
	
	calcEigenBasisKernel.setArg(3, primitiveBuffer);
	
	initVariablesKernel = cl::Kernel(program, "initVariables");
	CLCommon::setArgs(initVariablesKernel, stateBuffer, primitiveBuffer);

	updatePrimitivesKernel = cl::Kernel(program, "updatePrimitives");
	CLCommon::setArgs(updatePrimitivesKernel, primitiveBuffer, stateBuffer);

	constrainStateKernel = cl::Kernel(program, "constrainState");
	constrainStateKernel.setArg(0, stateBuffer);
}

void SRHDRoe::createEquation() {
	equation = std::make_shared<HydroGPU::Equation::SRHD>(app);
}

std::vector<std::string> SRHDRoe::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back("#include \"SRHDRoe.cl\"\n");
	return sources;
}

void SRHDRoe::resetState() {
	//store Newtonian Euler equation state variables in stateBuffer
	Super::resetState();

	commands.enqueueNDRangeKernel(initVariablesKernel, offsetNd, globalSize, localSize);
}

/*
And once flux integration has taken place,
constrain the state variables to make sure they do not become unphysical.
I could add this to the 'updatePrimitives' kernel ...
... but then displayed variables could potentially be erroneous.

here's a dilemma ...
A single Forward Euler integration step takes place and the prims go out of sync.
Then you do root finding to re-update them.
What about multi-stage schemes?
The easy way is to just recompute prims before each flux integration.
But that is a lot of wasted updates.
It would be more efficient to save the prims along with the state vector.
That'd mean abstracting the push/pop functions of RK4...
*/
void SRHDRoe::step(real dt) {
	Super::step(dt);
	
	/*
	These are two separate kernels.
	They could be one.
	But moving the two lines of 'constrainState' into the top of 'updatePrimitives' is causing the sim to explode. 
	I blame AMD.
	*/
	commands.enqueueNDRangeKernel(constrainStateKernel, offsetNd, globalSize, localSize);
	commands.enqueueNDRangeKernel(updatePrimitivesKernel, offsetNd, globalSize, localSize);
}

/*
the primitive buffer components line up with the conservative buffer components,
so just use the same kernels on them
*/
void SRHDRoe::boundary() {
	Super::boundary();
	cl::NDRange offset, global, local;
	for (int i = 0; i < app->dim; ++i) {
		getBoundaryRanges(i, offset, global, local);
		for (int j = 0; j < numStates(); ++j) {
			for (int minmax = 0; minmax < 2; ++minmax) {
				int boundaryKernelIndex = equation->stateGetBoundaryKernelForBoundaryMethod(i, j, minmax);
				if (boundaryKernelIndex < 0 || boundaryKernelIndex >= boundaryKernels.size()) continue;
				cl::Kernel& kernel = boundaryKernels[boundaryKernelIndex][i][minmax];
				kernel.setArg(0, primitiveBuffer);
				kernel.setArg(1, numStates());
				kernel.setArg(2, j);
				commands.enqueueNDRangeKernel(kernel, offset, global, local);
			}
		}
	}
}
	
cl::Buffer SRHDRoe::getPrimitiveBuffer() {
	return primitiveBuffer;
}

}
}
