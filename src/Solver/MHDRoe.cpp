#include "HydroGPU/Solver/MHDRoe.h"
#include "HydroGPU/Equation/MHD.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Solver {

void MHDRoe::initBuffers() {
	Super::initBuffers();
	
	//allocate flux flag buffer for determining if any flux values had to be pre-filled for bad eigenstate areas
	fluxFlagBuffer = cl.alloc(sizeof(char) * getVolume() * app->dim);
}

void MHDRoe::initKernels() {
	Super::initKernels();

	//all Euler and MHD systems also have a separate potential buffer...
	
	CLCommon::setArgs(calcEigenBasisKernel,
		eigenvaluesBuffer, eigenvectorsBuffer, stateBuffer, selfgrav->potentialBuffer,
		//selfgrav->solidBuffer,
		fluxBuffer,
		fluxFlagBuffer);
	
	//just like ordinary calcMHDFluxKernel -- and calls the ordinary
	// -- but with an extra step to bail out of the associated fluxFlag is already set 
	calcMHDFluxKernel = cl::Kernel(program, "calcMHDFlux");
	CLCommon::setArgs(calcMHDFluxKernel,
		fluxBuffer, stateBuffer, eigenvaluesBuffer, eigenvectorsBuffer, deltaQTildeBuffer,
		0, //dt
		//selfgrav->solidBuffer,
		fluxFlagBuffer);
}

void MHDRoe::createEquation() {
	equation = std::make_shared<HydroGPU::Equation::MHD>(app);
}

int MHDRoe::getEigenTransformStructSize() {
	return 7 * 7 * 2;
}

int MHDRoe::getEigenSpaceDim() {
	return 7;
}

std::vector<std::string> MHDRoe::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	
	//also in Roe.cl ...
	sources.push_back("#define EIGEN_TRANSFORM_STRUCT_SIZE "+std::to_string(getEigenTransformStructSize())+"\n");
	sources.push_back("#define EIGEN_SPACE_DIM "+std::to_string(getEigenSpaceDim())+"\n");
	
	sources.push_back("#include \"MHDRoe.cl\"\n");
	return sources;
}

//transforms are found within MHDRoe.cl
std::vector<std::string> MHDRoe::getEigenProgramSources() {
	return {};
}

void MHDRoe::initFlux() {
	//MHD-Roe is special in that it can write fluxes during the calc eigen basis kernel
	//(in the case of negative fluxes)
	// so for that, I'm going to fill the flux kernel to some flag beforehand.
	// zero is a safe flag, right?  no ... not for steady states ...
	cl.zero(fluxFlagBuffer, getVolume() * app->dim);

	Super::initFlux();
}

//override parent call
//call this instead
//it'll call through the CL code if it's needed
void MHDRoe::calcFlux(real dt) {
	calcMHDFluxKernel.setArg(5, dt);
	commands.enqueueNDRangeKernel(calcMHDFluxKernel, offsetNd, globalSize, localSize);
}

void MHDRoe::step(real dt) {
	Super::step(dt);
#warning add this once you know 1D MHD is working without it
	//selfgrav->applyPotential(dt);
	//divfree->update();
}

}
}
