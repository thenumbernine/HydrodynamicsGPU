#include "HydroGPU/Solver/MHDRoe.h"
#include "HydroGPU/Equation/MHD.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

namespace HydroGPU {
namespace Solver {

MHDRoe::MHDRoe(HydroGPUApp& app_)
: Super(app_)
{
	equation = std::make_shared<HydroGPU::Equation::MHD>(*this);
}

void MHDRoe::initKernels() {
	Super::initKernels();

//debugging:
//app.setArgs(convertToTexKernel, eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, fluidTexMem);

	initVariablesKernel = cl::Kernel(program, "initVariables");
	app.setArgs(initVariablesKernel, stateBuffer);

	addMHDSourceKernel = cl::Kernel(program, "addMHDSource");
	addMHDSourceKernel.setArg(1, stateBuffer);
}

std::vector<std::string> MHDRoe::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back(Common::File::read("MHDRoe.cl"));
	return sources;
}

void MHDRoe::resetState() {
	Super::resetState();
	commands.enqueueNDRangeKernel(initVariablesKernel, offsetNd, globalSize, localSize);
}

void MHDRoe::calcDeriv(cl::Buffer derivBuffer) {
	addMHDSourceKernel.setArg(0, derivBuffer);
	commands.enqueueNDRangeKernel(addMHDSourceKernel, offsetNd, globalSize, localSize);

	Super::calcDeriv(derivBuffer);
}

}
}

