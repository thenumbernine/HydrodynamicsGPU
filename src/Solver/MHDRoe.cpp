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

	//matches MHDBurgers -- belongs in the MHDEquation class maybe?
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
/*
some MHD equations, like the 1998 and 1999 citations in MHDRoe.cl, use a source term to ensure the system is divergence-free.
I think my latest attempt was using Trangenstein, which only mentions gravity in the source term.
"it is common to assume B is divergence-free".  If we are assuming it, are we also enforcing it?
*/
//	addMHDSourceKernel.setArg(0, derivBuffer);
//	commands.enqueueNDRangeKernel(addMHDSourceKernel, offsetNd, globalSize, localSize);

	Super::calcDeriv(derivBuffer);
}

}
}

