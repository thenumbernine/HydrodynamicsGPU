#include "HydroGPU/MHDRoe.h"
#include "HydroGPU/MHDEquation.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

MHDRoe::MHDRoe(HydroGPUApp& app_)
: Super(app_)
{
	equation = std::make_shared<MHDEquation>(*this);
}

void MHDRoe::initKernels() {
	Super::initKernels();

	initVariablesKernel = cl::Kernel(program, "initVariables");
	app.setArgs(initVariablesKernel, stateBuffer);
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

void MHDRoe::step() {
	//TODO apply source terms
	//...or do like the paper does and apply it in an iterative integration method
	
	Super::step();
}

