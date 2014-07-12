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

	addMHDSourceKernel = cl::Kernel(program, "addMHDSource");
	app.setArgs(addMHDSourceKernel, stateBuffer, dtBuffer);
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
	//apply source terms
	//NOTICE the MHD paper says to apply them alongside the flux integration
	//but I am adding them here, coinciding with where the Hydrodynamics ii paper says to add the source terms due to potential energy
	//The difference is that the additions here will influence the Roe solver eigenbasis, whereas adding them alongside would not
	//Come to think of it, treating an addition of source terms at the beginning of this frame as equivalent of an addition of source terms at the end of the last frame,
	//there won't be any difference until I add in a better explicit integrator -- and the MHD paper says it uses a better-than-Forward-Euler integrator
	commands.enqueueNDRangeKernel(addMHDSourceKernel, offsetNd, globalSize, localSize);
	
	Super::step();
}

