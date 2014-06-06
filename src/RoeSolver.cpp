#include "HydroGPU/Shared/Types.h"	//OpenCL shared header
#include "HydroGPU/RoeSolver.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Image/System.h"
#include "Common/Exception.h"
#include "Common/Macros.h"
#include "Common/File.h"
#include "Tensor/Vector.h"
#include <OpenGL/gl.h>
#include <fstream>

RoeSolver::RoeSolver(HydroGPUApp &app_)
: app(app_)
, calcEigenBasisEvent("calcEigenBasis")
, calcCFLEvent("calcCFL")
, calcDeltaQTildeEvent("calcDeltaQTilde")
, calcCFLMinReduceEvent("calcCFLMinReduce")
, calcCFLMinFinalEvent("calcCFLMinFinal")
, calcFluxEvent("calcFlux")
, updateStateEvent("updateState")
, addSourceEvent("addSource")
, cfl(.5f)
, drop(false)
{
	cl::Device device = app.device;
	cl::Context context = app.context;
	cl::CommandQueue commands = app.commands;
	cl::ImageGL fluidTexMem = app.fluidTexMem;
	cl::ImageGL gradientTexMem = app.gradientTexMem;
	real2 xmin = app.xmin;
	real2 xmax = app.xmax;
	cl_int2 size = app.size;
	bool useGPU = app.useGPU;
	
	entries.push_back(&calcEigenBasisEvent);
	entries.push_back(&calcCFLEvent);
	entries.push_back(&calcDeltaQTildeEvent);
	entries.push_back(&calcCFLMinReduceEvent);
	entries.push_back(&calcCFLMinFinalEvent);
	entries.push_back(&calcFluxEvent);
	entries.push_back(&updateStateEvent);

	size_t maxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	std::vector<size_t> maxWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
	Tensor::Vector<size_t,DIM> globalSizeVec, localSizeVec;
	for (int n = 0; n < DIM; ++n) {
		globalSizeVec(n) = size.s[n];
		localSizeVec(n) = std::min<size_t>(maxWorkItemSizes[n], size.s[n]);
	}
	while (localSizeVec.volume() > maxWorkGroupSize) {
		for (int n = 0; n < DIM; ++n) {
			localSizeVec(n) = (size_t)ceil((double)localSizeVec(n) * .5);
		}
	}
	//hmm...
	if (!useGPU) localSizeVec(0) >>= 1;
	std::cout << "global_size\t" << globalSizeVec << std::endl;
	std::cout << "local_size\t" << localSizeVec << std::endl;

	globalSize = cl::NDRange(globalSizeVec(0), globalSizeVec(1));
	localSize = cl::NDRange(localSizeVec(0), localSizeVec(1));

	std::string kernelSource = Common::File::read("Roe.cl");
	std::vector<std::pair<const char *, size_t>> sources = {
		std::pair<const char *, size_t>(kernelSource.c_str(), kernelSource.length())
	};
	program = cl::Program(context, sources);
 
	try {
		program.build({device}, "-I include");
	} catch (cl::Error &err) {
		throw Common::Exception() 
			<< "failed to build program executable!\n"
			<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
	}

	//warnings?
	std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;

	//memory

	int volume = size.s[0] * size.s[1];

	stateBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real4) * volume);
	eigenvaluesBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real4) * volume * 2);
	eigenvectorsBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real16) * volume * 2);
	eigenvectorsInverseBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real16) * volume * 2);
	deltaQTildeBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real4) * volume * 2);
	fluxBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real4) * volume * 2);
	cflBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume);
	cflTimestepBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real));

	{
		const real noise = .01;
		int index[DIM];

		std::vector<real4> stateVec(volume);
		real4* state = &stateVec[0];	
		//for (index[2] = 0; index[2] < size.s[2]; ++index[2]) {
			for (index[1] = 0; index[1] < size.s[1]; ++index[1]) {
				for (index[0] = 0; index[0] < size.s[0]; ++index[0], ++state) {
					
					bool lhs = true;
					Tensor::Vector<real, 2> x;
					for (int n = 0; n < DIM; ++n) {
						x(n) = real(xmax.s[n] - xmin.s[n]) * real(index[n]) / real(size.s[n]) + real(xmin.s[n]);
						//if (cell->x.s[n] > real(.3) * real(xmax.s[n]) + real(.7) * real(xmin.s[n]))
						if (fabs(x(n)) > real(.15))
						{
							lhs = false;
							break;
						}
					}

					//sod init
					real density = lhs ? 1. : .1;
					real velocity[DIM];
					real energyKinetic = real();
					for (int n = 0; n < DIM; ++n) {
						velocity[n] = crand() * noise;
						energyKinetic += velocity[n] * velocity[n];
					}
					energyKinetic *= real(.5);
					real energyThermal = 1.;
					real energyTotal = energyKinetic + energyThermal;

					state->s[0] = density;
					for (int n = 0; n < DIM; ++n) {
						state->s[n+1] = density * velocity[n];
					}
					state->s[DIM+1] = density * energyTotal;
				}
			}
		//}
	
		commands.enqueueWriteBuffer(stateBuffer, CL_TRUE, 0, sizeof(real4) * volume, &stateVec[0]);
	}
	
	real2 dx;
	for (int i = 0; i < DIM; ++i) {
		dx.s[i] = (xmax.s[i] - xmin.s[i]) / (float)size.s[i];
	}

	calcEigenBasisKernel = cl::Kernel(program, "calcEigenBasis");
	calcEigenBasisKernel.setArg(0, eigenvaluesBuffer);
	calcEigenBasisKernel.setArg(1, eigenvectorsBuffer);
	calcEigenBasisKernel.setArg(2, eigenvectorsInverseBuffer);
	calcEigenBasisKernel.setArg(3, stateBuffer);
	calcEigenBasisKernel.setArg(4, size);
	
	calcCFLKernel = cl::Kernel(program, "calcCFL");
	calcCFLKernel.setArg(0, cflBuffer);
	calcCFLKernel.setArg(1, eigenvaluesBuffer);
	calcCFLKernel.setArg(2, size);
	calcCFLKernel.setArg(3, dx);
	
	calcDeltaQTildeKernel = cl::Kernel(program, "calcDeltaQTilde");
	calcDeltaQTildeKernel.setArg(0, deltaQTildeBuffer);
	calcDeltaQTildeKernel.setArg(1, eigenvectorsInverseBuffer);
	calcDeltaQTildeKernel.setArg(2, stateBuffer);
	calcDeltaQTildeKernel.setArg(3, size);
	calcDeltaQTildeKernel.setArg(4, dx);

	calcCFLMinReduceKernel = cl::Kernel(program, "calcCFLMinReduce");
	calcCFLMinReduceKernel.setArg(0, cflBuffer);
	calcCFLMinReduceKernel.setArg(1, cl::Local(localSizeVec(0) * sizeof(real)));
	
	calcCFLMinFinalKernel = cl::Kernel(program, "calcCFLMinFinal");
	calcCFLMinFinalKernel.setArg(0, cflBuffer);
	calcCFLMinFinalKernel.setArg(1, cl::Local(localSizeVec(0) * sizeof(real)));
	calcCFLMinFinalKernel.setArg(2, cflTimestepBuffer);
	calcCFLMinFinalKernel.setArg(3, cfl);

	calcFluxKernel = cl::Kernel(program, "calcFlux");
	calcFluxKernel.setArg(0, fluxBuffer);
	calcFluxKernel.setArg(1, stateBuffer);
	calcFluxKernel.setArg(2, eigenvaluesBuffer);
	calcFluxKernel.setArg(3, eigenvectorsBuffer);
	calcFluxKernel.setArg(4, eigenvectorsInverseBuffer);
	calcFluxKernel.setArg(5, deltaQTildeBuffer);
	calcFluxKernel.setArg(6, size);
	calcFluxKernel.setArg(7, dx);
	calcFluxKernel.setArg(8, cflTimestepBuffer);
	
	updateStateKernel = cl::Kernel(program, "updateState");
	updateStateKernel.setArg(0, stateBuffer); 
	updateStateKernel.setArg(1, fluxBuffer); 
	updateStateKernel.setArg(2, size); 
	updateStateKernel.setArg(3, dx);
	updateStateKernel.setArg(4, cflTimestepBuffer);
	
	convertToTexKernel = cl::Kernel(program, "convertToTex");
	convertToTexKernel.setArg(0, stateBuffer);
	convertToTexKernel.setArg(1, size);
	convertToTexKernel.setArg(2, fluidTexMem);
	convertToTexKernel.setArg(3, gradientTexMem);

	addDropKernel = cl::Kernel(program, "addDrop");
	addDropKernel.setArg(0, stateBuffer);
	addDropKernel.setArg(1, size);
	addDropKernel.setArg(2, xmin);
	addDropKernel.setArg(3, xmax);
	addDropKernel.setArg(4, cflTimestepBuffer);

	addSourceKernel = cl::Kernel(program, "addSource");
	addSourceKernel.setArg(0, stateBuffer);
	addSourceKernel.setArg(1, size);
	addSourceKernel.setArg(2, xmin);
	addSourceKernel.setArg(3, xmax);
	addSourceKernel.setArg(4, cflTimestepBuffer);
}

RoeSolver::~RoeSolver() {
	std::cout << "OpenCL profiling info:" << std::endl;
	for (EventProfileEntry *entry : entries) {
		std::cout << entry->name << "\t" 
			<< "duration " << entry->stat
			<< std::endl;
	}
}

void RoeSolver::update() {
	cl::CommandQueue commands = app.commands;
	cl::ImageGL fluidTexMem = app.fluidTexMem;
	cl_int2 size = app.size;
	bool useGPU = app.useGPU;
	
	cl::NDRange offset2d(0, 0);

	if (drop) {
		addDropKernel.setArg(5, dropPos);
		addDropKernel.setArg(6, dropVel);
		commands.enqueueNDRangeKernel(addDropKernel, offset2d, globalSize, localSize);
		drop = false;
	}

	//commands.enqueueNDRangeKernel(addSourceKernel, offset2d, globalSize, localSize, NULL, &addSourceEvent.clEvent);
	commands.enqueueNDRangeKernel(calcEigenBasisKernel, offset2d, globalSize, localSize, NULL, &calcEigenBasisEvent.clEvent);
	commands.enqueueNDRangeKernel(calcCFLKernel, offset2d, globalSize, localSize, NULL, &calcCFLEvent.clEvent);
	commands.enqueueNDRangeKernel(calcDeltaQTildeKernel, offset2d, globalSize, localSize, NULL, &calcDeltaQTildeEvent.clEvent);

	{
		cl::NDRange offset1d(0);
		cl::NDRange reduceGlobalSize(globalSize[0] * globalSize[1] / 4);
		cl::NDRange reduceLocalSize(localSize[0]);
		commands.enqueueNDRangeKernel(calcCFLMinReduceKernel, offset1d, reduceGlobalSize, reduceLocalSize, NULL, &calcCFLMinReduceEvent.clEvent);
	
		while (reduceGlobalSize[0] / localSize[0] > localSize[0]) {
			reduceGlobalSize = cl::NDRange(reduceGlobalSize[0] / localSize[0]);
			commands.enqueueNDRangeKernel(calcCFLMinReduceKernel, offset1d, reduceGlobalSize, reduceLocalSize);
		}
		reduceGlobalSize = cl::NDRange(reduceGlobalSize[0] / localSize[0]);

		calcCFLMinFinalKernel.setArg(4, reduceGlobalSize);
		commands.enqueueNDRangeKernel(calcCFLMinFinalKernel, offset1d, reduceLocalSize, reduceLocalSize, NULL, &calcCFLMinFinalEvent.clEvent);
	}

	commands.enqueueNDRangeKernel(calcFluxKernel, offset2d, globalSize, localSize, NULL, &calcFluxEvent.clEvent);
	commands.enqueueNDRangeKernel(updateStateKernel, offset2d, globalSize, localSize, NULL, &updateStateEvent.clEvent);

	glFlush();
	glFinish();
	
	std::vector<cl::Memory> acquireGLMems = {fluidTexMem};
	commands.enqueueAcquireGLObjects(&acquireGLMems);

	if (useGPU) {
		commands.enqueueNDRangeKernel(convertToTexKernel, offset2d, globalSize, localSize);
	} else {
		int volume = size.s[0] * size.s[1];
		std::vector<real4> stateVec(volume);
		commands.enqueueReadBuffer(stateBuffer, CL_TRUE, 0, sizeof(real4) * volume, &stateVec[0]);  
		std::vector<Tensor::Vector<char,4>> texVec(volume);
		for (int i = 0; i < volume; ++i) {
			texVec[i](0) = (char)(255.f * stateVec[i].s[0] * .9f);
		}
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, size.s[0], size.s[1], GL_RGBA, GL_UNSIGNED_BYTE, &texVec[0].v);
	}

	commands.enqueueReleaseGLObjects(&acquireGLMems);
	commands.flush();
	commands.finish();

	for (EventProfileEntry *entry : entries) {
		cl_ulong start = entry->clEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong end = entry->clEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
		entry->stat.accum((double)(end - start) * 1e-9);
	}
}

void RoeSolver::addDrop(Tensor::Vector<float,2> pos, Tensor::Vector<float,2> vel) {
	dropPos.s[0] = pos(0);
	dropPos.s[1] = pos(1);
	dropVel.s[0] = vel(0);
	dropVel.s[1] = vel(1);
	drop = true;
}

void RoeSolver::screenshot() {
	for (int i = 0; i < 1000; ++i) {
		std::string filename = std::string("screenshot") + std::to_string(i) + ".fits";
		if (!Common::File::exists(filename)) {
			std::shared_ptr<Image::ImageType<float>> image = std::make_shared<Image::ImageType<float>>(
				Tensor::Vector<int,2>(app.size.s[0], app.size.s[1]),
				nullptr, 1);
			int volume = app.size.s[0] * app.size.s[1];
			std::vector<real4> stateVec(volume);
			app.commands.enqueueReadBuffer(stateBuffer, CL_TRUE, 0, sizeof(real4) * volume, &stateVec[0]);
			app.commands.flush();
			app.commands.finish();
			real4* state = &stateVec[0];
			for (int j = 0; j < app.size.s[1]; ++j) {
				for (int i = 0; i < app.size.s[0]; ++i, ++state) {
					(*image)(i,j) = state->s[0];
				}
			}
			Image::system->write(filename, image);
			return;
		}
	}
	throw Common::Exception() << "couldn't find an available filename";
}

