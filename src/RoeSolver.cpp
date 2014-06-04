#include "HydroGPU/RoeSolver.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/Exception.h"
#include "Common/Macros.h"
#include "Common/File.h"
#include "Tensor/Vector.h"
#include <OpenGL/gl.h>
#include <fstream>

RoeSolver::RoeSolver(HydroGPUApp &app_)
: app(app_)
, calcEigenDecompositionEvent("calcEigenDecomposition")
, calcCFLAndDeltaQTildeEvent("calcCFLAndDeltaQTilde")
, calcCFLMinReduceEvent("calcCFLMinReduce")
, calcCFLMinFinalEvent("calcCFLMinFinal")
, calcFluxEvent("calcFlux")
, updateStateEvent("updateState")
, cfl(.5f)
{
	cl::Device device = app.device;
	cl::Context context = app.context;
	cl::CommandQueue commands = app.commands;
	cl_mem fluidTexMem = app.fluidTexMem;
	cl_mem gradientTexMem = app.gradientTexMem;
	Tensor::Vector<real,2> xmin = app.xmin;
	Tensor::Vector<real,2> xmax = app.xmax;
	cl_int2 size = app.size;
	bool useGPU = app.useGPU;
	
	entries.push_back(&calcEigenDecompositionEvent);
	entries.push_back(&calcCFLAndDeltaQTildeEvent);
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

	std::string kernelSource = Common::File::read("roe_euler_2d.cl");
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

	std::vector<Cell> cells(size.s[0] * size.s[1]);
	{
		const real noise = .01;
		int index[DIM];
		
		Cell *cell = &cells[0];
		//for (index[2] = 0; index[2] < size.s[2]; ++index[2]) {
			for (index[1] = 0; index[1] < size.s[1]; ++index[1]) {
				for (index[0] = 0; index[0] < size.s[0]; ++index[0], ++cell) {
					bool lhs = true;
					for (int n = 0; n < DIM; ++n) {
						cell->x.s[n] = real(xmax(n) - xmin(n)) * real(index[n]) / real(size.s[n]) + real(xmin(n));
						if (cell->x.s[n] > real(.3) * real(xmax(n)) + real(.7) * real(xmin(n))) {
							lhs = false;
						}
					}

					for (int m = 0; m < DIM; ++m) {
						for (int n = 0; n < DIM; ++n) {
							cell->interfaces[m].x.s[n] = cell->x.s[n];
							if (m == n) {
								cell->interfaces[m].x.s[n] -= real(xmax(n) - xmin(n)) * real(.5) / real(size.s[n]);
							}
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

					cell->q.s[0] = density;
					for (int n = 0; n < DIM; ++n) {
						cell->q.s[n+1] = density * velocity[n];
					}
					cell->q.s[DIM+1] = density * energyTotal;
				}
			}
		//}
	}

	unsigned int count = size.s[0] * size.s[1];
	cellsMem = cl::Buffer(context,  CL_MEM_READ_WRITE, sizeof(Cell) * count);
	commands.enqueueWriteBuffer(cellsMem, CL_TRUE, 0, sizeof(Cell) * count, &cells[0]);
	
	cflMem = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * count);
	cflTimestepMem = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real));

	calcEigenDecompositionKernel = cl::Kernel(program, "calcEigenDecomposition");
	calcCFLAndDeltaQTildeKernel = cl::Kernel(program, "calcCFLAndDeltaQTilde");
	calcCFLMinReduceKernel = cl::Kernel(program, "calcCFLMinReduce");
	calcCFLMinFinalKernel = cl::Kernel(program, "calcCFLMinFinal");
	calcFluxKernel = cl::Kernel(program, "calcFlux");
	updateStateKernel = cl::Kernel(program, "updateState");
	convertToTexKernel = cl::Kernel(program, "convertToTex");
	addDropKernel = cl::Kernel(program, "addDrop");

	std::vector<cl::Kernel*> kernels = {
		&calcEigenDecompositionKernel,
		&calcCFLAndDeltaQTildeKernel,
		&calcFluxKernel,
		&updateStateKernel,
		&addDropKernel,
	};
	std::for_each(kernels.begin(), kernels.end(), [&](cl::Kernel* kernel) {
		kernel->setArg(0, cellsMem);
		kernel->setArg(1, size);
	});

	real2 dx;
	for (int i = 0; i < DIM; ++i) {
		dx.s[i] = (xmax(i) - xmin(i)) / (float)size.s[i];
	}
	
	calcFluxKernel.setArg(2, dx);
	calcFluxKernel.setArg(3, cflTimestepMem);

	updateStateKernel.setArg(2, dx);
	updateStateKernel.setArg(3, cflTimestepMem);
	
	calcCFLAndDeltaQTildeKernel.setArg(2, cflMem);
	calcCFLAndDeltaQTildeKernel.setArg(3, dx);

	calcCFLMinReduceKernel.setArg(0, cflMem);
	calcCFLMinReduceKernel.setArg(1, cl::__local(localSizeVec(0) * sizeof(real)));
	
	calcCFLMinFinalKernel.setArg(0, cflMem);
	calcCFLMinFinalKernel.setArg(1, cl::__local(localSizeVec(0) * sizeof(real)));
	calcCFLMinFinalKernel.setArg(2, cflTimestepMem);
	calcCFLMinFinalKernel.setArg(3, cfl);

	addDropKernel.setArg(2, cflTimestepMem);

	//if (useGPU) 
	{
		convertToTexKernel.setArg(0, cellsMem);
		convertToTexKernel.setArg(1, size);
		convertToTexKernel.setArg(2, fluidTexMem);
		convertToTexKernel.setArg(3, gradientTexMem);
	}
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
	cl_mem fluidTexMem = app.fluidTexMem;
	cl_int2 size = app.size;
	bool useGPU = app.useGPU;
	
	cl::NDRange offset2d(0, 0);

	commands.enqueueNDRangeKernel(calcEigenDecompositionKernel, offset2d, globalSize, localSize, NULL, &calcEigenDecompositionEvent.clEvent);
	commands.enqueueNDRangeKernel(calcCFLAndDeltaQTildeKernel, offset2d, globalSize, localSize, NULL, &calcCFLAndDeltaQTildeEvent.clEvent);

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
	clEnqueueAcquireGLObjects(commands(), 1, &fluidTexMem, 0, 0, 0);

	if (useGPU) {
		commands.enqueueNDRangeKernel(convertToTexKernel, offset2d, globalSize, localSize);
	} else {
		int count = size.s[0] * size.s[1];
		std::vector<Cell> cells(count);
		commands.enqueueReadBuffer(cellsMem, CL_TRUE, 0, sizeof(Cell) * count, &cells[0]);  
		std::vector<Tensor::Vector<char,4>> buffer(count);
		for (int i = 0; i < count; ++i) {
			buffer[i](0) = (char)(255.f * cells[i].q.s[0] * .9f);
		}
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, size.s[0], size.s[1], GL_RGBA, GL_UNSIGNED_BYTE, &buffer[0].v);
	}

	clEnqueueReleaseGLObjects(commands(), 1, &fluidTexMem, 0, 0, 0);
	commands.flush();
	commands.finish();

	for (EventProfileEntry *entry : entries) {
		cl_ulong start = entry->clEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong end = entry->clEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
		entry->stat.accum((double)(end - start) * 1e-9);
	}
}

void RoeSolver::addDrop(Tensor::Vector<float,2> pos, Tensor::Vector<float,2> vel) {
	addSourcePos.s[0] = pos(0);
	addSourcePos.s[1] = pos(1);
	addSourceVel.s[0] = vel(0);
	addSourceVel.s[1] = vel(1);
	addDropKernel.setArg(3, addSourcePos);
	addDropKernel.setArg(4, addSourceVel);
	commands.enqueueNDRangeKernel(addDropKernel, cl::NDRange(0,0), globalSize, localSize);
}

