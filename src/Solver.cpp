#include "HydroGPU/Solver.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

cl::Buffer Solver::clAlloc(size_t size) {
	totalAlloc += size;
	std::cout << "allocating gpu mem size " << size << " running total " << totalAlloc << std::endl; 
	return cl::Buffer(app.context, CL_MEM_READ_WRITE, size);
}

template<typename T> std::string toNumericString(T value);

template<> std::string toNumericString<double>(double value) {
	std::string s = std::to_string(value);
	if (s.find("e") == std::string::npos) {
		if (s.find(".") == std::string::npos) {
			s += ".";
		}
	}
	return s;
}

template<> std::string toNumericString<float>(float value) {
	return toNumericString<double>(value) + "f";
}

Solver::Solver(
	HydroGPUApp& app_,
	std::vector<std::string> programFilenames)
: app(app_)
, commands(app.commands)
, totalAlloc(0)
{
	cl::Device device = app.device;
	
	stateBoundaryKernels.resize(NUM_BOUNDARY_METHODS);
	for (std::vector<cl::Kernel>& v : stateBoundaryKernels) {
		v.resize(app.dim);
	}
	
	// NDRanges

#if 0
	Tensor::Vector<size_t,3> localSizeVec;
	size_t maxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	std::vector<size_t> maxWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
	for (int n = 0; n < 3; ++n) {
		localSizeVec(n) = std::min<size_t>(maxWorkItemSizes[n], app.size.s[n]);
	}
	while (localSizeVec.volume() > maxWorkGroupSize) {
		for (int n = 0; n < 3; ++n) {
			localSizeVec(n) = (size_t)ceil((double)localSizeVec(n) * .5);
		}
	}
#endif	
	
	//if dim 2 is size 1 then tell opencl to treat it like a 1D problem
	switch (app.dim) {
	case 1:
		globalSize = cl::NDRange(app.size.s[0]);
		localSize = cl::NDRange(16);
		localSize1d = cl::NDRange(localSize[0]);
		offset1d = cl::NDRange(0);
		offsetNd = cl::NDRange(0);
		break;
	case 2:
		globalSize = cl::NDRange(app.size.s[0], app.size.s[1]);
		localSize = cl::NDRange(16, 16);
		localSize1d = cl::NDRange(localSize[0]);
		offset1d = cl::NDRange(0);
		offsetNd = cl::NDRange(0, 0);
		break;
	case 3:
		globalSize = cl::NDRange(app.size.s[0], app.size.s[1], app.size.s[2]);
		localSize = cl::NDRange(8, 8, 8);
		localSize1d = cl::NDRange(localSize[0]);
		offset1d = cl::NDRange(0);
		offsetNd = cl::NDRange(0, 0, 0);
		break;
	}
	
	std::cout << "global_size\t" << globalSize << std::endl;
	std::cout << "local_size\t" << localSize << std::endl;
	
	{
		std::vector<std::string> kernelSources = std::vector<std::string>{
			std::string() + "#define GAMMA " + toNumericString<real>(app.gamma) + "\n" +
			std::string() + "#define DIM " + std::to_string(app.dim) + "\n" +
			std::string() + "#define SIZE_X " + std::to_string(app.size.s[0]) + "\n" +
			std::string() + "#define SIZE_Y " + std::to_string(app.size.s[1]) + "\n" +
			std::string() + "#define SIZE_Z " + std::to_string(app.size.s[2]) + "\n" +
			std::string() + "#define DX " + toNumericString<real>(app.dx.s[0]) + "\n" +
			std::string() + "#define DY " + toNumericString<real>(app.dx.s[1]) + "\n" +
			std::string() + "#define DZ " + toNumericString<real>(app.dx.s[2]) + "\n" +
			std::string() + "#define SLOPE_LIMITER_" + app.slopeLimiterName + "\n"
		};
		kernelSources.push_back(Common::File::read("Common.cl"));
		kernelSources.push_back(Common::File::read("SlopeLimiter.cl"));
		for (const std::string& filename : programFilenames) {
			kernelSources.push_back(Common::File::read(filename));
		}
		std::vector<std::pair<const char *, size_t>> sources;
		for (const std::string &s : kernelSources) {
			sources.push_back(std::pair<const char *, size_t>(s.c_str(), s.length()));
		}
		program = cl::Program(app.context, sources);
	}

	try {
		program.build({device}, "-I include");// -Werror -cl-fast-relaxed-math");
	} catch (cl::Error &err) {
		throw Common::Exception() 
			<< "failed to build program executable!\n"
			<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
	}

	//warnings?
	std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;

	//for curiousity's sake
	{
		cl_int err;
		
		size_t size = 0;
		err = clGetProgramInfo(program(), CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &size, NULL);
		if (err != CL_SUCCESS) throw Common::Exception() << "failed to get binary size";
	
		std::vector<char> binary(size);
		err = clGetProgramInfo(program(), CL_PROGRAM_BINARIES, size, &binary[0], NULL);
		if (err != CL_SUCCESS) throw Common::Exception() << "failed to get binary";

		Common::File::write("program.cl.bin", std::string(&binary[0], binary.size()));
	}

	int volume = app.size.s[0] * app.size.s[1] * app.size.s[2];
	
	cflBuffer = clAlloc(sizeof(real) * volume);
	cflSwapBuffer = clAlloc(sizeof(real) * volume / localSize[0]);
	dtBuffer = clAlloc(sizeof(real16));
	gravityPotentialBuffer = clAlloc(sizeof(real) * volume);
	
	//get the edges, so reduction doesn't
	{
		std::vector<real> cflVec(volume);
		for (real &r : cflVec) { r = std::numeric_limits<real>::max(); }
		commands.enqueueWriteBuffer(cflBuffer, CL_TRUE, 0, sizeof(real) * volume, &cflVec[0]);
	}
	
	if (app.useFixedDT) {
		commands.enqueueWriteBuffer(dtBuffer, CL_TRUE, 0, sizeof(real), &app.fixedDT);
	}
}

void Solver::initKernels() {
	
	int volume = app.size.s[0] * app.size.s[1] * app.size.s[2];
	
	for (int boundaryIndex = 0; boundaryIndex < NUM_BOUNDARY_METHODS; ++boundaryIndex) {
		for (int side = 0; side < app.dim; ++side) {
			std::string name = "stateBoundary";
			switch (boundaryIndex) {
			case BOUNDARY_PERIODIC:
				name += "Periodic";
				break;
			case BOUNDARY_MIRROR:
				name += "Mirror";
				break;
			case BOUNDARY_FREEFLOW:
				name += "FreeFlow";
				break;
			default:
				throw Common::Exception() << "no kernel for boundary method " << boundaryIndex;
			}
			switch (side) {
			case 0:
				name += "X";
				break;
			case 1:
				name += "Y";
				break;
			case 2:
				name += "Z";
				break;
			}
			stateBoundaryKernels[boundaryIndex][side] = cl::Kernel(program, name.c_str());
			app.setArgs(stateBoundaryKernels[boundaryIndex][side], stateBuffer);
		}
	}
	
	calcCFLMinReduceKernel = cl::Kernel(program, "calcCFLMinReduce");
	app.setArgs(calcCFLMinReduceKernel, cflBuffer, cl::Local(localSize[0] * sizeof(real)), volume, cflSwapBuffer);
	
	poissonRelaxKernel = cl::Kernel(program, "poissonRelax");
	app.setArgs(poissonRelaxKernel, gravityPotentialBuffer, stateBuffer);
	
	addGravityKernel = cl::Kernel(program, "addGravity");
	app.setArgs(addGravityKernel, stateBuffer, gravityPotentialBuffer, dtBuffer);
}

void Solver::findMinTimestep() {
	int reduceSize = app.size.s[0] * app.size.s[1] * app.size.s[2];
	cl::Buffer dst = cflSwapBuffer;
	cl::Buffer src = cflBuffer;
	while (reduceSize > 1) {
		int nextSize = (reduceSize >> 4) + !!(reduceSize & ((1 << 4) - 1));
		cl::NDRange reduceGlobalSize(std::max<int>(reduceSize, localSize[0]));
		calcCFLMinReduceKernel.setArg(0, src);
		calcCFLMinReduceKernel.setArg(2, reduceSize);
		calcCFLMinReduceKernel.setArg(3, nextSize == 1 ? dtBuffer : dst);
		commands.enqueueNDRangeKernel(calcCFLMinReduceKernel, offset1d, reduceGlobalSize, localSize1d);
		commands.finish();
		std::swap(dst, src);
		reduceSize = nextSize;
	}
}

void Solver::initStep() {
}

void Solver::update() {
	if (app.showTimestep) {
		real dt;
		commands.enqueueReadBuffer(dtBuffer, CL_TRUE, 0, sizeof(real), &dt);
		std::cout << "dt " << dt << std::endl;
	}

	setPoissonRelaxRepeatArg();

	//commands.enqueueNDRangeKernel(addSourceKernel, offsetNd, globalSize, localSize, NULL, &addSourceEvent.clEvent);

	boundary();
	
	initStep();
	
	if (!app.useFixedDT) {
		calcTimestep();
	}
	
	step();

/*
	for (EventProfileEntry *entry : entries) {
		cl_ulong start = entry->clEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong end = entry->clEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
		entry->stat.accum((double)(end - start) * 1e-9);
	}
*/

}

void Solver::setPoissonRelaxRepeatArg() {
	cl_int4 repeat;
	for (int i = 0; i < app.dim; ++i) {
		switch (app.boundaryMethods(0)) {	//TODO per dimension
		case BOUNDARY_PERIODIC:
			repeat.s[i] = 1;
			break;
		case BOUNDARY_MIRROR:
		case BOUNDARY_FREEFLOW:
			repeat.s[i] = 0;
			break;
		default:
			throw Common::Exception() << "unknown boundary method " << app.boundaryMethods(0);
		}	
	}	
	poissonRelaxKernel.setArg(2, repeat);
}

