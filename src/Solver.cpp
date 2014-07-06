#include "HydroGPU/Solver.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

cl::Buffer Solver::clAlloc(size_t size) {
	totalAlloc += size;
	std::cout << "allocating gpu mem size " << size << " running total " << totalAlloc << std::endl; 
	return cl::Buffer(app.context, CL_MEM_READ_WRITE, size);
}

Solver::Solver(
	HydroGPUApp& app_)
: app(app_)
, commands(app.commands)
, totalAlloc(0)
{
}

void Solver::init() {
	cl::Device device = app.device;
	
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
		std::vector<std::string> sourceStrs = getProgramSources();
		std::vector<std::pair<const char *, size_t>> sources;
		for (const std::string &s : sourceStrs) {
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
	stateBuffer = clAlloc(sizeof(real) * equation->numStates * volume);
	
	//get the edges, so reduction doesn't
	{
		std::vector<real> cflVec(volume);
		for (real &r : cflVec) { r = std::numeric_limits<real>::max(); }
		commands.enqueueWriteBuffer(cflBuffer, CL_TRUE, 0, sizeof(real) * volume, &cflVec[0]);
	}
	
	commands.enqueueWriteBuffer(dtBuffer, CL_TRUE, 0, sizeof(real), &app.fixedDT);
}

std::vector<std::string> Solver::getProgramSources() {
	std::vector<std::string> sourceStrs = std::vector<std::string>{
		std::string() +
		"#define DIM " + std::to_string(app.dim) + "\n" +
		"#define SIZE_X " + std::to_string(app.size.s[0]) + "\n" +
		"#define SIZE_Y " + std::to_string(app.size.s[1]) + "\n" +
		"#define SIZE_Z " + std::to_string(app.size.s[2]) + "\n" +
		"#define STEP_X 1\n" +
		"#define STEP_Y " + std::to_string(app.size.s[0]) + "\n" +
		"#define STEP_Z " + std::to_string(app.size.s[0] * app.size.s[1]) + "\n" +
		"#define STEP_W " + std::to_string(app.size.s[0] * app.size.s[1] * app.size.s[2]) + "\n" +
		"#define DX " + toNumericString<real>(app.dx.s[0]) + "\n" +
		"#define DY " + toNumericString<real>(app.dx.s[1]) + "\n" +
		"#define DZ " + toNumericString<real>(app.dx.s[2]) + "\n" +
		"#define SLOPE_LIMITER_" + app.slopeLimiterName + "\n" +
		"#define NUM_STATES " + std::to_string(equation->numStates) + "\n"
	};
	sourceStrs.push_back(Common::File::read("SlopeLimiter.cl"));
	sourceStrs.push_back(Common::File::read("Common.cl"));
	equation->getProgramSources(*this, sourceStrs);
	return sourceStrs;
}

//Euler-specific
void Solver::resetState() {
	int volume = app.size.s[0] * app.size.s[1] * app.size.s[2];

	std::vector<real> stateVec(equation->numStates * volume);

	if (!app.lua.ref()["initState"].isFunction()) throw Common::Exception() << "expected initState to be defined in config file";

	std::cout << "initializing..." << std::endl;
	real* state = &stateVec[0];	
	int index[3];
	for (index[2] = 0; index[2] < app.size.s[2]; ++index[2]) {
		for (index[1] = 0; index[1] < app.size.s[1]; ++index[1]) {
			for (index[0] = 0; index[0] < app.size.s[0]; ++index[0], state += equation->numStates) {
				real4 pos;
				for (int i = 0; i < 3; ++i) {
					pos.s[i] = real(app.xmax.s[i] - app.xmin.s[i]) * (real(index[i]) + .5) / real(app.size.s[i]) + real(app.xmin.s[i]);
				}
				pos.s[3] = 0;
			
				LuaCxx::Stack stack = app.lua.stack();
				
				stack
				.getGlobal("initState")
				.push(pos.s[0], pos.s[1], pos.s[2])
				.call(3,5);	//TODO multret and have each equation interpret the results
			
				real density;
				real momentumX, momentumY, momentumZ;
				real energyTotal;
				
				stack
				.pop(energyTotal)
				.pop(momentumZ)
				.pop(momentumY)
				.pop(momentumX)
				.pop(density);
				state[0] = density;
				state[1] = momentumX;
				if (app.dim > 1) {
					state[2] = momentumY;
				}
				if (app.dim > 2) {
					state[3] = momentumZ;
				}
				state[app.dim+1] = energyTotal;
			}
		}
	}
	std::cout << "...done" << std::endl;

	//grad^2 Phi = - 4 pi G rho
	//solve inverse discretized linear system to find Psi
	//D_ij / (-4 pi G) Phi_j = rho_i
	//once you get that, plug it into the total energy
	
	//write state density first for gravity potential, to then update energy
	commands.enqueueWriteBuffer(stateBuffer, CL_TRUE, 0, sizeof(real) * equation->numStates * volume, &stateVec[0]);
	
	//here's our initial guess to sor
	std::vector<real> gravityPotentialVec(volume);
	for (size_t i = 0; i < volume; ++i) {
		if (app.useGravity) {
			gravityPotentialVec[i] = stateVec[0 + equation->numStates * i];
		} else {
			gravityPotentialVec[i] = 0.;
		}
	}
	
	commands.enqueueWriteBuffer(gravityPotentialBuffer, CL_TRUE, 0, sizeof(real) * volume, &gravityPotentialVec[0]);

	if (app.useGravity) {
		int energyTotalIndex = 1 + app.dim;

		setPoissonRelaxRepeatArg();
		
		//solve for gravitational potential via gauss seidel
		for (int i = 0; i < 20; ++i) {
			commands.enqueueNDRangeKernel(poissonRelaxKernel, offsetNd, globalSize, localSize);
		}

		//update total energy
		for (int i = 0; i < volume; ++i) {
			stateVec[energyTotalIndex + equation->numStates * i] += gravityPotentialVec[i];
		}
	}
	
	commands.enqueueWriteBuffer(stateBuffer, CL_TRUE, 0, sizeof(real) * equation->numStates * volume, &stateVec[0]);
	commands.finish();
}

static std::string boundaryKernelNames[NUM_BOUNDARY_KERNELS] = {
	"Periodic",
	"Mirror",
	"Reflect",
	"FreeFlow"
};

void Solver::initKernels() {
	
	int volume = app.size.s[0] * app.size.s[1] * app.size.s[2];

	boundaryKernels.resize(NUM_BOUNDARY_KERNELS);
	for (std::vector<cl::Kernel>& v : boundaryKernels) {
		v.resize(app.dim);
	}

	std::vector<std::string> dimNames = {"X", "Y", "Z"};
	for (int boundaryIndex = 0; boundaryIndex < NUM_BOUNDARY_KERNELS; ++boundaryIndex) {
		for (int side = 0; side < app.dim; ++side) {
			std::string name = "stateBoundary" + boundaryKernelNames[boundaryIndex] + dimNames[side];
			boundaryKernels[boundaryIndex][side] = cl::Kernel(program, name.c_str());
			app.setArgs(boundaryKernels[boundaryIndex][side], stateBuffer);
		}
	}
	
	calcCFLMinReduceKernel = cl::Kernel(program, "calcCFLMinReduce");
	app.setArgs(calcCFLMinReduceKernel, cflBuffer, cl::Local(localSize[0] * sizeof(real)), volume, cflSwapBuffer);
	
	poissonRelaxKernel = cl::Kernel(program, "poissonRelax");
	app.setArgs(poissonRelaxKernel, gravityPotentialBuffer, stateBuffer);
	
	addGravityKernel = cl::Kernel(program, "addGravity");
	app.setArgs(addGravityKernel, stateBuffer, gravityPotentialBuffer, dtBuffer);
}

void Solver::boundary() {
	switch (app.dim) {
	case 1:
	case 2:
		//boundary
		for (int i = 0; i < app.dim; ++i) {
			cl::NDRange globalSize1d(app.size.s[i]);
			for (int j = 0; j < equation->numStates; ++j) {
				int boundaryKernelIndex = equation->getBoundaryKernelForBoundaryMethod(*this, i, j);
				cl::Kernel &kernel = boundaryKernels[boundaryKernelIndex][i];
				kernel.setArg(1, equation->numStates);
				kernel.setArg(2, j);
				commands.enqueueNDRangeKernel(kernel, offset1d, globalSize1d, localSize1d);
			}
		}
		break;
	case 3:
		//boundary
		cl::NDRange offset2d(0, 0);
		cl::NDRange localSize2d(localSize[0], localSize[1]);
		for (int i = 0; i < app.dim; ++i) {
			cl::NDRange globalSize2d;
			switch (i) {
			case 0:
				globalSize2d = cl::NDRange(app.size.s[0], app.size.s[1]);
				break;
			case 1:
				globalSize2d = cl::NDRange(app.size.s[0], app.size.s[2]);
				break;
			case 2:
				globalSize2d = cl::NDRange(app.size.s[1], app.size.s[2]);
				break;
			}
			for (int j = 0; j < equation->numStates; ++j) {
				int boundaryKernelIndex = equation->getBoundaryKernelForBoundaryMethod(*this, i, j);
				cl::Kernel& kernel = boundaryKernels[boundaryKernelIndex][i];
				kernel.setArg(1, equation->numStates);
				kernel.setArg(2, j);
				commands.enqueueNDRangeKernel(kernel, offset2d, globalSize2d, localSize2d);
			}
		}
		break;
	}
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

//TODO specific to Euler
void Solver::setPoissonRelaxRepeatArg() {
	cl_int4 repeat;
	for (int i = 0; i < app.dim; ++i) {
		switch (app.boundaryMethods(0)) {	//TODO per dimension
		case 0://BOUNDARY_PERIODIC:
			repeat.s[i] = 1;
			break;
		case 1://BOUNDARY_MIRROR:
		case 2://BOUNDARY_FREEFLOW:
			repeat.s[i] = 0;
			break;
		default:
			throw Common::Exception() << "unknown boundary method " << app.boundaryMethods(0);
		}	
	}	
	poissonRelaxKernel.setArg(2, repeat);
}

