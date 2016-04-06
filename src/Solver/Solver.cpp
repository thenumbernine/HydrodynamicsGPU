#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/Integrator/ForwardEuler.h"
#include "HydroGPU/Integrator/RungeKutta4.h"
#include "HydroGPU/Integrator/BackwardEulerConjugateGradient.h"
#include "HydroGPU/Plot/VectorField.h"
#include "HydroGPU/Plot/Plot.h"
#include "HydroGPU/Boundary/Boundary.h"
#include "HydroGPU/Equation/Equation.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/toNumericString.h"
#include "Image/System.h"
#include "Common/File.h"
#include <iomanip>

namespace HydroGPU {
namespace Solver {

Solver::CL::CL(Solver* solver_)
: solver(solver_)
, totalAlloc(0)
{}

void Solver::CL::initKernels() {
	zeroKernel = cl::Kernel(solver->program, "zero");
}

void Solver::CL::zero(cl::Buffer buffer, size_t sizeInReals) {
#ifndef AMD_SUCKS
	solver->commands.enqueueFillBuffer(buffer, 0.f, 0, sizeof(real) * sizeInReals);
#else	//because 'enqueueFillBuffer' is not working ... ???!!!!
	zeroKernel.setArg(0, buffer);
	solver->commands.enqueueNDRangeKernel(zeroKernel, solver->offset1d, cl::NDRange(sizeInReals), solver->localSize1d);
#endif
}

cl::Buffer Solver::CL::alloc(size_t size, const std::string& name) {
	totalAlloc += size;
	std::cout << "allocating gpu mem " << name << " size " << size << " running total " << totalAlloc << std::endl; 
	return cl::Buffer(solver->app->clCommon->context, CL_MEM_READ_WRITE, size);
}

Solver::Solver(HydroGPUApp* app_)
: app(app_)
, commands(app->clCommon->commands)
, frame(0)
, cl(this)
{
}

void Solver::init() {
	//we need this first, so don't trust child classes to assign it prior to calling Super::init
	//instead make them provide this method
	//TODO non-virtual init() and make it call out construction code in a particular order
	createEquation();

	cl::Device device = app->clCommon->device;
	
	// NDRanges

#if 0
	Tensor::Vector<size_t,3> localSizeVec;
	size_t maxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	std::vector<size_t> maxWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
	for (int n = 0; n < 3; ++n) {
		localSizeVec(n) = std::min<size_t>(maxWorkItemSizes[n], app->size.s[n]);
	}
	while (localSizeVec.volume() > maxWorkGroupSize) {
		for (int n = 0; n < 3; ++n) {
			localSizeVec(n) = (size_t)ceil((double)localSizeVec(n) * .5);
		}
	}
#endif	

/*
What's this for?
I was going to useGPU=false for debugging, so I could use a kernel size of 1 and output debug stuff sequentially
... but I'm getting crashes inside of kernels *only* when GPU is disabled.
SO I thought maybe I could useGPU=true and still output sequentially...
but there's currently an issue with the reduce kernel when doing that, which I need to disable.
OR I could just have the debug printfs also output their thread ID and filter all the debug output
*/
//#define DEBUG_OVERRIDE
	
	//if dim 2 is size 1 then tell opencl to treat it like a 1D problem
	switch (app->dim) {
	case 1:
		globalSize = cl::NDRange(app->size.s[0]);
		{	
			int n = app->clCommon->useGPU ? 16 : 1;
#ifdef DEBUG_OVERRIDE
			n = 1;
#endif
			localSize = cl::NDRange(n);
		}
		localSize1d = cl::NDRange(localSize[0]);
		offset1d = cl::NDRange(0);
		offsetNd = cl::NDRange(0);
		break;
	case 2:
		globalSize = cl::NDRange(app->size.s[0], app->size.s[1]);
		{
			//I never did get why, when the max work item size is 256^3, the largest local size is still just 16
			int n = app->clCommon->useGPU ? 16 : 1;
#ifdef DEBUG_OVERRIDE
			n = 1;
#endif
			localSize = cl::NDRange(n, n);
		}
		localSize1d = cl::NDRange(localSize[0]);
		offset1d = cl::NDRange(0);
		offsetNd = cl::NDRange(0, 0);
		break;
	case 3:
		globalSize = cl::NDRange(app->size.s[0], app->size.s[1], app->size.s[2]);
		{
			int n = app->clCommon->useGPU ? 8 : 1;
#ifdef AMD_SUCKS //the AMD card doesn't like having a local size of ... anything
			n = 1;
#endif
#ifdef DEBUG_OVERRIDE
			n = 1;
#endif
			localSize = cl::NDRange(n, n, n);
		}
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
		program = cl::Program(app->clCommon->context, sources);
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
	if (app->clCommon->useGPU) {
		cl_int err;
		
		size_t size = 0;
		err = clGetProgramInfo(program(), CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &size, nullptr);
		if (err != CL_SUCCESS) throw Common::Exception() << "failed to get binary size";
	
		std::vector<char> binary(size);
		err = clGetProgramInfo(program(), CL_PROGRAM_BINARIES, size, &binary[0], nullptr);
		if (err != CL_SUCCESS) throw Common::Exception() << "failed to get binary";

		Common::File::write("program.cl.bin", std::string(&binary[0], binary.size()));
	}
	
	cl.initKernels();

	initBuffers();
	initKernels();

	//create integrator
	std::string integratorName = "ForwardEuler";
	app->lua.ref()["integratorName"] >> integratorName;
	if (integratorName == "ForwardEuler") {
		integrator = std::make_shared<HydroGPU::Integrator::ForwardEuler>(this);
	} else if (integratorName == "RungeKutta4") {
		integrator = std::make_shared<HydroGPU::Integrator::RungeKutta4>(this);
	} else if (integratorName == "BackwardEulerConjugateGradient") {
		integrator = std::make_shared<HydroGPU::Integrator::BackwardEulerConjugateGradient>(this);
	} else {
		throw Common::Exception() << "failed to find integrator named " << integratorName;
	}
}


void Solver::initBuffers() {
	int volume = getVolume();

	//not necessary for fixed timestep.  TODO don't allocate in that case.
	dtBuffer = cl.alloc(sizeof(real) * volume * app->dim, "Solver::dtBuffer");
	dtSwapBuffer = cl.alloc(sizeof(real) * volume * app->dim / localSize[0], "Solver::dtSwapBuffer");
	
	stateBuffer = cl.alloc(sizeof(real) * numStates() * volume, "Solver::stateBuffer");
	
	//get the edges, so reduction doesn't
	{
		std::vector<real> dtVec(volume * app->dim);
		for (real &r : dtVec) { r = std::numeric_limits<real>::max(); }
		commands.enqueueWriteBuffer(dtBuffer, CL_TRUE, 0, sizeof(real) * volume * app->dim, &dtVec[0]);
	}
}

static std::string boundaryKernelNames[NUM_BOUNDARY_KERNELS] = {
	"Periodic",
	"Mirror",
	"Reflect",
	"FreeFlow"
};

void Solver::initKernels() {
	
	int volume = getVolume();

	boundaryKernels.resize(NUM_BOUNDARY_KERNELS);
	for (std::vector<std::vector<cl::Kernel>>& v : boundaryKernels) {
		v.resize(app->dim);
		for (std::vector<cl::Kernel>& u : v) {
			u.resize(2);
		}
	}

	std::vector<std::string> dimNames = {"X", "Y", "Z"};
	std::vector<std::string> minmaxNames = {"Min", "Max"};
	for (int boundaryIndex = 0; boundaryIndex < NUM_BOUNDARY_KERNELS; ++boundaryIndex) {
		for (int dimIndex = 0; dimIndex < app->dim; ++dimIndex) {
			for (int minmaxIndex = 0; minmaxIndex < 2; ++minmaxIndex) {
				std::string name = "stateBoundary" + boundaryKernelNames[boundaryIndex] + dimNames[dimIndex] + minmaxNames[minmaxIndex];
				cl::Kernel kernel = cl::Kernel(program, name.c_str());
				kernel.setArg(0, stateBuffer);
				kernel.setArg(1, numStates());
				boundaryKernels[boundaryIndex][dimIndex][minmaxIndex] = kernel;
			}
		}
	}
	
	findMinTimestepKernel = cl::Kernel(program, "findMinTimestep");
	CLCommon::setArgs(findMinTimestepKernel, dtBuffer, cl::Local(localSize[0] * sizeof(real)), volume * app->dim, dtSwapBuffer);	
}

std::vector<std::string> Solver::getProgramSources() {
	std::vector<std::string> sourceStrs = std::vector<std::string>{
		std::string() +
		"#define DIM " + std::to_string(app->dim) + "\n" +
		"#define SIZE_X " + std::to_string(app->size.s[0]) + "\n" +
		"#define SIZE_Y " + std::to_string(app->size.s[1]) + "\n" +
		"#define SIZE_Z " + std::to_string(app->size.s[2]) + "\n" +
		"#define STEP_X 1\n" +
		"#define STEP_Y " + std::to_string(app->size.s[0]) + "\n" +
		"#define STEP_Z " + std::to_string(app->size.s[0] * app->size.s[1]) + "\n" +
		"#define STEP_W " + std::to_string(app->size.s[0] * app->size.s[1] * app->size.s[2]) + "\n" +
		"#define DX " + toNumericString<real>(app->dx.s[0]) + "\n" +
		"#define DY " + toNumericString<real>(app->dx.s[1]) + "\n" +
		"#define DZ " + toNumericString<real>(app->dx.s[2]) + "\n" +
		"#define XMIN " + toNumericString<real>(app->xmin.s[0]) + "\n" +
		"#define YMIN " + toNumericString<real>(app->xmin.s[1]) + "\n" +
		"#define ZMIN " + toNumericString<real>(app->xmin.s[2]) + "\n" +
		"#define XMAX " + toNumericString<real>(app->xmax.s[0]) + "\n" +
		"#define YMAX " + toNumericString<real>(app->xmax.s[1]) + "\n" +
		"#define ZMAX " + toNumericString<real>(app->xmax.s[2]) + "\n" +
		"#define NUM_STATES " + std::to_string(numStates()) + "\n" +
		"#define NUM_FLUX_STATES "+std::to_string(getNumFluxStates())+"\n"
	};

	std::string slopeLimiterName = "Superbee";
	app->lua.ref()["slopeLimiter"] >> slopeLimiterName;
	sourceStrs[0] += "#define SLOPE_LIMITER_" + slopeLimiterName + "\n";

	real gravitationalConstant = 1.f;
	app->lua.ref()["gravitationalConstant"] >> gravitationalConstant;
	sourceStrs[0] += "#define GRAVITATIONAL_CONSTANT " + toNumericString<real>(gravitationalConstant) + "\n";

	sourceStrs.push_back("#include \"SlopeLimiter.cl\"\n");
	sourceStrs.push_back("#include \"Common.cl\"\n");
	equation->getProgramSources(sourceStrs);
	return sourceStrs;
}

std::shared_ptr<Solver::Converter> Solver::createConverter() {
	return std::make_shared<Converter>(this);
}

Solver::Converter::Converter(Solver* solver_)
: solver(solver_)
, stateVec(solver_->getVolume() * solver_->numStates()) {}

int Solver::Converter::numChannels() {
	return solver->equation->numReadStateChannels();
}

void Solver::Converter::setValues(int index, const std::vector<real>& cellValues) {
	solver->equation->readStateCell(stateVec.data() + index * solver->numStates(), cellValues.data());
}

void Solver::Converter::toGPU() {
	//write state density first for gravity potential, to then update energy
	solver->commands.enqueueWriteBuffer(solver->stateBuffer, CL_TRUE, 0, sizeof(real) * solver->numStates() * solver->getVolume(), stateVec.data());
	solver->commands.finish();
}

void Solver::Converter::fromGPU() {
	solver->commands.enqueueReadBuffer(solver->stateBuffer, CL_TRUE, 0, sizeof(real) * solver->numStates() * solver->getVolume(), stateVec.data());
	solver->commands.finish();
}

real Solver::Converter::getValue(int index, int channel) {
	int numStates = solver->numStates();
	if (channel < numStates) return stateVec[channel + numStates * index];
	return std::nan("");
}

void Solver::resetState() {
	if (!app->lua.ref()["initState"].isFunction()) throw Common::Exception() << "expected initState to be defined in config file";
	std::cout << "initializing..." << std::endl;
	
	std::shared_ptr<Converter> converter = createConverter();
	std::vector<real> cellResults(converter->numChannels());

	int flattenedIndex = 0;
	int index[3];
	for (index[2] = 0; index[2] < app->size.s[2]; ++index[2]) {
		for (index[1] = 0; index[1] < app->size.s[1]; ++index[1]) {
			for (index[0] = 0; index[0] < app->size.s[0]; ++index[0], ++flattenedIndex ) {
				real4 pos;
				for (int i = 0; i < 3; ++i) {
					pos.s[i] = real(app->xmax.s[i] - app->xmin.s[i]) * (real(index[i]) + .5) / real(app->size.s[i]) + real(app->xmin.s[i]);
				}
				pos.s[3] = 0;
			
				LuaCxx::Stack stack = app->lua.stack();
				
				stack
				.getGlobal("initState")
				.push(pos.s[0], pos.s[1], pos.s[2])
				.call(3, cellResults.size());	
				
				for (int i = (int)cellResults.size()-1; i >= 0; --i) {
					cellResults[i] = real();
					stack.pop(cellResults[i]);
				}
				converter->setValues(flattenedIndex, cellResults);
			}
		}
	}
	std::cout << "...done" << std::endl;

	//grad^2 Phi = - 4 pi G rho
	//solve inverse discretized linear system to find Psi
	//D_ij / (-4 pi G) Phi_j = rho_i
	//once you get that, plug it into the total energy

	//pass context to child class 
	converter->toGPU();
}

int Solver::numStates() {
	return (int)equation->states.size();
}

/*
flux vector.  typically equal to the state vector size
(when used with the default finite-volume integrator)
I should move the flux allocation down to this class,
or put an intermediate finite-volume solver class with the flux information.
*/
int Solver::getNumFluxStates() {
	return numStates();
}

int Solver::getVolume() {
	return app->size.s[0] * app->size.s[1] * app->size.s[2];
}

void Solver::getBoundaryRanges(int dimIndex, cl::NDRange &offset, cl::NDRange &global, cl::NDRange &local) {
	switch (app->dim) {
	case 1:
		offset = offset1d;
		local = localSize1d;
		global = cl::NDRange(app->size.s[dimIndex]);
		break;
	case 2:
		offset = offset1d;
		local = localSize1d;
		global = cl::NDRange(app->size.s[!dimIndex]);
		break;
	case 3:
		offset = cl::NDRange(0, 0);
		local = cl::NDRange(localSize[0], localSize[1]);
		switch (dimIndex) {
		case 0:
			global = cl::NDRange(app->size.s[1], app->size.s[2]);
			break;
		case 1:
			global = cl::NDRange(app->size.s[0], app->size.s[2]);
			break;
		case 2:
			global = cl::NDRange(app->size.s[0], app->size.s[1]);
			break;
		default:
			throw Common::Exception() << "can't handle dim " << dimIndex;
		}
		break;
	default:
		throw Common::Exception() << "can't handle dim " << dimIndex;
	}
}

//on AMD, 2D problem boundaries <512 work fine (once variables are manually inlined in the kernels).
// beyond 512 gets mysery errors.
void Solver::boundary() {
#if 1	//OpenCL
	cl::NDRange offset, global, local;
	for (int i = 0; i < app->dim; ++i) {
		getBoundaryRanges(i, offset, global, local);
		for (int j = 0; j < numStates(); ++j) {
			for (int minmax = 0; minmax < 2; ++minmax) {
				int boundaryKernelIndex = equation->stateGetBoundaryKernelForBoundaryMethod(i, j, minmax);
				if (boundaryKernelIndex < 0 || boundaryKernelIndex >= boundaryKernels.size()) continue;
				cl::Kernel& kernel = boundaryKernels[boundaryKernelIndex][i][minmax];
				kernel.setArg(2, j);
				commands.enqueueNDRangeKernel(kernel, offset, global, local);
			}
		}
	}
#else	//CPU debugging on AMD ...
		//behaves exactly the same -- with the same errors on the read cells than the GPU version
		// this makes me think the errors are coming from somewhere beforehond
		//only works for 2D, only handles NONE & PERIODIC
	assert(app->dim == 2);
	
	std::vector<real> state(getVolume() * numStates());
	commands.enqueueReadBuffer(stateBuffer, CL_TRUE, 0, sizeof(real) * getVolume() * numStates(), state.data());

	int SIZE_X = app->size.s[0];
	int SIZE_Y = app->size.s[1];
	int spacing = numStates();
	for (int i = 0; i < SIZE_Y; ++i) {
		switch (app->boundaryMethods(0,0)) {	//x min
		case BOUNDARY_METHOD_NONE:
			break;
		case BOUNDARY_METHOD_PERIODIC:
			for (int offset = 0; offset < numStates(); ++offset) {
				state[offset + spacing * INDEX(0, i, 0)] = state[offset + spacing * INDEX(SIZE_X - 4, i, 0)];
				state[offset + spacing * INDEX(1, i, 0)] = state[offset + spacing * INDEX(SIZE_X - 3, i, 0)];
			}
			break;
		}
		switch (app->boundaryMethods(0,1)) {	//x max
		case BOUNDARY_METHOD_NONE:
			break;
		case BOUNDARY_METHOD_PERIODIC:
			for (int offset = 0; offset < numStates(); ++offset) {
				state[offset + spacing * INDEX(SIZE_X - 2, i, 0)] = state[offset + spacing * INDEX(2, i, 0)];
				state[offset + spacing * INDEX(SIZE_X - 1, i, 0)] = state[offset + spacing * INDEX(3, i, 0)];
			}
			break;
		}
	}
	for (int i = 0; i < SIZE_X; ++i) {
		switch (app->boundaryMethods(1,0)) {	//y min
		case BOUNDARY_METHOD_NONE:
			break;
		case BOUNDARY_METHOD_PERIODIC:
			for (int offset = 0; offset < numStates(); ++offset) {
				state[offset + spacing * INDEX(i, 0, 0)] = state[offset + spacing * INDEX(i, SIZE_Y - 4, 0)];
				state[offset + spacing * INDEX(i, 1, 0)] = state[offset + spacing * INDEX(i, SIZE_Y - 3, 0)];
			}
			break;
		}
		switch (app->boundaryMethods(1,1)) {	//y max
		case BOUNDARY_METHOD_NONE:
			break;
		case BOUNDARY_METHOD_PERIODIC:
			for (int offset = 0; offset < numStates(); ++offset) {
				state[offset + spacing * INDEX(i, SIZE_Y - 2, 0)] = state[offset + spacing * INDEX(i, 2, 0)];
				state[offset + spacing * INDEX(i, SIZE_Y - 1, 0)] = state[offset + spacing * INDEX(i, 3, 0)];
			}
			break;
		}
	}
	
	commands.enqueueWriteBuffer(stateBuffer, CL_TRUE, 0, sizeof(real) * getVolume() * numStates(), state.data());
#endif
}

real Solver::findMinTimestep() {
	int reduceSize = getVolume();
	cl::Buffer dst = dtSwapBuffer;
	cl::Buffer src = dtBuffer;
	while (reduceSize > 1) {
		int nextSize = (reduceSize >> 4) + !!(reduceSize & ((1 << 4) - 1));
		cl::NDRange reduceGlobalSize(std::max<int>(reduceSize, localSize[0]));
		findMinTimestepKernel.setArg(0, src);
		findMinTimestepKernel.setArg(2, reduceSize);
		findMinTimestepKernel.setArg(3, dst);
		commands.enqueueNDRangeKernel(findMinTimestepKernel, offset1d, reduceGlobalSize, localSize1d);
		if (app->clCommon->useGPU) commands.finish();
		std::swap(dst, src);
		reduceSize = nextSize;
	}
	real dt = real();
	commands.enqueueReadBuffer(src, CL_TRUE, 0, sizeof(real), &dt);
	return dt * app->cfl;
}

void Solver::initStep() {
}

void Solver::update() {
	//commands.enqueueNDRangeKernel(addSourceKernel, offsetNd, globalSize, localSize, nullptr, &addSourceEvent.clEvent);

	boundary();
	
	initStep();

	real dt = app->useFixedDT ? app->fixedDT : calcTimestep();

	if (app->showTimestep) {
		std::cout << "dt " << dt << std::endl;
	}

	step(dt);

	++frame;
/* 
	for (EventProfileEntry *entry : entries) {
std::cout << "event " << entry->name << std::endl;
		cl_ulong start = entry->clEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong end = entry->clEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
		entry->stat.accum((double)(end - start) * 1e-9);
	}
*/
}

void Solver::screenshot() {
	for (int i = 0; i < 10000; ++i) {
		std::stringstream ss;
		ss << "screenshot" << std::setw(5) << std::setfill('0') << i << ".png";
		std::string filename = ss.str();
		if (!Common::File::exists(filename)) {
			app->plot->screenshot(filename);
			return;
		}
	}
	throw Common::Exception() << "couldn't find an available filename";
}

//returns the first available index
//uses the first channel name to test for existence
std::vector<std::string> Solver::getSaveChannelNames() {
	return equation->states;
}

int Solver::getSaveIndex() {
	std::string firstChannelName = getSaveChannelNames()[0];
	for (int i = 0; i < 1000000; ++i) {
		std::string filename = firstChannelName + std::to_string(i) + ".fits";
		if (!Common::File::exists(filename)) return i;
	}
	throw Common::Exception() << "failed to find available save filename";
}


void Solver::save() {
	int saveIndex = getSaveIndex();
	
	std::shared_ptr<Converter> converter = createConverter();
	converter->fromGPU();

	std::vector<std::string> channelNames = getSaveChannelNames(); 
	
	//hmm, rather than a plane per variable, now that I'm saving 3D stuff,
	// how about a plane per 3rd dim, and separate save files per variable?
	std::shared_ptr<Image::ImageType<float>> image = std::make_shared<Image::ImageType<float>>(Tensor::Vector<int,2>(app->size.s[0], app->size.s[1]), nullptr, 1, app->size.s[2]);
		
	for (int channel = 0; channel < channelNames.size(); ++channel) {
		for (int z = 0; z < app->size.s[2]; ++z) {	
			for (int y = 0; y < app->size.s[1]; ++y) {
				for (int x = 0; x < app->size.s[0]; ++x) {
					int cellIndex = x + app->size.s[0] * (y + app->size.s[1] * z);
					real value = converter->getValue(cellIndex, channel);
					(*image)(x,y,0,z) = value;
				}
			}
		}
		std::string filename = channelNames[channel] + std::to_string(saveIndex) + ".fits";
		std::cout << "saving file " << filename << std::endl;
		Image::system->write(filename, image); 
	}
}

}
}
