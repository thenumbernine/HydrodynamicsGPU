#include "HydroGPU/Shared/Common.h"	//OpenCL shared header
#include "HydroGPU/BurgersSolver.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Image/System.h"
#include "Image/FITS_IO.h"
#include "Common/Exception.h"
#include "Common/Macros.h"
#include "Common/File.h"
#include "Tensor/Vector.h"
#include <OpenGL/gl.h>
#include <fstream>

BurgersSolver::BurgersSolver(
	HydroGPUApp &app_,
	std::vector<real4> stateVec)
: app(app_)
, calcCFLEvent("calcCFL")
, calcCFLMinReduceEvent("calcCFLMinReduce")
, calcInterfaceVelocityEvent("calcInterfaceVelocity")
, calcFluxEvent("calcFlux")
, integrateFluxEvent("integrateFlux")
, computePressureEvent("computePressure")
, diffuseMomentumEvent("diffuseMomentum")
, diffuseWorkEvent("diffuseWork")
, addSourceEvent("addSource")
, poissonRelaxEvent("poissonRelax")
, addGravityEvent("addGravity")
{
	stateBoundaryKernels.resize(NUM_BOUNDARY_METHODS);
	for (std::vector<cl::Kernel> &v : stateBoundaryKernels) {
		v.resize(DIM);
	}
	
	cl::Device device = app.device;
	cl::Context context = app.context;
	cl::CommandQueue commands = app.commands;
	cl::ImageGL fluidTexMem = app.fluidTexMem;
	cl::ImageGL gradientTexMem = app.gradientTexMem;
	real2 xmin = app.xmin;
	real2 xmax = app.xmax;
	bool useGPU = app.useGPU;
	
	if (!app.useFixedDT) {
		entries.push_back(&calcCFLEvent);
		entries.push_back(&calcCFLMinReduceEvent);
	}
	entries.push_back(&calcInterfaceVelocityEvent);
	entries.push_back(&calcFluxEvent);
	entries.push_back(&integrateFluxEvent);
	entries.push_back(&computePressureEvent);
	entries.push_back(&diffuseMomentumEvent);
	entries.push_back(&diffuseWorkEvent);
	if (app.useGravity) {
		entries.push_back(&poissonRelaxEvent);
		entries.push_back(&addGravityEvent);
	}

	size_t maxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	std::vector<size_t> maxWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
	Tensor::Vector<size_t,DIM> localSizeVec;
	for (int n = 0; n < DIM; ++n) {
		localSizeVec(n) = std::min<size_t>(maxWorkItemSizes[n], app.size(n));
	}
	while (localSizeVec.volume() > maxWorkGroupSize) {
		for (int n = 0; n < DIM; ++n) {
			localSizeVec(n) = (size_t)ceil((double)localSizeVec(n) * .5);
		}
	}
	//hmm...
	if (!useGPU) localSizeVec(0) >>= 1;
	std::cout << "global_size\t" << app.size << std::endl;
	std::cout << "local_size\t" << localSizeVec << std::endl;

	globalSize = cl::NDRange(app.size(0), app.size(1));
	localSize = cl::NDRange(localSizeVec(0), localSizeVec(1));

	std::vector<std::string> kernelSources = std::vector<std::string>{
		std::string() + "#define GAMMA " + std::to_string(app.gamma) + "f\n",
		Common::File::read("Common.cl"),
		Common::File::read("Burgers.cl")
	};
	std::vector<std::pair<const char *, size_t>> sources;
	for (const std::string &s : kernelSources) {
		sources.push_back(std::pair<const char *, size_t>(s.c_str(), s.length()));
	}
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

	int volume = app.size.volume();
	int volumeWithGhost = app.sizeWithGhost.volume();
	std::cout << "size " << app.size << std::endl;
	std::cout << "size with ghost " << app.sizeWithGhost << std::endl;
	std::cout << "num ghost cells " << NUM_GHOST_CELLS << std::endl;
	std::cout << "volume " << volume << std::endl;
	std::cout << "volume with ghost " << volumeWithGhost << std::endl;

	//use ghost cells
	stateBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real4) * volumeWithGhost);
	interfaceVelocityBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real2) * volumeWithGhost);
	fluxBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real4) * volumeWithGhost * 2);
	pressureBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volumeWithGhost);
	gravityPotentialBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volumeWithGhost);
	
	//don't use ghost cells
	cflBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume);
	cflSwapBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume / localSize[0]);
	dtBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real));

	commands.enqueueWriteBuffer(stateBuffer, CL_TRUE, 0, sizeof(real4) * volumeWithGhost, &stateVec[0]);

	//get the edges, so reduction doesn't
	{
		std::vector<real> cflVec(volume);
		for (real &r : cflVec) { r = std::numeric_limits<real>::max(); }
		commands.enqueueWriteBuffer(cflBuffer, CL_TRUE, 0, sizeof(real) * volume, &cflVec[0]);
	}

	//here's our initial guess to sor
	std::vector<real> gravityPotentialVec(volumeWithGhost);
	for (size_t i = 0; i < volumeWithGhost; ++i) {
		if (app.useGravity) {
			gravityPotentialVec[i] = stateVec[i].s[0];
		} else {
			gravityPotentialVec[i] = 0.;
		}
	}
	commands.enqueueWriteBuffer(gravityPotentialBuffer, CL_TRUE, 0, sizeof(real) * volumeWithGhost, &gravityPotentialVec[0]);

	if (app.useFixedDT) {
		commands.enqueueWriteBuffer(dtBuffer, CL_TRUE, 0, sizeof(real), &app.fixedDT);
	}

	real2 dx;
	for (int i = 0; i < DIM; ++i) {
		dx.s[i] = (xmax.s[i] - xmin.s[i]) / (float)app.size(i);
	}
	std::cout << "xmin " << xmin.s[0] << ", " << xmin.s[1] << std::endl;
	std::cout << "xmax " << xmax.s[0] << ", " << xmax.s[1] << std::endl;
	std::cout << "size " << app.size << std::endl;
	std::cout << "dx " << dx.s[0] << ", " << dx.s[1] << std::endl;
	
	cl_int2 size;
	size.s[0] = app.size(0);
	size.s[1] = app.size(1);

	calcCFLKernel = cl::Kernel(program, "calcCFL");
	app.setArgs(calcCFLKernel, cflBuffer, stateBuffer, gravityPotentialBuffer, size, dx, app.cfl);

	calcCFLMinReduceKernel = cl::Kernel(program, "calcCFLMinReduce");
	app.setArgs(calcCFLMinReduceKernel, cflBuffer, cl::Local(localSizeVec(0) * sizeof(real)), volume, cflSwapBuffer);

	for (int boundaryIndex = 0; boundaryIndex < NUM_BOUNDARY_METHODS; ++boundaryIndex) {
		for (int side = 0; side < DIM; ++side) {
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
				name += "Horizontal";
				break;
			case 1:
				name += "Vertical";
				break;
			default:
				throw Common::Exception() << "no kernel for boundary side " << side;
			}
			stateBoundaryKernels[boundaryIndex][side] = cl::Kernel(program, name.c_str());
			app.setArgs(stateBoundaryKernels[boundaryIndex][side], stateBuffer, size);
		}
	}
	
	calcInterfaceVelocityKernel = cl::Kernel(program, "calcInterfaceVelocity");
	app.setArgs(calcInterfaceVelocityKernel, interfaceVelocityBuffer, stateBuffer, size, dx);
	
	calcInterfaceVelocityHorizontalKernel = cl::Kernel(program, "calcInterfaceVelocityHorizontal");
	app.setArgs(calcInterfaceVelocityHorizontalKernel, interfaceVelocityBuffer, stateBuffer, size, dx, app.size(1));
	
	calcInterfaceVelocityVerticalKernel = cl::Kernel(program, "calcInterfaceVelocityVertical");
	app.setArgs(calcInterfaceVelocityVerticalKernel, interfaceVelocityBuffer, stateBuffer, size, dx, app.size(0));

	calcFluxKernel = cl::Kernel(program, "calcFlux");
	app.setArgs(calcFluxKernel, fluxBuffer, stateBuffer, interfaceVelocityBuffer, size, dx, dtBuffer);
	
	calcFluxHorizontalKernel = cl::Kernel(program, "calcFluxHorizontal");
	app.setArgs(calcFluxHorizontalKernel, fluxBuffer, stateBuffer, interfaceVelocityBuffer, size, dx, dtBuffer, app.size(1));
	
	calcFluxVerticalKernel = cl::Kernel(program, "calcFluxVertical");
	app.setArgs(calcFluxVerticalKernel, fluxBuffer, stateBuffer, interfaceVelocityBuffer, size, dx, dtBuffer, app.size(0));
	
	integrateFluxKernel = cl::Kernel(program, "integrateFlux");
	app.setArgs(integrateFluxKernel, stateBuffer, fluxBuffer, size, dx, dtBuffer);
	
	poissonRelaxKernel = cl::Kernel(program, "poissonRelax");
	app.setArgs(poissonRelaxKernel, gravityPotentialBuffer, stateBuffer, size, dx);
	
	computePressureKernel = cl::Kernel(program, "computePressure");
	app.setArgs(computePressureKernel, pressureBuffer, stateBuffer, gravityPotentialBuffer, size);
	
	computePressureHorizontalKernel = cl::Kernel(program, "computePressureHorizontal");
	app.setArgs(computePressureHorizontalKernel, pressureBuffer, stateBuffer, gravityPotentialBuffer, size);
	
	computePressureVerticalKernel = cl::Kernel(program, "computePressureVertical");
	app.setArgs(computePressureVerticalKernel, pressureBuffer, stateBuffer, gravityPotentialBuffer, size);

	addGravityKernel = cl::Kernel(program, "addGravity");
	app.setArgs(addGravityKernel, stateBuffer, gravityPotentialBuffer, size, dx, dtBuffer);

	diffuseMomentumKernel = cl::Kernel(program, "diffuseMomentum");
	app.setArgs(diffuseMomentumKernel, stateBuffer, pressureBuffer, size, dx, dtBuffer);
	
	diffuseWorkKernel = cl::Kernel(program, "diffuseWork");
	app.setArgs(diffuseWorkKernel, stateBuffer, pressureBuffer, size, dx, dtBuffer);
	
	convertToTexKernel = cl::Kernel(program, "convertToTex");
	app.setArgs(convertToTexKernel, stateBuffer, gravityPotentialBuffer, size, fluidTexMem, gradientTexMem);

	addDropKernel = cl::Kernel(program, "addDrop");
	app.setArgs(addDropKernel, stateBuffer, size, xmin, xmax);

	addSourceKernel = cl::Kernel(program, "addSource");
	app.setArgs(addSourceKernel, stateBuffer, size, xmin, xmax, dtBuffer);
	
	//grad^2 Phi = - 4 pi G rho
	//solve inverse discretized linear system to find Psi
	//D_ij / (-4 pi G) Phi_j = rho_i
	//once you get that, plug it into the total energy

	if (app.useGravity) {
		//solve for gravitational potential via gauss seidel
		cl::NDRange offset2d(0, 0);
		for (int i = 0; i < 20; ++i) {
			apply1DBoundary(gravityPotentialBuffer);
			commands.enqueueNDRangeKernel(poissonRelaxKernel, offset2d, globalSize, localSize);
		}

		if (stateVec.size() != volumeWithGhost) throw Common::Exception() << "stateVec is of incorrect size";
		//update internal energy
		for (int i = 0; i < volumeWithGhost; ++i) {
			stateVec[i].s[3] += gravityPotentialVec[i];
		}
		commands.enqueueWriteBuffer(stateBuffer, CL_TRUE, 0, sizeof(real4) * volumeWithGhost, &stateVec[0]);
	}	
}

BurgersSolver::~BurgersSolver() {
	std::cout << "OpenCL profiling info:" << std::endl;
	for (EventProfileEntry *entry : entries) {
		std::cout << entry->name << "\t" 
			<< "duration " << entry->stat
			<< std::endl;
	}
}

void BurgersSolver::apply1DBoundary(cl::Buffer buffer) {
	switch (app.boundaryMethod) {
	case BOUNDARY_PERIODIC:
	case BOUNDARY_MIRROR:
		//same same
		
		break;
	case BOUNDARY_FREEFLOW:
		break;
	default:
		throw Common::Exception() << "got unknown boundary method " << app.boundaryMethod;
	}
}

void BurgersSolver::update() {
	cl::CommandQueue commands = app.commands;
	cl::ImageGL fluidTexMem = app.fluidTexMem;
	cl_int2 size;
	size.s[0] = app.size(0);
	size.s[1] = app.size(1);
	bool useGPU = app.useGPU;

	cl::NDRange offset1d(0);
	cl::NDRange offset2d(0, 0);
	cl::NDRange localSize1d(localSize[0]);
	cl::NDRange globalWidth(size.s[0]);
	cl::NDRange globalHeight(size.s[1]);
	
	//commands.enqueueNDRangeKernel(addSourceKernel, offset2d, globalSize, localSize, NULL, &addSourceEvent.clEvent);
	if (!app.useFixedDT) {
		commands.enqueueNDRangeKernel(calcCFLKernel, offset2d, globalSize, localSize, NULL, &calcCFLEvent.clEvent);
		
		int reduceSize = globalSize[0] * globalSize[1];
		cl::Buffer dst = cflSwapBuffer;
		cl::Buffer src = cflBuffer;
		while (reduceSize > 1) {
			int nextSize = (reduceSize >> 4) + !!(reduceSize & ((1 << 4) - 1));
			cl::NDRange offset1d(0);
			cl::NDRange reduceGlobalSize(std::max<int>(reduceSize, localSize[0]));
			calcCFLMinReduceKernel.setArg(0, src);
			calcCFLMinReduceKernel.setArg(2, reduceSize);
			calcCFLMinReduceKernel.setArg(3, nextSize == 1 ? dtBuffer : dst);
			commands.enqueueNDRangeKernel(calcCFLMinReduceKernel, offset1d, reduceGlobalSize, localSize1d, NULL, nextSize > 1 ? nullptr : &calcCFLMinReduceEvent.clEvent);
			commands.flush();
			commands.finish();
			std::swap(dst, src);
			reduceSize = nextSize;
		}
	}

	//boundary
	commands.enqueueNDRangeKernel(stateBoundaryKernels[app.boundaryMethod][0], offset1d, globalWidth, localSize1d);
	commands.enqueueNDRangeKernel(stateBoundaryKernels[app.boundaryMethod][1], offset1d, globalHeight, localSize1d);	

	//interface velocity
	commands.enqueueNDRangeKernel(calcInterfaceVelocityKernel, offset2d, globalSize, localSize, NULL, &calcInterfaceVelocityEvent.clEvent);
	commands.enqueueNDRangeKernel(calcInterfaceVelocityHorizontalKernel, offset1d, globalWidth, localSize1d);
	commands.enqueueNDRangeKernel(calcInterfaceVelocityVerticalKernel, offset1d, globalHeight, localSize1d);
	
	//flux
	commands.enqueueNDRangeKernel(calcFluxKernel, offset2d, globalSize, localSize, NULL, &calcFluxEvent.clEvent);
	commands.enqueueNDRangeKernel(calcFluxHorizontalKernel, offset1d, globalWidth, localSize1d);
	commands.enqueueNDRangeKernel(calcFluxVerticalKernel, offset1d, globalHeight, localSize1d);
	
	commands.enqueueNDRangeKernel(integrateFluxKernel, offset2d, globalSize, localSize, NULL, &integrateFluxEvent.clEvent);

	if (app.useGravity) {
		//recompute poisson solution to gravitational potential
		const int maxIter = 20;
		for (int i = 0; i < maxIter; ++i) {
			apply1DBoundary(gravityPotentialBuffer);
			commands.enqueueNDRangeKernel(poissonRelaxKernel, offset2d, globalSize, localSize, NULL, i < maxIter - 1 ? nullptr : &poissonRelaxEvent.clEvent);
		}
	}	

	//boundary
	commands.enqueueNDRangeKernel(stateBoundaryKernels[app.boundaryMethod][0], offset1d, globalWidth, localSize1d);
	commands.enqueueNDRangeKernel(stateBoundaryKernels[app.boundaryMethod][1], offset1d, globalHeight, localSize1d);	

	commands.enqueueNDRangeKernel(computePressureKernel, offset2d, globalSize, localSize, NULL, &computePressureEvent.clEvent);
	computePressureHorizontalKernel.setArg(4, -1);
	commands.enqueueNDRangeKernel(computePressureHorizontalKernel, offset1d, globalWidth, localSize1d);
	computePressureHorizontalKernel.setArg(4, app.size(1));
	commands.enqueueNDRangeKernel(computePressureHorizontalKernel, offset1d, globalWidth, localSize1d);
	computePressureVerticalKernel.setArg(4, -1);
	commands.enqueueNDRangeKernel(computePressureVerticalKernel, offset1d, globalHeight, localSize1d);
	computePressureVerticalKernel.setArg(4, app.size(0));
	commands.enqueueNDRangeKernel(computePressureVerticalKernel, offset1d, globalHeight, localSize1d);
	
	if (app.useGravity) {
		commands.enqueueNDRangeKernel(addGravityKernel, offset2d, globalSize, localSize, NULL, &addGravityEvent.clEvent);	
		//boundary
		commands.enqueueNDRangeKernel(stateBoundaryKernels[app.boundaryMethod][0], offset1d, globalWidth, localSize1d);
		commands.enqueueNDRangeKernel(stateBoundaryKernels[app.boundaryMethod][1], offset1d, globalHeight, localSize1d);	
	}
		
	commands.enqueueNDRangeKernel(diffuseMomentumKernel, offset2d, globalSize, localSize, NULL, &diffuseMomentumEvent.clEvent);

	//boundary
	commands.enqueueNDRangeKernel(stateBoundaryKernels[app.boundaryMethod][0], offset1d, globalWidth, localSize1d);
	commands.enqueueNDRangeKernel(stateBoundaryKernels[app.boundaryMethod][1], offset1d, globalHeight, localSize1d);	
	
	commands.enqueueNDRangeKernel(diffuseWorkKernel, offset2d, globalSize, localSize, NULL, &diffuseWorkEvent.clEvent);

	glFlush();
	glFinish();
	
	std::vector<cl::Memory> acquireGLMems = {fluidTexMem};
	commands.enqueueAcquireGLObjects(&acquireGLMems);

	if (useGPU) {
		convertToTexKernel.setArg(5, app.displayMethod);
		convertToTexKernel.setArg(6, app.displayScale);
		commands.enqueueNDRangeKernel(convertToTexKernel, offset2d, globalSize, localSize);
	} else {
		int volume = app.size.volume();
		int volumeWithGhost = app.sizeWithGhost.volume();
		std::vector<real4> stateVec(volumeWithGhost);
		commands.enqueueReadBuffer(stateBuffer, CL_TRUE, 0, sizeof(real4) * volumeWithGhost, &stateVec[0]);  
		std::vector<Tensor::Vector<char,4>> texVec(volume);
		for (int y = 0; y < app.size(1); ++y) {
			for (int x = 0; x < app.size(0); ++x) {
				int ig = (x + NUM_GHOST_CELLS) + app.sizeWithGhost(0) * (y + NUM_GHOST_CELLS);
				real value;
				switch (app.displayMethod) {
				case DISPLAY_DENSITY:	//density
					value = stateVec[ig].s[0];
					break;
				case DISPLAY_VELOCITY:	//velocity
					value = sqrt(stateVec[ig].s[1] * stateVec[ig].s[1] + stateVec[ig].s[2] * stateVec[ig].s[2]) / stateVec[ig].s[0];
					break;
				case DISPLAY_PRESSURE:	//pressure
					value = (app.gamma - 1.f) * stateVec[ig].s[3] * stateVec[ig].s[0];
					break;
				default:
					value = .5f;
					break;
				}		
				int i = x + app.size(0) * y;
				texVec[i](0) = (char)(255.f * value * app.displayScale);
			}
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

void BurgersSolver::addDrop(Tensor::Vector<float,2> pos, Tensor::Vector<float,2> vel) {
	cl::NDRange offset2d(0, 0);
	addSourcePos.s[0] = pos(0);
	addSourcePos.s[1] = pos(1);
	addSourceVel.s[0] = vel(0);
	addSourceVel.s[1] = vel(1);
	addDropKernel.setArg(4, addSourcePos);
	addDropKernel.setArg(5, addSourceVel);
	commands.enqueueNDRangeKernel(addDropKernel, offset2d, globalSize, localSize);
}

void BurgersSolver::screenshot() {
	for (int i = 0; i < 1000; ++i) {
		std::string filename = std::string("screenshot") + std::to_string(i) + ".png";
		if (!Common::File::exists(filename)) {
			std::shared_ptr<Image::Image> image = std::make_shared<Image::Image>(app.size, nullptr, 3);
			glBindTexture(GL_TEXTURE_2D, app.fluidTex);
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, image->getData());
			glBindTexture(GL_TEXTURE_2D, 0);
			
			Image::system->write(filename, image);
			return;
		}
	}
	throw Common::Exception() << "couldn't find an available filename";
}

void BurgersSolver::save() {
	for (int i = 0; i < 1000; ++i) {
		std::string filename = std::string("save") + std::to_string(i) + ".fits";
		if (!Common::File::exists(filename)) {
			std::shared_ptr<Image::ImageType<float>> image = std::make_shared<Image::ImageType<float>>(app.sizeWithGhost, nullptr, 1, 5);
			
			std::vector<real4> stateVec(app.sizeWithGhost.volume());
			app.commands.enqueueReadBuffer(stateBuffer, CL_TRUE, 0, sizeof(real4) * stateVec.size(), &stateVec[0]);
				
			std::vector<real> gravVec(app.sizeWithGhost.volume());
			app.commands.enqueueReadBuffer(gravityPotentialBuffer, CL_TRUE, 0, sizeof(real) * gravVec.size(), &gravVec[0]);
			
			app.commands.flush();
			app.commands.finish();
				
			for (int j = 0; j < app.sizeWithGhost(1); ++j) {
				for (int i = 0; i < app.sizeWithGhost(0); ++i) {
					real4 *state = &stateVec[i + app.sizeWithGhost(0) * j];
					real grav = gravVec[i + app.sizeWithGhost(0) * j];
					(*image)(i,j,0,0) = state->s[0];
					(*image)(i,j,0,1) = state->s[1] / state->s[0];
					(*image)(i,j,0,2) = state->s[2] / state->s[0];
					(*image)(i,j,0,3) = state->s[3] / state->s[0];
					(*image)(i,j,0,4) = grav;
				}
			}
			Image::system->write(filename, image); 
			return;
		}
	}
	throw Common::Exception() << "couldn't find an available filename";
}

