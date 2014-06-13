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
, applyBoundaryHorizontalEvent("applyBoundaryHorizontal")
, applyBoundaryVerticalEvent("applyBoundaryVertical")
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
	cl::Device device = app.device;
	cl::Context context = app.context;
	cl::CommandQueue commands = app.commands;
	cl::ImageGL fluidTexMem = app.fluidTexMem;
	cl::ImageGL gradientTexMem = app.gradientTexMem;
	real2 xmin = app.xmin;
	real2 xmax = app.xmax;
	cl_int2 size = app.size;
	bool useGPU = app.useGPU;
	
	if (!app.useFixedDT) {
		entries.push_back(&calcCFLEvent);
		entries.push_back(&calcCFLMinReduceEvent);
	}
	entries.push_back(&applyBoundaryHorizontalEvent);
	entries.push_back(&applyBoundaryVerticalEvent);
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
		localSizeVec(n) = std::min<size_t>(maxWorkItemSizes[n], size.s[n]);
	}
	while (localSizeVec.volume() > maxWorkGroupSize) {
		for (int n = 0; n < DIM; ++n) {
			localSizeVec(n) = (size_t)ceil((double)localSizeVec(n) * .5);
		}
	}
	//hmm...
	if (!useGPU) localSizeVec(0) >>= 1;
	std::cout << "global_size\t" << size.s[0] << ", " << size.s[1] << std::endl;
	std::cout << "local_size\t" << localSizeVec << std::endl;

	globalSize = cl::NDRange(size.s[0], size.s[1]);
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

	int volume = size.s[0] * size.s[1];

	stateBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real4) * volume);
	interfaceVelocityBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real2) * volume);
	fluxBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real4) * volume * 2);
	pressureBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume);
	cflBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume);
	cflSwapBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume / localSize[0]);
	gravityPotentialBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume);
	dtBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real));

	commands.enqueueWriteBuffer(stateBuffer, CL_TRUE, 0, sizeof(real4) * volume, &stateVec[0]);

	//get the edges, so reduction doesn't
	{
		std::vector<real> cflVec(volume);
		for (real &r : cflVec) { r = std::numeric_limits<real>::max(); }
		commands.enqueueWriteBuffer(cflBuffer, CL_TRUE, 0, sizeof(real) * volume, &cflVec[0]);
	}

	//here's our initial guess to sor
	std::vector<real> gravityPotentialVec(stateVec.size());
	for (size_t i = 0; i < stateVec.size(); ++i) {
		if (app.useGravity) {
			gravityPotentialVec[i] = stateVec[i].s[0];
		} else {
			gravityPotentialVec[i] = 0.;
		}
	}
	commands.enqueueWriteBuffer(gravityPotentialBuffer, CL_TRUE, 0, sizeof(real) * volume, &gravityPotentialVec[0]);

	if (app.useFixedDT) {
		commands.enqueueWriteBuffer(dtBuffer, CL_TRUE, 0, sizeof(real), &app.fixedDT);
	}

	real2 dx;
	for (int i = 0; i < DIM; ++i) {
		dx.s[i] = (xmax.s[i] - xmin.s[i]) / (float)size.s[i];
	}
	std::cout << "xmin " << xmin.s[0] << ", " << xmin.s[1] << std::endl;
	std::cout << "xmax " << xmax.s[0] << ", " << xmax.s[1] << std::endl;
	std::cout << "size " << size.s[0] << ", " << size.s[1] << std::endl;
	std::cout << "dx " << dx.s[0] << ", " << dx.s[1] << std::endl;
		
	calcCFLKernel = cl::Kernel(program, "calcCFL");
	app.setArgs(calcCFLKernel, cflBuffer, stateBuffer, gravityPotentialBuffer, size, dx, app.cfl);

	calcCFLMinReduceKernel = cl::Kernel(program, "calcCFLMinReduce");
	app.setArgs(calcCFLMinReduceKernel, cflBuffer, cl::Local(localSizeVec(0) * sizeof(real)), volume, cflSwapBuffer);

	applyBoundaryHorizontalKernel = cl::Kernel(program, "applyBoundaryHorizontal");
	app.setArgs(applyBoundaryHorizontalKernel, stateBuffer, size);
	
	applyBoundaryVerticalKernel = cl::Kernel(program, "applyBoundaryVertical");
	app.setArgs(applyBoundaryVerticalKernel, stateBuffer, size);

	calcInterfaceVelocityKernel = cl::Kernel(program, "calcInterfaceVelocity");
	app.setArgs(calcInterfaceVelocityKernel, interfaceVelocityBuffer, stateBuffer, size, dx);

	calcFluxKernel = cl::Kernel(program, "calcFlux");
	app.setArgs(calcFluxKernel, fluxBuffer, stateBuffer, interfaceVelocityBuffer, size, dx, dtBuffer);
	
	integrateFluxKernel = cl::Kernel(program, "integrateFlux");
	app.setArgs(integrateFluxKernel, stateBuffer, fluxBuffer, size, dx, dtBuffer);
	
	poissonRelaxKernel = cl::Kernel(program, "poissonRelax");
	app.setArgs(poissonRelaxKernel, gravityPotentialBuffer, stateBuffer, size, dx);
	
	computePressureKernel = cl::Kernel(program, "computePressure");
	app.setArgs(computePressureKernel, pressureBuffer, stateBuffer, gravityPotentialBuffer, size);

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
			commands.enqueueNDRangeKernel(poissonRelaxKernel, offset2d, globalSize, localSize);
		}

		//update internal energy
		for (int i = 0; i < volume; ++i) {
			stateVec[i].s[3] += gravityPotentialVec[i];
		}
		commands.enqueueWriteBuffer(stateBuffer, CL_TRUE, 0, sizeof(real4) * volume, &stateVec[0]);
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

void BurgersSolver::update() {
	cl::CommandQueue commands = app.commands;
	cl::ImageGL fluidTexMem = app.fluidTexMem;
	cl_int2 size = app.size;
	bool useGPU = app.useGPU;
	
	applyBoundaryHorizontalKernel.setArg(2, app.boundaryMethod);
	applyBoundaryVerticalKernel.setArg(2, app.boundaryMethod);
	
	cl::NDRange offset1d(0);
	cl::NDRange offset2d(0, 0);
	cl::NDRange localSize1d(localSize[0]);
	
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

	cl::NDRange globalWidth(size.s[0]);
	cl::NDRange globalHeight(size.s[1]);
	commands.enqueueNDRangeKernel(applyBoundaryHorizontalKernel, offset1d, globalWidth, localSize1d, NULL, &applyBoundaryHorizontalEvent.clEvent);
	commands.enqueueNDRangeKernel(applyBoundaryVerticalKernel, offset1d, globalHeight, localSize1d, NULL, &applyBoundaryVerticalEvent.clEvent);
	commands.enqueueNDRangeKernel(calcInterfaceVelocityKernel, offset2d, globalSize, localSize, NULL, &calcInterfaceVelocityEvent.clEvent);
	commands.enqueueNDRangeKernel(calcFluxKernel, offset2d, globalSize, localSize, NULL, &calcFluxEvent.clEvent);
	commands.enqueueNDRangeKernel(integrateFluxKernel, offset2d, globalSize, localSize, NULL, &integrateFluxEvent.clEvent);

	if (app.useGravity) {
		//recompute poisson solution to gravitational potential
		const int maxIter = 20;
		for (int i = 0; i < maxIter; ++i) {
			commands.enqueueNDRangeKernel(poissonRelaxKernel, offset2d, globalSize, localSize, NULL, i < maxIter - 1 ? nullptr : &poissonRelaxEvent.clEvent);
		}
	}	

	commands.enqueueNDRangeKernel(applyBoundaryHorizontalKernel, offset1d, globalWidth, localSize1d, NULL, &applyBoundaryHorizontalEvent.clEvent);
	commands.enqueueNDRangeKernel(applyBoundaryVerticalKernel, offset1d, globalHeight, localSize1d, NULL, &applyBoundaryVerticalEvent.clEvent);

	commands.enqueueNDRangeKernel(computePressureKernel, offset2d, globalSize, localSize, NULL, &computePressureEvent.clEvent);
	if (app.useGravity) {
		commands.enqueueNDRangeKernel(addGravityKernel, offset2d, globalSize, localSize, NULL, &addGravityEvent.clEvent);
	}
	commands.enqueueNDRangeKernel(diffuseMomentumKernel, offset2d, globalSize, localSize, NULL, &diffuseMomentumEvent.clEvent);
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
		int volume = size.s[0] * size.s[1];
		std::vector<real4> stateVec(volume);
		commands.enqueueReadBuffer(stateBuffer, CL_TRUE, 0, sizeof(real4) * volume, &stateVec[0]);  
		std::vector<Tensor::Vector<char,4>> texVec(volume);
		for (int i = 0; i < volume; ++i) {
			real value;
			switch (app.displayMethod) {
			case DISPLAY_DENSITY:	//density
				value = stateVec[i].s[0];
				break;
			case DISPLAY_VELOCITY:	//velocity
				value = sqrt(stateVec[i].s[1] * stateVec[i].s[1] + stateVec[i].s[2] * stateVec[i].s[2]) / stateVec[i].s[0];
				break;
			case DISPLAY_PRESSURE:	//pressure
				value = (app.gamma - 1.f) * stateVec[i].s[3] * stateVec[i].s[0];
				break;
			default:
				value = .5f;
				break;
			}		
			texVec[i](0) = (char)(255.f * value * app.displayScale);
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
			std::shared_ptr<Image::Image> image = std::make_shared<Image::Image>(
				Tensor::Vector<int,2>(app.size.s[0], app.size.s[1]),
				nullptr, 3);
			
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
			std::shared_ptr<Image::ImageType<float>> image = std::make_shared<Image::ImageType<float>>(Tensor::Vector<int,2>(app.size.s[0], app.size.s[1]), nullptr, 1, 5);
			
			std::vector<real4> stateVec(app.size.s[0] * app.size.s[1]);
			app.commands.enqueueReadBuffer(stateBuffer, CL_TRUE, 0, sizeof(real4) * stateVec.size(), &stateVec[0]);
				
			std::vector<real> gravVec(app.size.s[0] * app.size.s[1]);
			app.commands.enqueueReadBuffer(gravityPotentialBuffer, CL_TRUE, 0, sizeof(real) * gravVec.size(), &gravVec[0]);
			
			app.commands.flush();
			app.commands.finish();
				
			for (int j = 0; j < app.size.s[1]; ++j) {
				for (int i = 0; i < app.size.s[0]; ++i) {
					real4 *state = &stateVec[i + app.size.s[0] * j];
					real grav = gravVec[i + app.size.s[0] * j];
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

