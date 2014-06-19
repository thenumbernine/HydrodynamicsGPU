#include "HydroGPU/Solver2D.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Image/System.h"
#include "Image/FITS_IO.h"
#include "Common/File.h"
#include "Common/Exception.h"
#include <OpenGL/gl.h>
#include <iostream>

const int DIM = 2;

Solver2D::Solver2D(
	HydroGPUApp &app_,
	const std::string &programFilename)
: app(app_)
, commands(app.commands)
, fluidTex(GLuint())
, viewZoom(1.f)
{
	cl::Device device = app.device;
	cl::Context context = app.context;
		
	stateBoundaryKernels.resize(NUM_BOUNDARY_METHODS);
	for (std::vector<cl::Kernel> &v : stateBoundaryKernels) {
		v.resize(DIM);
	}

	//get a texture going for visualizing the output
	glGenTextures(1, &fluidTex);
	glBindTexture(GL_TEXTURE_2D, fluidTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, 4, app.size.s[0], app.size.s[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	int err = glGetError();
	if (err != 0) throw Common::Exception() << "failed to create GL texture.  got error " << err;

	//hmm, my cl.hpp version only supports clCreateFromGLTexture2D, which is deprecated ... do I use the deprecated method, or do I stick with the C structures?
	// ... or do I look for a more up-to-date version of cl.hpp
	fluidTexMem = cl::ImageGL(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, fluidTex);

	size_t maxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	std::vector<size_t> maxWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
	Tensor::Vector<size_t,DIM> localSizeVec;
	for (int n = 0; n < DIM; ++n) {
		localSizeVec(n) = std::min<size_t>(maxWorkItemSizes[n], app.size.s[n]);
	}
	while (localSizeVec.volume() > maxWorkGroupSize) {
		for (int n = 0; n < DIM; ++n) {
			localSizeVec(n) = (size_t)ceil((double)localSizeVec(n) * .5);
		}
	}
	
	//hmm...
	if (!app.useGPU) localSizeVec(0) >>= 1;
	std::cout << "global_size\t" << app.size << std::endl;
	std::cout << "local_size\t" << localSizeVec << std::endl;

	globalSize = cl::NDRange(app.size.s[0], app.size.s[1]);
	globalWidth = cl::NDRange(app.size.s[0]);
	globalHeight = cl::NDRange(app.size.s[1]);
	localSize = cl::NDRange(localSizeVec(0), localSizeVec(1));
	localSize1d = cl::NDRange(localSizeVec(0));
	offset1d = cl::NDRange(0);
	offset2d = cl::NDRange(0, 0);

	std::vector<std::string> kernelSources = std::vector<std::string>{
		std::string() + "#define GAMMA " + std::to_string(app.gamma) + "f\n",
		Common::File::read("Common2D.cl"),
		Common::File::read(programFilename)
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

	int volume = app.size.s[0] * app.size.s[1];

	stateBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real4) * volume);
	cflBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume);
	cflSwapBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume / localSize[0]);
	dtBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real));
	gravityPotentialBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume);

	//get the edges, so reduction doesn't
	{
		std::vector<real> cflVec(volume);
		for (real &r : cflVec) { r = std::numeric_limits<real>::max(); }
		commands.enqueueWriteBuffer(cflBuffer, CL_TRUE, 0, sizeof(real) * volume, &cflVec[0]);
	}

	if (app.useFixedDT) {
		commands.enqueueWriteBuffer(dtBuffer, CL_TRUE, 0, sizeof(real), &app.fixedDT);
	}

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
			app.setArgs(stateBoundaryKernels[boundaryIndex][side], stateBuffer, app.size);
		}
	}

	calcCFLMinReduceKernel = cl::Kernel(program, "calcCFLMinReduce");
	app.setArgs(calcCFLMinReduceKernel, cflBuffer, cl::Local(localSizeVec(0) * sizeof(real)), volume, cflSwapBuffer);

	poissonRelaxKernel = cl::Kernel(program, "poissonRelax");
	app.setArgs(poissonRelaxKernel, gravityPotentialBuffer, stateBuffer, app.size, app.dx);
	
	addGravityKernel = cl::Kernel(program, "addGravity");
	app.setArgs(addGravityKernel, stateBuffer, gravityPotentialBuffer, app.size, app.dx, dtBuffer);

	convertToTexKernel = cl::Kernel(program, "convertToTex");
	app.setArgs(convertToTexKernel, stateBuffer, gravityPotentialBuffer, app.size, fluidTexMem, app.gradientTexMem);

	addDropKernel = cl::Kernel(program, "addDrop");
	app.setArgs(addDropKernel, stateBuffer, app.size, app.xmin, app.xmax);

	addSourceKernel = cl::Kernel(program, "addSource");
	app.setArgs(addSourceKernel, stateBuffer, app.size, app.xmin, app.xmax, dtBuffer);
}

void Solver2D::resetState(std::vector<real8> state3DVec) {
	int volume = app.size.s[0] * app.size.s[1];
	if (volume != state3DVec.size()) throw Common::Exception() << "state vec is of bad size";

	//convert 3D to 2D
	std::vector<real4> stateVec;
	for (int i = 0; i < volume; ++i) {
		real8 &state3D = state3DVec[i];
		real4 state;
		state.s[0] = state3D.s[0];
		state.s[1] = state3D.s[1];
		state.s[2] = state3D.s[2];
		state.s[3] = state3D.s[4];
		stateVec.push_back(state);
	}
	if (volume != stateVec.size()) throw Common::Exception() << "state vec is of bad size";

	//grad^2 Phi = - 4 pi G rho
	//solve inverse discretized linear system to find Psi
	//D_ij / (-4 pi G) Phi_j = rho_i
	//once you get that, plug it into the total energy
	
	//write state density first for gravity potential, to then update energy
	commands.enqueueWriteBuffer(stateBuffer, CL_TRUE, 0, sizeof(real4) * volume, &stateVec[0]);
	
	//here's our initial guess to sor
	std::vector<real> gravityPotentialVec(volume);
	for (size_t i = 0; i < volume; ++i) {
		if (app.useGravity) {
			gravityPotentialVec[i] = stateVec[i].s[0];
		} else {
			gravityPotentialVec[i] = 0.;
		}
	}
	commands.enqueueWriteBuffer(gravityPotentialBuffer, CL_TRUE, 0, sizeof(real) * volume, &gravityPotentialVec[0]);

	if (app.useGravity) {
		setPoissonRelaxRepeatArg();
		
		//solve for gravitational potential via gauss seidel
		cl::NDRange offset2d(0, 0);
		for (int i = 0; i < 20; ++i) {
			commands.enqueueNDRangeKernel(poissonRelaxKernel, offset2d, globalSize, localSize);
		}

		//update internal energy
		for (int i = 0; i < volume; ++i) {
			stateVec[i].s[3] += gravityPotentialVec[i];
		}
	}

	commands.enqueueWriteBuffer(stateBuffer, CL_TRUE, 0, sizeof(real4) * volume, &stateVec[0]);
	commands.finish();
}

Solver2D::~Solver2D() {
	glDeleteTextures(1, &fluidTex);
	
	std::cout << "OpenCL profiling info:" << std::endl;
	for (EventProfileEntry *entry : entries) {
		std::cout << entry->name << "\t" 
			<< "duration " << entry->stat
			<< std::endl;
	}
}

void Solver2D::boundary() {
	//boundary
	for (int i = 0; i < app.dim; ++i) {
		cl::NDRange globalSize1d(app.size.s[i]);
		commands.enqueueNDRangeKernel(stateBoundaryKernels[app.boundaryMethods(i)][i], offset1d, globalSize1d, localSize1d);
	}
}

void Solver2D::setPoissonRelaxRepeatArg() {
	switch (app.boundaryMethods(0)) {	//TODO per dimension
	case BOUNDARY_PERIODIC:
		poissonRelaxKernel.setArg(4, true);
		break;
	case BOUNDARY_MIRROR:
	case BOUNDARY_FREEFLOW:
		poissonRelaxKernel.setArg(4, false);
		break;
	default:
		throw Common::Exception() << "unknown boundary method " << app.boundaryMethods(0);
	}	
}

void Solver2D::initStep() {
}

void Solver2D::findMinTimestep() {
	int reduceSize = globalSize[0] * globalSize[1];
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

void Solver2D::update() {
	//CPU need to bind beforehand for roe/cpu to use it
	//GPU needs it unbound until after the update
	if (!app.useGPU) {
		glBindTexture(GL_TEXTURE_2D, fluidTex);
	}
	
	setPoissonRelaxRepeatArg();

	//commands.enqueueNDRangeKernel(addSourceKernel, offset2d, globalSize, localSize, NULL, &addSourceEvent.clEvent);

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

void Solver2D::display() {
	glFinish();
	
	std::vector<cl::Memory> acquireGLMems = {fluidTexMem};
	commands.enqueueAcquireGLObjects(&acquireGLMems);

	if (app.useGPU) {
		convertToTexKernel.setArg(5, app.displayMethod);
		convertToTexKernel.setArg(6, app.displayScale);
		commands.enqueueNDRangeKernel(convertToTexKernel, offset2d, globalSize, localSize);
	} else {
		int volume = app.size.s[0] * app.size.s[1];
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
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, app.size.s[0], app.size.s[1], GL_RGBA, GL_UNSIGNED_BYTE, &texVec[0].v);
	}

	commands.enqueueReleaseGLObjects(&acquireGLMems);
	commands.finish();
	
	glPushMatrix();
	glTranslatef(-viewPos(0), -viewPos(1), 0);
	glScalef(viewZoom, viewZoom, viewZoom);
	glBindTexture(GL_TEXTURE_2D, fluidTex);
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0,0); glVertex2f(-.5f,-.5f);
	glTexCoord2f(1,0); glVertex2f(.5f,-.5f);
	glTexCoord2f(1,1); glVertex2f(.5f,.5f);
	glTexCoord2f(0,1); glVertex2f(-.5f,.5f);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);
	glPopMatrix();

	{int err = glGetError();
	if (err) std::cout << "GL error " << err << " at " << __LINE__ << std::endl;}
}

void Solver2D::resize() {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-app.aspectRatio *.5, app.aspectRatio * .5, -.5, .5, -1., 1.);
	glMatrixMode(GL_MODELVIEW);
}

void Solver2D::addDrop() {
	addSourcePos.s[0] = mousePos(0);
	addSourcePos.s[1] = mousePos(1);
	addSourceVel.s[0] = mouseVel(0);
	addSourceVel.s[1] = mouseVel(1);
	addDropKernel.setArg(4, addSourcePos);
	addDropKernel.setArg(5, addSourceVel);
	commands.enqueueNDRangeKernel(addDropKernel, offset2d, globalSize, localSize);
}

void Solver2D::screenshot() {
	for (int i = 0; i < 1000; ++i) {
		std::string filename = std::string("screenshot") + std::to_string(i) + ".png";
		if (!Common::File::exists(filename)) {
			std::shared_ptr<Image::Image> image = std::make_shared<Image::Image>(
				Tensor::Vector<int,2>(app.size.s[0], app.size.s[1]),
				nullptr, 3);
			
			glBindTexture(GL_TEXTURE_2D, fluidTex);
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, image->getData());
			glBindTexture(GL_TEXTURE_2D, 0);
			
			Image::system->write(filename, image);
			return;
		}
	}
	throw Common::Exception() << "couldn't find an available filename";
}

void Solver2D::save() {
	for (int i = 0; i < 1000; ++i) {
		std::string filename = std::string("save") + std::to_string(i) + ".fits";
		if (!Common::File::exists(filename)) {
			save(filename);
			return;
		}
	}
	throw Common::Exception() << "couldn't find an available filename";
}

void Solver2D::save(std::string filename) {
	std::shared_ptr<Image::ImageType<float>> image = std::make_shared<Image::ImageType<float>>(Tensor::Vector<int,2>(app.size.s[0], app.size.s[1]), nullptr, 1, 5);
	
	std::vector<real4> stateVec(app.size.s[0] * app.size.s[1]);
	app.commands.enqueueReadBuffer(stateBuffer, CL_TRUE, 0, sizeof(real4) * stateVec.size(), &stateVec[0]);
		
	std::vector<real> gravVec(app.size.s[0] * app.size.s[1]);
	app.commands.enqueueReadBuffer(gravityPotentialBuffer, CL_TRUE, 0, sizeof(real) * gravVec.size(), &gravVec[0]);
	
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
}

void Solver2D::mouseMove(int x, int y, int dx, int dy) {
	mousePos(0) = (float)x / (float)app.screenSize(0) * (app.xmax.s[0] - app.xmin.s[0]) + app.xmin.s[0];
	mousePos(0) *= app.aspectRatio;	//only if xmin/xmax is symmetric. otehrwise more math required.
	mousePos(1) = (1.f - (float)y / (float)app.screenSize(1)) * (app.xmax.s[1] - app.xmin.s[1]) + app.xmin.s[1];
	mousePos += viewPos;
	mousePos /= viewZoom;
	mouseVel(0) = (float)dx / (float)app.screenSize(0);
	mouseVel(1) = (float)dy / (float)app.screenSize(1);
}

void Solver2D::mousePan(int dx, int dy) {
	viewPos += Tensor::Vector<float,2>(-(float)dx * app.aspectRatio / (float)app.screenSize(0), (float)dy / (float)app.screenSize(1));
}

void Solver2D::mouseZoom(int dz) {
	float scale = exp((float)dz * -.03f); 
	viewPos *= scale;
	viewZoom *= scale; 
}


